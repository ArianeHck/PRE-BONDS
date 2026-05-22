import os
import glob
import warnings
import re


import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import mne
import pandas as pd
from mne.preprocessing.nirs import (optical_density,
                                      temporal_derivative_distribution_repair,
                                      beer_lambert_law,
                                      scalp_coupling_index)
import pywt
import pycwt

# =============================================================================
# PARAMETRES
# =============================================================================

# Pipeline 1 : exports Homer3
BASE_MAT_BY_CONDITION = {
    "play":  r"C:\Users\LaPsyDe\Documents\bizzego data\final_analysis\play\exported",
    "video": r"C:\Users\LaPsyDe\Documents\bizzego data\final_analysis\video\exported",
}

# Pipelines Python : redirection vers les dossiers contenant les .snirf bruts.
BASE_SNIRF_BY_CONDITION = {
    "play": r"C:\Users\LaPsyDe\Documents\bizzego data\final_analysis\play",
    "video": r"C:\Users\LaPsyDe\Documents\bizzego data\final_analysis\video",
}

CONDITIONS = ["play", "video"]
ROLES = ["parent", "child"]

SCI_THRESH = 0.70
CV_THRESH = 7.50
CVW_THRESH = 5.00

FILTER_LOW = 0.01
FILTER_HIGH = 0.50

FREQ_MIN = 0.02
FREQ_MAX = 0.10

OUTPUT_DIR = "outputs_pipeline_comparison"


# =============================================================================
# UTILITAIRES CHEMINS / CHARGEMENT
# =============================================================================

def get_ppf(role):
    if role == "child":
        return (5.449, 4.387)
    return (6.432, 5.370)


def find_snirf_path(condition, dyade_id, role):
    base = BASE_SNIRF_BY_CONDITION[condition]

    if condition == "play":
        dataverse_patterns = [
            os.path.join(base, "mothers", "mother_copy", f"*{dyade_id}*{role}*.snirf"),
            os.path.join(base, "mothers", "mothers", dyade_id, role, "*.snirf"),
            os.path.join(base, "*", "*", dyade_id, role, "*.snirf"),
            os.path.join(base, "*", "*", f"*{dyade_id}*{role}*.snirf"),
            os.path.join(base, "**", dyade_id, role, "*.snirf"),
            os.path.join(base, "**", f"*{dyade_id}*{role}*.snirf"),
        ]
        id_regex = re.compile(rf"(?<![A-Za-z0-9]){re.escape(dyade_id)}(?![A-Za-z0-9])")
        role_regex = re.compile(rf"(?<![A-Za-z0-9]){re.escape(role)}(?![A-Za-z0-9])")
        for pattern in dataverse_patterns:
            matches = sorted({
                path for path in glob.glob(pattern, recursive=True)
                if (
                    id_regex.search(os.path.basename(path)) or id_regex.search(path)
                ) and (
                    role_regex.search(os.path.basename(path)) or role_regex.search(path)
                )
            })
            if matches:
                return matches[0]

    # Recherche non récursive par nom, avec verification stricte de l'identifiant, pour ne pas confondre NSFD2 et NSFD20
    patterns = [
        os.path.join(base, f"*{condition}*{dyade_id}*{role}*.snirf"),
        os.path.join(base, f"*{dyade_id}*{role}*.snirf"),
    ]
    matches = []
    for pattern in patterns:
        matches.extend(glob.glob(pattern))
    id_regex = re.compile(rf"(?<![A-Za-z0-9]){re.escape(dyade_id)}(?![A-Za-z0-9])")
    matches = sorted({
        path for path in matches
        if id_regex.search(os.path.basename(path)) or id_regex.search(path)
    })
    if not matches:
        return None
    return matches[0]


def find_homer_path(condition, dyade_id, role):
    base = BASE_MAT_BY_CONDITION[condition]
    pattern = os.path.join(base, f"{dyade_id}_{role}_*.mat")
    matches = glob.glob(pattern)
    id_regex = re.compile(rf"(?<![A-Za-z0-9]){re.escape(dyade_id)}(?![A-Za-z0-9])")
    matches = [p for p in matches if id_regex.search(os.path.basename(p))]
    return matches[0] if matches else None


def get_available_dyades():
    dyades = set()
    for condition in CONDITIONS:
        for mat_file in glob.glob(os.path.join(BASE_MAT_BY_CONDITION[condition], "*_parent_*.mat")):
            basename = os.path.basename(mat_file)
            match = re.match(r"^(NSFD\d+)_parent_", basename)
            if not match:
                continue
            dyade_id = match.group(1)
            has_child_mat    = find_homer_path(condition, dyade_id, "child") is not None
            has_parent_snirf = find_snirf_path(condition, dyade_id, "parent") is not None
            has_child_snirf  = find_snirf_path(condition, dyade_id, "child") is not None
            if has_child_mat and has_parent_snirf and has_child_snirf:
                dyades.add(dyade_id)
    return sorted(dyades)


def load_homer_mat(condition, dyade_id, role):
    path = find_homer_path(condition, dyade_id, role)
    if path is None:
        return None, None, None

    mat = scipy.io.loadmat(path, simplify_cells=True)
    hbo = mat["hbo_homer"]
    time = np.squeeze(mat["time_homer"]) if "time_homer" in mat else None

    if hbo.ndim == 1:
        hbo = hbo[:, np.newaxis]
    if hbo.shape[0] < hbo.shape[1]:
        hbo = hbo.T

    # Conversion des données en fonction de ce qu'on récupère par Homer3
    nonzero = np.abs(hbo[np.abs(hbo) > 1e-12])
    if nonzero.size:
        median_val = np.median(nonzero)
        if median_val < 1e-3: 
            hbo = hbo * 1e6
        elif median_val < 1.0:
            hbo = hbo * 1e3 

    sfreq = None
    if time is not None and len(time) > 1:
        sfreq = 1.0 / np.median(np.diff(time))

    return hbo, time, sfreq


def read_raw_snirf(condition, dyade_id, role):
    path = find_snirf_path(condition, dyade_id, role)
    if path is None:
        return None, None
    raw = mne.io.read_raw_snirf(path, preload=True, verbose=False)
    return raw, path


# =============================================================================
# QUALITE SIGNAL
# =============================================================================

def compute_pre_quality(raw):
    """Qualite avant preprocessing sur intensite / densite optique."""
    raw_od = optical_density(raw.copy())
    sci = scalp_coupling_index(raw_od)
    intensity = raw.get_data()

    n_pairs = min(len(raw.ch_names) // 2, 20)
    cv_760 = np.array([
        intensity[ch].std() / (intensity[ch].mean() + 1e-12) * 100
        for ch in range(n_pairs)
    ])
    cv_850 = np.array([
        intensity[ch + n_pairs].std() / (intensity[ch + n_pairs].mean() + 1e-12) * 100
        for ch in range(n_pairs)
    ])
    cvw = (cv_760 + cv_850) / 2

    mask_sci = sci[:n_pairs] >= SCI_THRESH
    mask_cv = (cv_760 < CV_THRESH) & (cv_850 < CV_THRESH)
    mask_cvw = cvw < CVW_THRESH
    good_idx = np.where(mask_sci & mask_cv & mask_cvw)[0]
    good_od_idx = np.concatenate([good_idx, good_idx + n_pairs])

    quality_rows = []
    for ch in range(n_pairs):
        quality_rows.append({
            "channel": ch + 1,
            "pre_good": bool(ch in good_idx),
            "sci": float(sci[ch]),
            "cv_760": float(cv_760[ch]),
            "cv_850": float(cv_850[ch]),
            "cvw": float(cvw[ch]),
        })

    return good_idx, good_od_idx, quality_rows


def compute_post_quality(hbo, channels):
    rows = []
    if hbo is None:
        return rows

    for ch in channels:
        sig = hbo[:, ch]
        finite = np.isfinite(sig)
        if finite.any():
            sig_finite = sig[finite]
            post_std = np.std(sig_finite)
            post_range = np.ptp(sig_finite)
            post_abs_p99 = np.percentile(np.abs(sig_finite), 99)
        else:
            post_std = np.nan
            post_range = np.nan
            post_abs_p99 = np.nan

        rows.append({
            "channel": ch + 1,
            "post_std": float(post_std) if np.isfinite(post_std) else np.nan,
            "post_range": float(post_range) if np.isfinite(post_range) else np.nan,
            "post_abs_p99": float(post_abs_p99) if np.isfinite(post_abs_p99) else np.nan,
            "post_nan_pct": float(100 * np.mean(~finite)),
        })
    return rows


# =============================================================================
# WAVELET HOMER3-LIKE
# =============================================================================

def normalization_noise(y, qmf_filter):
    n = len(y)
    c = np.real(np.fft.ifft(np.fft.fft(y, n) * np.fft.fft(qmf_filter, n)))
    y_downsampled = c[::2]
    mad = np.median(np.abs(y_downsampled - np.median(y_downsampled)))
    if mad != 0:
        y_norm = (1 / 1.4826) * y / mad
        coeff = 1 / (1.4826 * mad)
    else:
        y_norm = y.copy()
        coeff = 1.0
    return y_norm, coeff


def wt_inv(x, L, N, wavename="db2"):
    D = N - L
    n = len(x)
    wp = np.zeros((n, D + 1))
    wp[:, 0] = x.copy()

    for d in range(D):
        n_blocks = 2**d
        l_blocks = n // n_blocks
        for b in range(n_blocks):
            s = wp[b * l_blocks:(b + 1) * l_blocks, 0].copy()
            s_shift = np.concatenate([[s[-1]], s[:-1]])
            cA, cD = pywt.dwt(s, wavename, mode="periodization")
            cA_shift, cD_shift = pywt.dwt(s_shift, wavename, mode="periodization")
            half = l_blocks // 2
            wp[b * l_blocks:b * l_blocks + half, 0] = cA
            wp[b * l_blocks + half:(b + 1) * l_blocks, 0] = cA_shift
            wp[b * l_blocks:b * l_blocks + half, d + 1] = cD
            wp[b * l_blocks + half:(b + 1) * l_blocks, d + 1] = cD_shift

    return wp


def iwt_inv(stat_wt, wavename="db2"):
    n = stat_wt.shape[0]
    wp = stat_wt.copy()
    D = stat_wt.shape[1] - 1

    for d in range(D - 1, -1, -1):
        n_blocks = 2**d
        l_blocks = n // n_blocks
        for b in range(n_blocks):
            half = l_blocks // 2
            cA = wp[b * l_blocks:b * l_blocks + half, 0]
            cA_shift = wp[b * l_blocks + half:(b + 1) * l_blocks, 0]
            cD = wp[b * l_blocks:b * l_blocks + half, d + 1]
            cD_shift = wp[b * l_blocks + half:(b + 1) * l_blocks, d + 1]
            rec = pywt.idwt(cA, cD, wavename, mode="periodization")
            rec_shift = pywt.idwt(cA_shift, cD_shift, wavename, mode="periodization")
            rec_shift_back = np.concatenate([rec_shift[1:], [rec_shift[0]]])
            wp[b * l_blocks:(b + 1) * l_blocks, 0] = (rec + rec_shift_back) / 2

    return wp[:, 0]


def wavelet_analysis(stat_wt, L, wavename, iqr_factor, signal_length):
    n = stat_wt.shape[0]
    N = int(np.log2(n))
    wp = stat_wt.copy()
    signal_length_tmp = signal_length

    for j in range(1, N - L):
        signal_length_tmp = signal_length_tmp // 2
        n_blocks = 2**j
        l_blocks = n // n_blocks
        for b in range(n_blocks):
            sr = wp[b * l_blocks:(b + 1) * l_blocks, j].copy()
            sr_temp = sr[:signal_length_tmp]
            q25, q75 = np.percentile(sr_temp, [25, 75])
            iqr = q75 - q25
            upper = q75 + iqr * iqr_factor
            lower = q25 - iqr * iqr_factor
            sr[(sr > upper) | (sr < lower)] = 0
            wp[b * l_blocks:(b + 1) * l_blocks, j] = sr

    return iwt_inv(wp, wavename), wp


def homer3_wavelet_correct(signal, iqr_factor, wavename="db2", L=4):
    signal_length = len(signal)
    N = int(np.ceil(np.log2(signal_length)))
    data_padded = np.zeros(2**N)
    data_padded[:signal_length] = signal
    dc_val = np.mean(data_padded)
    data_padded = data_padded - dc_val

    wavelet = pywt.Wavelet(wavename)
    qmf_filter = np.array(wavelet.dec_hi)
    y_norm, norm_coeff = normalization_noise(data_padded, qmf_filter)
    stat_wt = wt_inv(y_norm, L, N, wavename)
    ar_signal, _ = wavelet_analysis(stat_wt, L, wavename, iqr_factor, signal_length)
    return (ar_signal / norm_coeff + dc_val)[:signal_length]


# =============================================================================
# PIPELINES PYTHON
# =============================================================================

def run_python_pipelines(raw, good_od_idx, role):
    ppf = get_ppf(role)
    results = {}
    n_pairs = len(raw.ch_names) // 2
    good_idx_hbo = good_od_idx[good_od_idx < n_pairs]
    iqr_factor = 0.8 if role == "child" else 1.5    

    # Pipeline 2 : Py-H3 no Spline.
    # OD -> Wavelet Homer3 -> bandpass -> HbO (sans Spline, sans TDDR)
    raw_od_h3 = optical_density(raw.copy())
    
    data_h3 = raw_od_h3.get_data().copy()
    for ch in good_od_idx:
        data_h3[ch] = homer3_wavelet_correct(data_h3[ch], iqr_factor=iqr_factor)
    raw_od_h3._data = data_h3

    raw_od_h3.filter(FILTER_LOW, FILTER_HIGH, method="iir", verbose=False)
    hbo_all = beer_lambert_law(raw_od_h3, ppf=ppf).pick("hbo").get_data().T * 1e6
    hbo_full = np.full_like(hbo_all, np.nan)
    hbo_full[:, good_idx_hbo] = hbo_all[:, good_idx_hbo]
    results["P2_PyH3_noSpline"] = hbo_full

    # Pipeline 3 : Wavelet Python indépendante + TDDR.
    raw_od_v1 = optical_density(raw.copy())
    
    data_v1 = raw_od_v1.get_data().copy()
    for ch in good_od_idx:
        coeffs = pywt.wavedec(data_v1[ch], "db4", level=4)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        thr = sigma * np.sqrt(2 * np.log(len(data_v1[ch])))
        coeffs_thresh = [coeffs[0]] + [
            pywt.threshold(c, thr, mode="soft") for c in coeffs[1:]
        ]
        data_v1[ch] = pywt.waverec(coeffs_thresh, "db4")[:data_v1.shape[1]]
    raw_od_v1._data = data_v1
    raw_od_v1 = temporal_derivative_distribution_repair(raw_od_v1)
    raw_od_v1.filter(FILTER_LOW, FILTER_HIGH, method="iir", verbose=False)
    hbo_all = beer_lambert_law(raw_od_v1, ppf=ppf).pick("hbo").get_data().T * 1e6
    # hbo_all a n_pairs colonnes car raw_od_v1 n'a pas été drop_channels
    # good_idx contient les indices des bons canaux dans [0, n_pairs)
    hbo_full = np.full_like(hbo_all, np.nan)
    hbo_full[:, good_idx_hbo] = hbo_all[:, good_idx_hbo]
    results["P3_PyV1_Wavelet_TDDR"] = hbo_full

    # Pipeline 4 : Wavelet Homer3-like + TDDR.
    raw_od_v2 = optical_density(raw.copy())
    
    data_v2 = raw_od_v2.get_data().copy()
    for ch in good_od_idx:
        data_v2[ch] = homer3_wavelet_correct(data_v2[ch], iqr_factor=iqr_factor)
    raw_od_v2._data = data_v2
    raw_od_v2 = temporal_derivative_distribution_repair(raw_od_v2)
    raw_od_v2.filter(FILTER_LOW, FILTER_HIGH, method="iir", verbose=False)
    hbo_all = beer_lambert_law(raw_od_v2, ppf=ppf).pick("hbo").get_data().T * 1e6
    
    hbo_full = np.full_like(hbo_all, np.nan)
    hbo_full[:, good_idx_hbo] = hbo_all[:, good_idx_hbo]
    results["P4_PyV2_H3Wavelet_TDDR"] = hbo_full

    return results


# =============================================================================
# WTC
# =============================================================================

def wtc_channel(sig1, sig2, sfreq):
    dt = 1.0 / sfreq
    n = min(len(sig1), len(sig2))
    s1 = sig1[:n]
    s2 = sig2[:n]

    s1 = (s1 - np.nanmean(s1)) / (np.nanstd(s1) + 1e-12)
    s2 = (s2 - np.nanmean(s2)) / (np.nanstd(s2) + 1e-12)

    WCT, _, coi, freq, _ = pycwt.wct(
        s1,
        s2,
        dt,
        dj=1 / 12,
        s0=2 * dt,
        J=-1,
        sig=False,
        significance_level=0.95,
    )

    freq_mask = (freq >= FREQ_MIN) & (freq <= FREQ_MAX)
    if freq_mask.sum() == 0:
        return np.nan

    wtc_map = WCT[freq_mask, :].copy()
    for i, period in enumerate(1.0 / freq[freq_mask]):
        wtc_map[i, coi < period] = np.nan

    if np.isnan(wtc_map).all():
        return np.nan
    return float(np.nanmean(wtc_map))


def compute_pipeline_wtc(hbo_parent, hbo_child, common_idx, sfreq):
    n = min(hbo_parent.shape[0], hbo_child.shape[0])
    vals = []
    for ch in common_idx:
        vals.append(wtc_channel(hbo_parent[:n, ch], hbo_child[:n, ch], sfreq))
    return np.array(vals, dtype=float)


# =============================================================================
# BOUCLE PRINCIPALE
# =============================================================================

def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dyades = get_available_dyades()
    print(f"Dyades disponibles : {dyades}")

    wtc_rows = []
    channel_wtc_rows = []
    quality_rows = []
    skipped_rows = []

    for condition in CONDITIONS:
        print(f"\n{'=' * 70}")
        print(f"CONDITION : {condition.upper()}")
        print(f"{'=' * 70}")

        for dyade_id in dyades:
            print(f"\n{dyade_id}")

            raw_by_role = {}
            sfreq_by_role = {}
            good_by_role = {}
            good_od_by_role = {}

            missing = False
            for role in ROLES:
                raw, snirf_path = read_raw_snirf(condition, dyade_id, role)
                if raw is None:
                    print(f"  {role:<6}: snirf manquant")
                    skipped_rows.append({
                        "condition": condition,
                        "dyade": dyade_id,
                        "reason": f"snirf manquant {role}",
                    })
                    missing = True
                    break

                raw_by_role[role] = raw
                sfreq_by_role[role] = raw.info["sfreq"]

                good_idx, good_od_idx, pre_rows = compute_pre_quality(raw)
                good_by_role[role] = good_idx
                good_od_by_role[role] = good_od_idx

                for row in pre_rows:
                    row.update({
                        "condition": condition,
                        "dyade": dyade_id,
                        "role": role,
                        "pipeline": "pre_raw",
                        "snirf_path": snirf_path,
                    })
                    quality_rows.append(row)

                print(f"  {role:<6}: {len(good_idx):>2}/20 bons canaux")

            if missing:
                continue

            MIN_COMMON_CHANNELS = 2   
            common_idx = np.intersect1d(good_by_role["parent"], good_by_role["child"])
            print(f"  communs: {len(common_idx):>2}/20 -> {common_idx + 1}")
            if len(common_idx) < MIN_COMMON_CHANNELS:
                print(f"  -> ignoré : seulement {len(common_idx)} canaux communs (min={MIN_COMMON_CHANNELS})")
                skipped_rows.append({
                    "condition": condition,
                    "dyade": dyade_id,
                    "reason": f"canaux communs insuffisants ({len(common_idx)}<{MIN_COMMON_CHANNELS})",
                })
                continue

            hbo_by_role = {"parent": {}, "child": {}}

            for role in ROLES:
                hbo_homer, _, sfreq_homer = load_homer_mat(condition, dyade_id, role)
                if hbo_homer is not None:
                    hbo_by_role[role]["P1_Homer3_Spline"] = hbo_homer
                    if sfreq_homer is not None:
                        sfreq_by_role[f"{role}_homer"] = sfreq_homer

                py_results = run_python_pipelines(
                    raw_by_role[role],
                    good_od_by_role[role],
                    role,
                )
                hbo_by_role[role].update(py_results)

                for pipeline, hbo in hbo_by_role[role].items():
                    for row in compute_post_quality(hbo, common_idx):
                        row.update({
                            "condition": condition,
                            "dyade": dyade_id,
                            "role": role,
                            "pipeline": pipeline,
                        })
                        quality_rows.append(row)

            pipelines = [
                "P1_Homer3_Spline",
                "P2_PyH3_noSpline",
                "P3_PyV1_Wavelet_TDDR",
                "P4_PyV2_H3Wavelet_TDDR",
            ]

            sfreq_map = {
                    "P1_Homer3_Spline": sfreq_by_role.get("parent_homer") or sfreq_by_role.get("parent"),
                    "P2_PyH3_noSpline": sfreq_by_role.get("parent"),
                    "P3_PyV1_Wavelet_TDDR": sfreq_by_role.get("parent"),
                    "P4_PyV2_H3Wavelet_TDDR": sfreq_by_role.get("parent"),
                }

            for pipeline in pipelines:
                    hbo_parent = hbo_by_role["parent"].get(pipeline)
                    hbo_child = hbo_by_role["child"].get(pipeline)
                    if hbo_parent is None or hbo_child is None:
                        skipped_rows.append({
                            "condition": condition,
                            "dyade": dyade_id,
                            "pipeline": pipeline,
                            "reason": "donnees pipeline manquantes",
                        })
                        continue

                    sfreq = sfreq_map.get(pipeline)
                    vals = compute_pipeline_wtc(hbo_parent, hbo_child, common_idx, sfreq)
                    mean = np.nanmean(vals)
                    std = np.nanstd(vals)
                    sem = std / np.sqrt(np.sum(np.isfinite(vals))) if np.isfinite(vals).any() else np.nan


                    wtc_rows.append({
                        "condition": condition,
                        "dyade": dyade_id,
                        "pipeline": pipeline,
                        "n_common": int(len(common_idx)),
                        "common_channels": ",".join(map(str, common_idx + 1)),
                        "wtc_mean": float(mean) if np.isfinite(mean) else np.nan,
                        "wtc_std": float(std) if np.isfinite(std) else np.nan,
                        "wtc_sem": float(sem) if np.isfinite(sem) else np.nan,
                        "freq_min": FREQ_MIN,
                        "freq_max": FREQ_MAX,
                    })

                    for ch, val in zip(common_idx, vals):
                        channel_wtc_rows.append({
                            "condition": condition,
                            "dyade": dyade_id,
                            "pipeline": pipeline,
                            "channel": int(ch + 1),
                            "wtc": float(val) if np.isfinite(val) else np.nan,
                        })

                    print(f"  {pipeline:<25} WTC={mean:.4f} ({len(common_idx)} ch)")


    # Garder seulement les dyades présentes dans les 2 conditions
    wtc_df         = pd.DataFrame(wtc_rows)
    channel_wtc_df = pd.DataFrame(channel_wtc_rows)
    quality_df     = pd.DataFrame(quality_rows)
    skipped_df     = pd.DataFrame(skipped_rows)

    # PUIS filtrer
    dyades_both = (
        wtc_df.groupby("dyade")["condition"]
        .nunique()
        .pipe(lambda s: s[s == 2].index)
    )
    wtc_df         = wtc_df[wtc_df.dyade.isin(dyades_both)]
    channel_wtc_df = channel_wtc_df[channel_wtc_df.dyade.isin(dyades_both)]
    quality_df     = quality_df[quality_df.dyade.isin(dyades_both)]

    wtc_df.to_csv(os.path.join(OUTPUT_DIR, "wtc_summary_by_dyade_pipeline.csv"), index=False)
    channel_wtc_df.to_csv(os.path.join(OUTPUT_DIR, "wtc_by_channel.csv"), index=False)
    quality_df.to_csv(os.path.join(OUTPUT_DIR, "quality_pre_post.csv"), index=False)
    skipped_df.to_csv(os.path.join(OUTPUT_DIR, "skipped_cases.csv"), index=False)

    xlsx_path = os.path.join(OUTPUT_DIR, "pipeline_comparison_results.xlsx")
    try:
        with pd.ExcelWriter(xlsx_path) as writer:
            wtc_df.to_excel(writer, sheet_name="wtc_summary", index=False)
            channel_wtc_df.to_excel(writer, sheet_name="wtc_channels", index=False)
            quality_df.to_excel(writer, sheet_name="quality_pre_post", index=False)
            skipped_df.to_excel(writer, sheet_name="skipped", index=False)
    except ModuleNotFoundError as err:
        if err.name == "openpyxl":
            print("\nExport Excel ignore : module openpyxl non installe.")
            print("Les exports CSV ont bien ete crees et suffisent pour l'analyse.")
        else:
            raise

    print(f"\nExports termines dans : {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
