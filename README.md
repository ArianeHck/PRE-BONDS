# PRE-BONDS
# fNIRS Hyperscanning Preprocessing Pipeline Comparison

This repository contains the preprocessing and analysis scripts developed as part of an M2 internship at LaPsyDÉ (Université Paris Cité), under the supervision of Iris Menu, in the context of the PRE-BONDS project.

## Overview

This project compares 4 fNIRS preprocessing pipelines on a parent-child hyperscanning dataset (Bizzego et al., 2022) and evaluates their impact on signal quality and neural synchrony measures (Wavelet Transform Coherence, WTC).

## Branches

- `matlab` — Homer3/MATLAB pipeline (P1: Spline + Homer3 wavelet)
- `python` — Python pipelines (P2, P3, P4) implemented with MNE, PyWavelets and pycwt

## Pipelines

| Pipeline | Method | Tools |
|----------|--------|-------|
| P1 | Homer3 wavelet + Spline | MATLAB / Homer3 |
| P2 | Homer3-like wavelet, no Spline, no TDDR | Python / MNE, PyWavelets |
| P3 | Donoho wavelet (db4) + TDDR | Python / MNE, PyWavelets |
| P4 | Homer3-like wavelet + TDDR | Python / MNE, PyWavelets |

## Dataset

Bizzego, A., Gabrieli, G., Azhari, A., Setoh, P., & Esposito, G. (2022). Dataset of parent-child hyperscanning functional near-infrared spectroscopy recordings. *Scientific Data*, 9, 625. https://doi.org/10.1038/s41597-022-01751-2

## Requirements (Python)

mne
pywt
pycwt
scipy
numpy
pandas

## Contact

LaPsyDÉ — Université Paris Cité
