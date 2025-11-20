# -*- coding: utf-8 -*-
"""
Input Sanitation Helpers for NaN/Padding/Interpolation Robustness
Paper 13: Quantifying the impact of input corruption and sanitation strategies

This module provides robust input sanitation methods for temporal and spectral
feature builders, including mask generation and statistical analysis.
"""

import numpy as np

def _linear_interp_1d(x: np.ndarray) -> np.ndarray:
    """Linear-interpolate NaNs in 1D; leaves leading/trailing NaNs to be padded."""
    x = x.astype(np.float32, copy=True)
    n = x.size
    isnan = np.isnan(x)
    if not isnan.any():
        return x
    idx = np.arange(n, dtype=np.float32)
    # valid points
    good = ~isnan
    if good.sum() == 0:
        return np.zeros_like(x)
    x[isnan] = np.interp(idx[isnan], idx[good], x[good])
    return x

def build_nan_mask(iq_complex: np.ndarray) -> np.ndarray:
    """1D float32 mask: 1 where either I or Q is NaN, else 0."""
    return ((np.isnan(iq_complex.real)) | (np.isnan(iq_complex.imag))).astype(np.float32)

def mask_run_stats(mask: np.ndarray):
    """
    Returns:
      nan_fraction: mean(mask)
      longest_run:  longest contiguous run of 1s
      run_count:    number of contiguous 1-runs
    """
    if mask.size == 0:
        return 0.0, 0, 0
    m = (mask.astype(np.uint8) == 1)
    if not m.any():
        return 0.0, 0, 0
    # transitions where runs start/end
    dm = np.diff(np.concatenate(([0], m.view(np.int8), [0])))
    starts = np.where(dm == 1)[0]
    ends   = np.where(dm == -1)[0]
    lengths = (ends - starts)
    return float(m.mean()), int(lengths.max() if lengths.size else 0), int(len(lengths))

def sanitize_temporal(iq_complex: np.ndarray, mode: str = "nan_to_num") -> np.ndarray:
    """
    Sanitize a complex IQ vector in time domain.
    mode ∈ {"none","nan_to_num","interp_lin","zero_pad","mask_preserve"}
    - "nan_to_num": np.nan_to_num on real/imag
    - "interp_lin": linear interpolate NaNs, then pad edge-NaNs with edge values
    - "zero_pad": replace NaNs with 0
    - "mask_preserve": zero-fill NaNs for downstream math; true mask provided separately
    """
    if mode == "none":
        return iq_complex

    re, im = iq_complex.real.copy(), iq_complex.imag.copy()
    if mode == "nan_to_num":
        re = np.nan_to_num(re, nan=0.0, posinf=0.0, neginf=0.0)
        im = np.nan_to_num(im, nan=0.0, posinf=0.0, neginf=0.0)
    elif mode == "interp_lin":
        re = _linear_interp_1d(re); im = _linear_interp_1d(im)
        # If any NaNs remain (e.g., all-NaN), zero them
        re = np.nan_to_num(re, nan=0.0); im = np.nan_to_num(im, nan=0.0)
    elif mode == "zero_pad" or mode == "mask_preserve":
        re = np.where(np.isnan(re), 0.0, re)
        im = np.where(np.isnan(im), 0.0, im)
    else:
        raise ValueError(f"Unknown temporal sanitize mode: {mode}")
    return re.astype(np.float32) + 1j * im.astype(np.float32)

def sanitize_spectral(iq_complex: np.ndarray, mode: str = "nan_to_num") -> np.ndarray:
    """
    Pre-FFT sanitation mirror; same modes as temporal. Apply BEFORE FFT to prevent NaNs → NaNs in PSD.
    """
    return sanitize_temporal(iq_complex, mode=mode)

def sanitize_complex(iq_complex: np.ndarray, mode: str) -> np.ndarray:
    """
    General complex sanitation function (alias for compatibility).
    mode ∈ {"none","nan_to_num","interp_lin","zero_pad","mask_preserve"}
    """
    return sanitize_temporal(iq_complex, mode=mode)