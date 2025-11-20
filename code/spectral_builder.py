# -*- coding: utf-8 -*-
"""
Spectral Feature Builder with Input Sanitation Hooks
Paper 13: NaN/Padding/Interpolation Robustness

This module creates spectral features (PSD) from IQ data with configurable
sanitation strategies and optional mask channels.
"""

import numpy as np
from .sanitize_io import sanitize_complex, build_nan_mask

def _resample_mask(mask: np.ndarray, nfft: int) -> np.ndarray:
    """Nearest-neighbor resample 1D mask to length nfft (fast + sufficient)."""
    if mask.size == nfft:
        return mask.astype(np.float32)
    if mask.size == 0:
        return np.zeros(nfft, dtype=np.float32)
    idx = (np.linspace(0, mask.size-1, nfft)).astype(np.int32)
    idx = np.clip(idx, 0, mask.size-1)  # Ensure valid indices
    return mask[idx].astype(np.float32)

def _create_spectral_input(
    iq: np.ndarray,
    *,
    sanitize_mode: str = "nan_to_num",
    nfft: int = 256,
    mask_channel: bool = False
) -> np.ndarray:
    """
    Create spectral features from IQ data with sanitation.
    
    Args:
        iq: Complex IQ data
        sanitize_mode: Sanitation strategy  
        nfft: FFT size
        mask_channel: Whether to include resampled mask as additional channel
        
    Returns:
        Features array [nfft,] or [nfft, 2] if mask_channel=True
    """
    # Mask from raw (pre-sanitization) and sanitize for FFT
    mask = build_nan_mask(iq)
    iq   = sanitize_complex(iq, mode=sanitize_mode)

    # window + pad/trim
    x = iq[:nfft] if len(iq) >= nfft else np.pad(iq, (0, nfft - len(iq)), mode="constant")
    win = np.hanning(nfft).astype(np.float32)
    X = np.fft.fftshift(np.fft.fft(x * win))
    psd = (np.abs(X) ** 2).astype(np.float32)
    psd /= (psd.max() + 1e-8)
    
    if mask_channel or sanitize_mode == "mask_preserve":
        m = _resample_mask(mask, nfft)
        feat = np.stack([psd, m], axis=1)  # [nfft,2]
    else:
        feat = psd                          # [nfft,]
    return feat

def create_spectral_features(signal, **kwargs):
    """
    Main entry point for spectral feature creation.
    
    Args:
        signal: Signal object with iq_data and metadata
        **kwargs: Additional parameters for _create_spectral_input
        
    Returns:
        Spectral features array
    """
    # Extract sanitation mode from metadata if available
    metadata = getattr(signal, 'metadata', {})
    sanitize_mode = metadata.get('sanitize_mode', kwargs.get('sanitize_mode', 'nan_to_num'))
    mask_channel = metadata.get('mask_channel', kwargs.get('mask_channel', False))
    nfft = kwargs.get('nfft', 256)
    
    return _create_spectral_input(
        signal.iq_data,
        sanitize_mode=sanitize_mode,
        nfft=nfft,
        mask_channel=mask_channel
    )