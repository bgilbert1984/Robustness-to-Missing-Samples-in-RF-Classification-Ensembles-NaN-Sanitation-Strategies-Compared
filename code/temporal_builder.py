# -*- coding: utf-8 -*-
"""
Temporal Feature Builder with Input Sanitation Hooks
Paper 13: NaN/Padding/Interpolation Robustness

This module creates temporal features from IQ data with configurable
sanitation strategies and optional mask channels.
"""

import numpy as np
from .sanitize_io import sanitize_complex, build_nan_mask

def _create_temporal_input(
    iq: np.ndarray,
    *,
    sanitize_mode: str = "nan_to_num",
    target_len: int = 128,
    mask_channel: bool = False
) -> np.ndarray:
    """
    Create temporal features from IQ data with sanitation.
    
    Args:
        iq: Complex IQ data
        sanitize_mode: Sanitation strategy
        target_len: Target sequence length
        mask_channel: Whether to include mask as additional channel
        
    Returns:
        Features array [T, 2] or [T, 3] if mask_channel=True
    """
    # Build mask before sanitation (true NaN spans)
    mask = build_nan_mask(iq)
    # Sanitize IQ for numeric stability
    iq = sanitize_complex(iq, mode=sanitize_mode)

    # pad/trim both iq and mask to target_len
    if iq.shape[0] < target_len:
        pad = target_len - iq.shape[0]
        iq  = np.pad(iq,  (0, pad), mode="constant")
        mask = np.pad(mask,(0, pad), mode="constant")
    elif iq.shape[0] > target_len:
        iq   = iq[:target_len]
        mask = mask[:target_len]

    feat = np.stack([iq.real.astype(np.float32), iq.imag.astype(np.float32)], axis=1)  # [T,2]
    if mask_channel or sanitize_mode == "mask_preserve":
        feat = np.concatenate([feat, mask[:, None].astype(np.float32)], axis=1)        # [T,3]
    return feat

def create_temporal_features(signal, **kwargs):
    """
    Main entry point for temporal feature creation.
    
    Args:
        signal: Signal object with iq_data and metadata
        **kwargs: Additional parameters for _create_temporal_input
        
    Returns:
        Temporal features array
    """
    # Extract sanitation mode from metadata if available
    metadata = getattr(signal, 'metadata', {})
    sanitize_mode = metadata.get('sanitize_mode', kwargs.get('sanitize_mode', 'nan_to_num'))
    mask_channel = metadata.get('mask_channel', kwargs.get('mask_channel', False))
    target_len = kwargs.get('target_len', 128)
    
    return _create_temporal_input(
        signal.iq_data,
        sanitize_mode=sanitize_mode,
        target_len=target_len,
        mask_channel=mask_channel
    )