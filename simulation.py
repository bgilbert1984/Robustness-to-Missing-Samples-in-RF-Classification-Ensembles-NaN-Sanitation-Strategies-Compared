#!/usr/bin/env python3
"""
Simple simulation module for testing NaN/Padding/Interpolation robustness
This provides a minimal dataset iterator for Paper 13 evaluation.
"""

import numpy as np
from dataclasses import dataclass
from typing import Iterator, Dict, Any
import time

@dataclass
class TestSignal:
    """Simple signal container for testing."""
    id: str
    iq_data: np.ndarray
    classification: str
    metadata: Dict[str, Any]
    sample_rate_hz: float = 1e6
    center_freq_hz: float = 100e6
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

def generate_test_signal(signal_id: int, modulation: str, snr_db: float, samples: int = 512) -> TestSignal:
    """Generate a simple test signal with specified characteristics."""
    
    # Simple modulation schemes
    if modulation == "BPSK":
        # Generate BPSK signal
        bits = np.random.randint(0, 2, samples // 4)
        symbols = 2 * bits - 1  # Map to +1/-1
        iq_data = np.repeat(symbols, 4).astype(np.complex64)
    elif modulation == "QPSK":
        # Generate QPSK signal
        bits_i = np.random.randint(0, 2, samples // 4)
        bits_q = np.random.randint(0, 2, samples // 4)
        symbols_i = 2 * bits_i - 1
        symbols_q = 2 * bits_q - 1
        symbols = (symbols_i + 1j * symbols_q) / np.sqrt(2)
        iq_data = np.repeat(symbols, 4).astype(np.complex64)
    elif modulation == "FM":
        # Generate FM signal
        t = np.arange(samples) / 1e6
        freq_dev = 10e3  # 10 kHz deviation
        message = np.sin(2 * np.pi * 1e3 * t)  # 1 kHz message
        phase = 2 * np.pi * np.cumsum(freq_dev * message) / 1e6
        iq_data = np.exp(1j * phase).astype(np.complex64)
    else:
        # Default to complex white noise
        iq_data = (np.random.randn(samples) + 1j * np.random.randn(samples)).astype(np.complex64)
    
    # Add noise based on SNR
    signal_power = np.mean(np.abs(iq_data)**2)
    noise_power = signal_power / (10**(snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (np.random.randn(samples) + 1j * np.random.randn(samples))
    iq_data = iq_data + noise.astype(np.complex64)
    
    # Create metadata
    metadata = {
        "snr_db": snr_db,
        "modulation": modulation,
        "signal_power": float(signal_power),
        "noise_power": float(noise_power)
    }
    
    return TestSignal(
        id=f"test_signal_{signal_id:04d}",
        iq_data=iq_data,
        classification=modulation,
        metadata=metadata
    )

def iter_eval(num_signals: int = 1000) -> Iterator[TestSignal]:
    """
    Iterator that generates test signals for evaluation.
    This matches the expected interface for DATASET_FUNC.
    """
    modulations = ["BPSK", "QPSK", "FM", "NOISE"]
    snr_range = [-10, -5, 0, 5, 10, 15, 20]
    
    for i in range(num_signals):
        # Cycle through modulations and SNRs
        modulation = modulations[i % len(modulations)]
        snr_db = snr_range[i % len(snr_range)]
        
        signal = generate_test_signal(i, modulation, snr_db)
        yield signal

# For testing the module directly
if __name__ == "__main__":
    print("Testing simulation module...")
    
    # Generate a few test signals
    for i, signal in enumerate(iter_eval()):
        if i >= 5:
            break
        print(f"Signal {i}: {signal.classification}, SNR: {signal.metadata['snr_db']} dB, "
              f"IQ shape: {signal.iq_data.shape}")
    
    print("✅ Simulation module test complete")