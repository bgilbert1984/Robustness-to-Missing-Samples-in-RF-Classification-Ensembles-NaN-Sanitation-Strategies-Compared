#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NaN/Padding/Interpolation Robustness Evaluation
Paper 13: Quantifying the impact of input corruption and sanitation strategies

This script injects controlled NaN corruption into IQ data and evaluates
the robustness of different sanitation strategies across corruption ratios.
"""

import os
import json
import time
import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from statistics import median

# Add code directory to path for imports
script_dir = Path(__file__).parent
code_dir = script_dir.parent / "code"
project_dir = script_dir.parent
sys.path.insert(0, str(code_dir))
sys.path.insert(0, str(project_dir))

from sanitize_io import build_nan_mask, mask_run_stats

# ---- dataset & classifier loading via env (matches your trilogy style) ----
# Expect: DATASET_FUNC="simulation:iter_eval" or "my_loader:iter_eval"
#         CLASSIFIER_SPEC="ensemble_ml_classifier:EnsembleMLClassifier"
def _import_by_spec(spec: str):
    """Import a function/class by module:name specification."""
    if ":" not in spec:
        raise ValueError(f"Invalid spec format: {spec}. Expected 'module:name'")
    mod, name = spec.split(":", 1)
    
    try:
        # Try importing normally first
        m = __import__(mod, fromlist=[name])
        return getattr(m, name)
    except (ImportError, AttributeError):
        # Try importing from project directory
        sys.path.insert(0, str(script_dir.parent))
        m = __import__(mod, fromlist=[name])
        return getattr(m, name)

def get_dataset_iter():
    fn = os.environ.get("DATASET_FUNC", "simulation:iter_eval")
    return _import_by_spec(fn)

def get_classifier():
    spec = os.environ.get("CLASSIFIER_SPEC", "ensemble_ml_classifier:EnsembleMLClassifier")
    Cls = _import_by_spec(spec)
    return Cls()

# ---- corruption generator ----

def inject_nan_corruption(iq: np.ndarray, ratio: float, burst: bool = True, seed: int = 1337) -> np.ndarray:
    """
    Replace a fraction 'ratio' of samples with NaNs. 'burst' → contiguous runs, else random scatter.
    """
    rng = np.random.default_rng(seed)
    iq = iq.copy()
    n = iq.shape[0]
    k = max(1, int(round(ratio * n)))
    if k <= 0:
        return iq
    if burst:
        start = rng.integers(0, max(1, n - k))
        idx = np.arange(start, min(n, start + k))
    else:
        idx = rng.choice(n, size=k, replace=False)
    re, im = iq.real, iq.imag
    re[idx] = np.nan; im[idx] = np.nan
    return re + 1j * im

# ---- metrics ----

def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Compute KL divergence between two probability distributions."""
    p = p.astype(np.float32); q = q.astype(np.float32)
    eps = 1e-8
    p = p / (p.sum() + eps); q = q / (q.sum() + eps)
    return float(np.sum(p * np.log((p + eps) / (q + eps))))

def parse_bins(bstr: str, pad_edges: bool):
    """Parse bin specification string into edges list."""
    # e.g., "-10,-5,0,5,10,15" -> edges list
    edges = [float(s.strip()) for s in bstr.split(",") if s.strip()!=""]
    if pad_edges:
        edges = [-float("inf")] + edges + [float("inf")]
    return edges

def label_bin(v, edges):
    """Assign value to bin based on edges."""
    # edges are sorted; return "[a, b)" with pretty infinities
    for i in range(len(edges)-1):
        a, b = edges[i], edges[i+1]
        if a <= v < b:
            def fmt(x): 
                if x == -float("inf"): return "-∞"
                if x ==  float("inf"): return "+∞"
                return f"{int(x)}" if abs(x-round(x))<1e-9 else f"{x:g}"
            return f"[{fmt(a)}, {fmt(b)})"
    return "N/A"

def evaluate(args):
    """Main evaluation function."""
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    figdir = Path(args.figdir); figdir.mkdir(parents=True, exist_ok=True)
    datadir = Path(args.datadir); datadir.mkdir(parents=True, exist_ok=True)

    ratios = [float(r) for r in args.ratios.split(",")]  # e.g., "0.0,0.05,0.1,0.2,0.4,0.6"
    modes  = args.modes.split(",")  # "none,nan_to_num,interp_lin,zero_pad"

    # SNR binning knobs
    snr_key   = args.snr_key
    snr_edges = parse_bins(args.snr_bins, args.pad_edges)

    # Load dataset and classifier
    try:
        dataset_iter = get_dataset_iter()
        clf = get_classifier()
    except Exception as e:
        print(f"Error loading dataset or classifier: {e}")
        print("Make sure DATASET_FUNC and CLASSIFIER_SPEC environment variables are set correctly")
        return

    # Prefetch evaluation pool
    pool = []
    try:
        for i, sig in enumerate(dataset_iter()):
            pool.append(sig)
            if len(pool) >= args.samples:
                break
    except Exception as e:
        print(f"Error loading samples from dataset: {e}")
        return
        
    if len(pool) == 0:
        raise RuntimeError("Dataset iterator yielded no samples.")

    print(f"Loaded {len(pool)} samples for evaluation")

    # Compute SNR bin labels for each sample
    snr_bins = {}
    for sig in pool:
        metadata = getattr(sig, "metadata", {})
        v = metadata.get(snr_key, None)
        if v is None:
            # try fallback "snr" or "snr_db"
            v = metadata.get("snr_db", metadata.get("snr", None))
        # Unavailable → put in a single "N/A"
        if v is None:
            snr_bins[getattr(sig, 'id', f'sig_{id(sig)}')] = "N/A"
        else:
            snr_bins[getattr(sig, 'id', f'sig_{id(sig)}')] = label_bin(float(v), snr_edges)

    # Import spectral builder for baseline PSD computation
    try:
        from spectral_builder import _create_spectral_input as spectral_builder
    except ImportError:
        print("Warning: spectral_builder not found, using simple FFT for PSD baseline")
        def spectral_builder(iq, **kwargs):
            nfft = kwargs.get('nfft', 256)
            x = iq[:nfft] if len(iq) >= nfft else np.pad(iq, (0, nfft - len(iq)), mode="constant")
            X = np.fft.fftshift(np.fft.fft(x))
            psd = (np.abs(X) ** 2).astype(np.float32)
            return psd / (psd.max() + 1e-8)

    # Compute baseline PSD cache (no corruption, no sanitization)
    baseline_psd_cache = {}
    for sig in pool:
        try:
            psd0 = spectral_builder(sig.iq_data, sanitize_mode="none", nfft=args.nfft)
            baseline_psd_cache[getattr(sig, 'id', f'sig_{id(sig)}')] = psd0
        except Exception as e:
            print(f"Warning: Failed to compute baseline PSD for signal: {e}")
            # Create dummy baseline
            baseline_psd_cache[getattr(sig, 'id', f'sig_{id(sig)}')] = np.ones(args.nfft, dtype=np.float32)

    # Main evaluation sweep
    sample_rows = []
    seed = args.seed

    for ratio in ratios:
        for mode in modes:
            print(f"Evaluating ratio={ratio:.3f}, mode={mode}")
            
            for j, sig in enumerate(pool):
                # Inject corruption
                iq_cor = inject_nan_corruption(sig.iq_data, ratio=ratio, burst=args.burst, seed=seed + j)
                
                # Compute mask stats before sanitation
                mask = build_nan_mask(iq_cor)
                nan_frac, nan_run_longest, nan_run_count = mask_run_stats(mask)
                
                # Create corrupted signal with metadata for sanitation
                sig_id = getattr(sig, 'id', f'sig_{id(sig)}')
                
                # Create a new signal object with corrupted IQ and sanitation metadata
                if hasattr(sig, '_replace'):
                    # If it's a dataclass with _replace method
                    new_metadata = {**getattr(sig, "metadata", {}), "sanitize_mode": mode}
                    sig_cor = sig._replace(iq_data=iq_cor, metadata=new_metadata)
                else:
                    # Create a simple container object
                    class SignalContainer:
                        def __init__(self, iq_data, metadata):
                            self.iq_data = iq_data
                            self.metadata = metadata
                            self.id = sig_id
                            # Copy other attributes if they exist
                            for attr in ['classification', 'sample_rate_hz', 'center_freq_hz', 'timestamp']:
                                if hasattr(sig, attr):
                                    setattr(self, attr, getattr(sig, attr))
                    
                    new_metadata = {**getattr(sig, "metadata", {}), "sanitize_mode": mode}
                    sig_cor = SignalContainer(iq_cor, new_metadata)
                
                # Measure classification time and accuracy
                t0 = time.perf_counter()
                try:
                    pred = clf.classify_signal(sig_cor)
                except Exception as e:
                    print(f"Warning: Classification failed for signal {sig_id}: {e}")
                    pred = "UNKNOWN"
                dt = (time.perf_counter() - t0) * 1000.0
                
                # Compute correctness
                true_class = getattr(sig, "classification", None)
                correct = (true_class is not None and pred == true_class)
                
                # Compute PSD distortion
                try:
                    psd_cor = spectral_builder(iq_cor, sanitize_mode=mode, nfft=args.nfft)
                    kl = kl_divergence(baseline_psd_cache[sig_id], psd_cor)
                except Exception as e:
                    print(f"Warning: PSD computation failed: {e}")
                    kl = 0.0
                
                sample_rows.append({
                    "id": sig_id,
                    "ratio": ratio,
                    "mode": mode,
                    "correct": correct,
                    "lat": dt,
                    "kl": kl,
                    "snr_bin": snr_bins[sig_id],
                    "nan_fraction": nan_frac,
                    "nan_run_longest": nan_run_longest,
                    "nan_run_count": nan_run_count,
                })

    print(f"Collected {len(sample_rows)} sample evaluations")

    # Global aggregation
    by_g = {}
    for r in sample_rows:
        key = (r["ratio"], r["mode"])
        by_g.setdefault(key, []).append(r)

    global_agg = []
    for (ratio, mode), rows in by_g.items():
        n = len(rows)
        acc = sum(1 if x["correct"] else 0 for x in rows)/n
        p50 = median([x["lat"] for x in rows])
        p95 = float(np.percentile([x["lat"] for x in rows], 95))
        klm = median([x["kl"] for x in rows])
        # Mask statistics
        mf  = median([x["nan_fraction"] for x in rows])
        mrl = median([float(x["nan_run_longest"]) for x in rows])
        mrc = median([float(x["nan_run_count"]) for x in rows])
        
        global_agg.append({
            "ratio": ratio, "mode": mode, "n": n,
            "accuracy": acc, "latency_p50_ms": p50, "latency_p95_ms": p95, "psd_kl_median": klm,
            "nan_fraction_median": mf,
            "nan_run_longest_median": mrl,
            "nan_run_count_median": mrc,
        })

    # Save global results
    jpath_global = datadir / "robustness_metrics.json"
    with open(jpath_global, "w") as f:
        json.dump(global_agg, f, indent=2)
    print(f"✅ Wrote {jpath_global}")

    # Per-SNR bin aggregation
    by_snr = {}
    for r in sample_rows:
        key = (r["ratio"], r["mode"], r["snr_bin"])
        by_snr.setdefault(key, []).append(r)

    perbin = []
    for (ratio, mode, snr_label), rows in by_snr.items():
        n = len(rows)
        acc = sum(1 if x["correct"] else 0 for x in rows)/n
        p50 = median([x["lat"] for x in rows])
        p95 = float(np.percentile([x["lat"] for x in rows], 95))
        klm = median([x["kl"] for x in rows])
        # Mask statistics
        mf  = median([x["nan_fraction"] for x in rows])
        mrl = median([float(x["nan_run_longest"]) for x in rows])
        mrc = median([float(x["nan_run_count"]) for x in rows])

        perbin.append({
            "ratio": ratio, "mode": mode, "snr_bin": snr_label, "n": n,
            "accuracy": acc, "latency_p50_ms": p50, "latency_p95_ms": p95, "psd_kl_median": klm,
            "nan_fraction_median": mf,
            "nan_run_longest_median": mrl,
            "nan_run_count_median": mrc,
        })

    jpath_bin = datadir / "robustness_metrics_snr.json"
    with open(jpath_bin, "w") as f:
        json.dump(perbin, f, indent=2)
    print(f"✅ Wrote {jpath_bin}")

    # Generate figures
    generate_figures(global_agg, perbin, figdir, ratios, modes, args)
    print("✅ Generated all figures")

def generate_figures(global_results, snr_results, figdir, ratios, modes, args):
    """Generate evaluation figures."""
    
    # Global figures (error vs corruption, latency vs corruption, etc.)
    for ykey, fname, ylabel in [
        ("accuracy", "error_vs_corruption.pdf", "Error (1 - Accuracy)"),
        ("latency_p50_ms", "latency_vs_corruption_p50.pdf", "Latency p50 (ms)"),
        ("latency_p95_ms", "latency_vs_corruption_p95.pdf", "Latency p95 (ms)"),
        ("psd_kl_median", "psd_kl_vs_corruption.pdf", "Median PSD KL"),
    ]:
        plt.figure(figsize=(7.5, 5.0))
        for mode in modes:
            xs, ys = [], []
            for r in ratios:
                matching = [x for x in global_results if x["ratio"] == r and x["mode"] == mode]
                if matching:
                    v = matching[0][ykey]
                    if ykey == "accuracy":
                        v = 1.0 - v  # Convert to error
                    xs.append(r)
                    ys.append(v)
            if xs:
                plt.plot(xs, ys, marker="o", label=mode)
        
        plt.xlabel("Corruption ratio (NaN fraction)")
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        plt.legend(title="Sanitizer")
        out = figdir / fname
        plt.tight_layout()
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"🖼  {out}")

    # Per-SNR bin figures
    bins_present = sorted({row["snr_bin"] for row in snr_results})
    for sn in bins_present:
        subset = [x for x in snr_results if x["snr_bin"] == sn]
        if not subset:
            continue
        
        # Plot error vs corruption by mode for this SNR bin
        plt.figure(figsize=(7.5, 5.0))
        for mode in modes:
            xs, ys = [], []
            for r in ratios:
                cand = [x for x in subset if x["ratio"]==r and x["mode"]==mode]
                if cand:
                    v = 1.0 - cand[0]["accuracy"]  # Convert to error
                    xs.append(r)
                    ys.append(v)
            if xs:
                plt.plot(xs, ys, marker="o", label=mode)
        
        plt.xlabel("Corruption ratio (NaN fraction)")
        plt.ylabel("Error (1 - Accuracy)")
        plt.title(f"SNR bin {sn}")
        plt.grid(True, alpha=0.3)
        plt.legend(title="Sanitizer")
        
        # Safe filename
        safe_sn = sn.replace(' ','_').replace('[','').replace(']','').replace(',','_').replace('∞','inf').replace('/','_').replace('N/A', 'NA')
        out = figdir / f"error_vs_corruption__snr_{safe_sn}.pdf"
        plt.tight_layout()
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"🖼  {out}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="NaN/Padding/Interpolation Robustness Evaluation")
    parser.add_argument("--ratios", default="0.0,0.05,0.1,0.2,0.4,0.6", help="Corruption ratios to test")
    parser.add_argument("--modes", default="none,nan_to_num,interp_lin,zero_pad", help="Sanitation modes to test")
    parser.add_argument("--nfft", type=int, default=256, help="FFT size for spectral analysis")
    parser.add_argument("--samples", type=int, default=200, help="Number of samples to evaluate")
    parser.add_argument("--burst", type=int, default=1, help="Use burst corruption (1) or scattered (0)")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for reproducibility")
    parser.add_argument("--figdir", default="figs", help="Output directory for figures")
    parser.add_argument("--datadir", default="data", help="Output directory for data files")
    parser.add_argument("--outdir", default=".", help="Base output directory")
    parser.add_argument("--snr-key", default="snr_db", help="Metadata key for SNR binning")
    parser.add_argument("--snr-bins", default="-10,-5,0,5,10,15", help="Comma-separated SNR bin edges")
    parser.add_argument("--pad-edges", action="store_true", help="Pad SNR bins with -∞ and +∞")
    
    args = parser.parse_args()
    args.burst = bool(args.burst)
    
    evaluate(args)

if __name__ == "__main__":
    main()