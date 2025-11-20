# NaN/Padding/Interpolation Robustness

This paper systematically evaluates the robustness of RF ensemble classifiers to input corruption, specifically focusing on the impact of different NaN sanitation strategies.

## Quick Start

```bash
# Set environment variables
export DATASET_FUNC="simulation:iter_eval"
export CLASSIFIER_SPEC="ensemble_ml_classifier:EnsembleMLClassifier"

# Generate baseline analysis
make press

# Generate extended SNR-stratified analysis  
make press-snr

# Quick development run (fewer samples)
make dev-quick
```

## Directory Structure

```
├── code/                           # Core implementation
│   ├── sanitize_io.py             # Input sanitation helpers
│   ├── temporal_builder.py        # Temporal feature extraction
│   ├── spectral_builder.py        # Spectral feature extraction
│   └── ...
├── scripts/                        # Evaluation scripts
│   ├── corruption_robustness.py   # Main evaluation script
│   ├── render_tables_*.py         # Table renderers
│   └── ...
├── templates/                      # Jinja2 templates
│   ├── robustness_tables.tex.j2   # Basic tables
│   ├── robustness_snr_tables.tex.j2 # SNR-stratified tables
│   └── robustness_mask_tables.tex.j2 # Mask statistics
├── figs/                          # Generated figures (created by make)
├── data/                          # Generated data (created by make)
├── tables/                        # Generated tables (created by make)
├── main_nan_padding_interp.tex   # Main LaTeX document
├── Makefile                       # Build automation
├── simulation.py                  # Test signal generator
└── ensemble_ml_classifier.py     # Test classifier
```

## Configuration

### Corruption Parameters

- `RATIOS`: Corruption levels to test (default: "0.0,0.05,0.1,0.2,0.4,0.6")
- `SAN_MODES`: Sanitation strategies (default: "none,nan_to_num,interp_lin,zero_pad")
- `SAMPLES`: Number of signals to evaluate (default: 200)
- `BURST`: Use burst corruption (1) vs scattered (0)

### SNR Stratification

- `SNR_KEY`: Metadata key for SNR values (default: "snr_db")  
- `SNR_BINS`: Bin edges for SNR stratification (default: "-10,-5,0,5,10,15")
- `PAD_EDGES`: Add -∞/+∞ bins (default: 1)
- `FOCAL`: Focal corruption ratio for detailed analysis (default: 0.2)

## Input Sanitation Strategies

1. **none**: No sanitation—NaNs propagate through the system
2. **nan_to_num**: Replace NaNs with zeros using `np.nan_to_num`
3. **interp_lin**: Linear interpolation of NaN spans
4. **zero_pad**: Direct replacement of NaNs with zero values
5. **mask_preserve**: Zero-fill NaNs while preserving mask information

## Integration with Your Pipeline

To integrate with your existing pipeline:

1. Set `DATASET_FUNC` to point to your dataset iterator
2. Set `CLASSIFIER_SPEC` to point to your classifier class
3. Ensure your classifier forwards `sanitize_mode` from signal metadata to feature builders
4. Import and use the sanitation functions in your feature extraction pipeline

Example integration:

```python
# In your classifier
def classify_signal(self, signal):
    mode = signal.metadata.get('sanitize_mode', 'nan_to_num')
    temporal = create_temporal_features(signal, sanitize_mode=mode)
    spectral = create_spectral_features(signal, sanitize_mode=mode)
    # ... rest of classification
```

## Output

The system generates:

### Figures
- Error vs corruption ratio by sanitizer
- Latency (p50/p95) vs corruption ratio  
- PSD distortion (KL divergence) vs corruption ratio
- Per-SNR bin analysis (if using press-snr)

### Tables
- Best sanitation strategy per corruption level
- Latency/accuracy trade-offs at focal corruption level
- SNR-stratified robustness analysis
- Mask statistics for Appendix A

### Data Files
- `data/robustness_metrics.json`: Global aggregated results
- `data/robustness_metrics_snr.json`: SNR-stratified results

## Development

```bash
# Test environment setup
make test-env

# Quick development iterations
make dev-quick    # 50 samples, 3 corruption levels
make dev-snr      # 50 samples, 3 SNR bins

# Clean all generated files
make clean
```

## Requirements

- Python 3.7+
- numpy, matplotlib, jinja2
- LaTeX distribution for PDF generation
- Your existing RF signal processing pipeline

## Citation

```bibtex
@article{gilbert2025robustness,
  title={NaN, Padding, and Interpolation Robustness in RF Ensembles},
  author={Gilbert, B.},
  journal={IEEE Transactions on Signal Processing},
  year={2025}
}
```

More at Spectrcyde https://172-234-197-23.ip.linodeusercontent.com/?p=4777
