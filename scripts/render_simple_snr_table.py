#!/usr/bin/env python3
"""Simple SNR-based robustness table renderer with LaTeX escaping."""

import json
import argparse
import math
from collections import defaultdict
import numpy as np

# LaTeX character escaping
LATEX_SUBS = {
    '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#',
    '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}',
    '^': r'\textasciicircum{}', '\\': r'\textbackslash{}',
}

def latex_escape(s: str) -> str:
    """Escape string for LaTeX."""
    if s is None: 
        return ''
    out = []
    for ch in str(s):
        out.append(LATEX_SUBS.get(ch, ch))
    return ''.join(out)

def safe_num(x, fmt="{:.3f}", dash=r"\textemdash{}"):
    """Format number safely, replacing NaN/inf with dash."""
    try:
        if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
            return dash
        return fmt.format(float(x))
    except Exception:
        return dash

def render_simple_snr_table(snr_data, output_path):
    """Render a compact SNR-based table focusing on key metrics."""
    
    # Aggregate by mode and corruption level across SNR bins
    mode_corr_stats = defaultdict(lambda: defaultdict(list))
    
    for entry in snr_data:
        mode = entry['mode']
        corr = entry['ratio'] 
        # Aggregate all SNR bins for this mode/corruption combo
        mode_corr_stats[mode][corr].append({
            'accuracy': entry['accuracy'],
            'latency_p50': entry['latency_p50_ms'],
            'latency_p95': entry['latency_p95_ms'],
            'psd_kl': entry.get('psd_kl_median', float('nan'))
        })
    
    # Create simplified table
    content = []
    content.append("% Auto-generated simple SNR table")
    content.append("\\begin{table}[t]")
    content.append("\\centering")
    content.append("\\small")
    content.append("\\begin{tabular}{lccccc}")
    content.append("\\toprule")
    content.append("Mode & Corruption & Acc (avg) & Latency P50 & Latency P95 & PSD KL \\\\")
    content.append("\\midrule")
    
    modes = ['none', 'nan_to_num', 'interp_lin', 'zero_pad', 'mask_preserve']
    corruptions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    
    for mode in modes:
        if mode in mode_corr_stats:
            for corr in corruptions:
                if corr in mode_corr_stats[mode]:
                    stats = mode_corr_stats[mode][corr]
                    
                    # Average across SNR bins
                    avg_acc = np.mean([s['accuracy'] for s in stats])
                    avg_lat_p50 = np.mean([s['latency_p50'] for s in stats if not np.isnan(s['latency_p50'])])
                    avg_lat_p95 = np.mean([s['latency_p95'] for s in stats if not np.isnan(s['latency_p95'])])
                    valid_kl = [s['psd_kl'] for s in stats if not np.isnan(s['psd_kl'])]
                    avg_psd_kl = np.mean(valid_kl) if valid_kl else float('nan')
                    
                    psd_str = safe_num(avg_psd_kl, "{:.3f}")
                    content.append(f"{latex_escape(mode)} & {corr*100:.0f}\\% & "
                                 f"{safe_num(avg_acc, '{:.3f}')} & "
                                 f"{safe_num(avg_lat_p50, '{:.2f}')} & "
                                 f"{safe_num(avg_lat_p95, '{:.2f}')} & "
                                 f"{psd_str} \\\\")
                                 
        content.append("\\midrule")  # Separate modes
    
    content.append("\\bottomrule")
    content.append("\\end{tabular}")
    content.append("\\caption{Robustness by sanitization mode across corruption levels (SNR-averaged).}")
    content.append("\\label{tab:robustness-snr}")
    content.append("\\end{table}")
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(content))
    
    print(f"📊 Simple SNR table written to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate simple SNR robustness table")
    parser.add_argument("--snr-json", required=True, help="SNR metrics JSON file")
    parser.add_argument("--output", required=True, help="Output LaTeX table file")
    
    args = parser.parse_args()
    
    # Load data
    with open(args.snr_json) as f:
        snr_data = json.load(f)
    
    render_simple_snr_table(snr_data, args.output)