#!/usr/bin/env python3
"""
Simple table generator to avoid Jinja template complexity
"""
import json
from pathlib import Path

def generate_simple_table():
    # Load data
    data = json.load(open('data/robustness_metrics.json'))
    by_ratio = {}
    for row in data:
        by_ratio.setdefault(row['ratio'], []).append(row)
    
    # Generate table
    table = []
    table.append('% Auto-generated; do not edit by hand.')
    table.append('')
    table.append(r'\begin{table}[t]')
    table.append(r'\centering')
    table.append(r'\small')
    table.append(r'\begin{tabular}{lcccc}')
    table.append(r'\toprule')
    table.append(r'Corruption & Best Sanitizer & Error (\%) & p50 (ms) & Median PSD KL \\')
    table.append(r'\midrule')
    
    for ratio, rows in sorted(by_ratio.items()):
        best = min(rows, key=lambda r: (1.0 - r['accuracy'], r['latency_p50_ms']))
        error_pct = 100 * (1 - best['accuracy'])
        mode_name = best['mode'].replace('_', '\\_')
        line = f"{100*ratio:.0f} & \\texttt{{{mode_name}}} & {error_pct:.1f} & {best['latency_p50_ms']:.2f} & {best['psd_kl_median']:.3f} \\\\"
        table.append(line)
    
    table.append(r'\bottomrule')
    table.append(r'\end{tabular}')
    table.append(r'\caption{Best-performing sanitation strategy per corruption ratio.}')
    table.append(r'\label{tab:robustness_best}')
    table.append(r'\end{table}')
    
    # Write to file
    Path('tables').mkdir(exist_ok=True)
    with open('tables/robustness_tables.tex', 'w') as f:
        f.write('\n'.join(table))
    print("Generated tables/robustness_tables.tex")

if __name__ == "__main__":
    generate_simple_table()