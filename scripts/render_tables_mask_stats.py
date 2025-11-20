#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mask Statistics Table Renderer
Paper 13: NaN/Padding/Interpolation Robustness

Renders LaTeX tables showing mask statistics for Appendix A.
"""

import argparse
import json
import math
from pathlib import Path
from statistics import median
from jinja2 import Environment, FileSystemLoader, select_autoescape

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

def safe_label(label: str) -> str:
    """Escape for LaTeX and ensure a row never starts with '[' (breaks \\[...])."""
    s = latex_escape(label)
    if s.startswith('['):
        s = '{}' + s
    return s

def dedup_by_ratio_bin(rows):
    """Collapse across modes: stats are identical per ratio/bin given identical corruption."""
    out = {}
    for r in rows:
        key = (r["ratio"], r.get("snr_bin", "ALL"))
        cur = out.get(key)
        cand = (r["nan_fraction_median"], r["nan_run_longest_median"], r["nan_run_count_median"], r["n"])
        if (cur is None) or (cand[-1] > cur[-1]):  # keep the largest-N summary
            out[key] = cand
    return [{
        "ratio": k[0],
        "snr_bin": k[1],
        "nan_fraction_median": v[0],
        "nan_run_longest_median": v[1],
        "nan_run_count_median": v[2],
        "n": v[3],
    } for k, v in out.items()]

def main():
    """Main entry point for mask statistics table rendering."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--global-json", default="data/robustness_metrics.json")
    ap.add_argument("--snr-json", default="data/robustness_metrics_snr.json")
    ap.add_argument("--templates", default="templates")
    ap.add_argument("--out", default="tables/robustness_mask_tables.tex")
    ap.add_argument("--focal_ratio", type=float, default=0.2)
    args = ap.parse_args()

    # Load data files
    grows = json.loads(Path(args.global_json).read_text()) if Path(args.global_json).exists() else []
    srows = json.loads(Path(args.snr_json).read_text()) if Path(args.snr_json).exists() else []

    # GLOBAL: one row per corruption ratio (mode-collapsed)
    g_collapsed = dedup_by_ratio_bin([{**r, "snr_bin": "ALL"} for r in grows])
    
    # SNR: one row per (ratio, snr_bin) (mode-collapsed)
    s_collapsed = dedup_by_ratio_bin(srows)

    # Slice SNR tables for focal ratio only (for compact Appendix)
    focal_snr = [r for r in s_collapsed if abs(r["ratio"] - args.focal_ratio) < 1e-9]

    # Render template
    env = Environment(
        loader=FileSystemLoader(args.templates), 
        autoescape=select_autoescape([])
    )
    env.filters["latex"] = latex_escape
    env.filters["snum"] = safe_num
    env.filters["slabel"] = safe_label
    tpl = env.get_template("robustness_mask_tables.tex.j2")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(tpl.render(
        global_rows=sorted(g_collapsed, key=lambda x: x["ratio"]),
        focal_snr=sorted(focal_snr, key=lambda x: x["snr_bin"]),
        focal=args.focal_ratio
    ))
    print(f"🧾 wrote {args.out}")

if __name__ == "__main__":
    main()