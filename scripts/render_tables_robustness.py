#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robustness Table Renderer
Paper 13: NaN/Padding/Interpolation Robustness

Renders LaTeX tables showing best sanitization strategies by corruption ratio.
"""

import argparse
import json
import math
from pathlib import Path
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

def main():
    """Main entry point for table rendering."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="data/robustness_metrics.json")
    ap.add_argument("--templates", default="templates")
    ap.add_argument("--out", default="tables/robustness_tables.tex")
    args = ap.parse_args()

    # Load data
    data = json.loads(Path(args.json).read_text())
    
    # Group by ratio
    by_ratio = {}
    for row in data:
        by_ratio.setdefault(row["ratio"], []).append(row)

    # Find best strategy per ratio (minimize error, tiebreak by latency)
    table_rows = []
    for ratio, rows in sorted(by_ratio.items()):
        def key_err(r): 
            return (1.0 - r["accuracy"], r["latency_p50_ms"])
        best = min(rows, key=key_err)
        table_rows.append({
            "ratio": ratio,
            "best_mode": best["mode"],
            "error_pct": 100.0 * (1.0 - best["accuracy"]),
            "lat_p50": best["latency_p50_ms"],
            "lat_p95": best["latency_p95_ms"],
            "kl": best["psd_kl_median"]
        })

    # Aggregate latency per mode at focal ratio (e.g., 0.2)
    focal = 0.2
    lat_rows = []
    for row in data:
        if abs(row["ratio"] - focal) < 1e-9:
            lat_rows.append({
                "mode": row["mode"],
                "p50": row["latency_p50_ms"],
                "p95": row["latency_p95_ms"],
                "err_pct": 100.0 * (1.0 - row["accuracy"])
            })
    lat_rows = sorted(lat_rows, key=lambda r: (r["p50"], r["err_pct"]))

    # Render template
    env = Environment(
        loader=FileSystemLoader(args.templates),
        autoescape=select_autoescape([])
    )
    env.filters["latex"] = latex_escape
    env.filters["snum"] = safe_num
    tpl = env.get_template("robustness_tables.tex.j2")
    out = tpl.render(best_by_ratio=table_rows, latency_at_focal=lat_rows, focal=focal)
    
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(out)
    print(f"🧾 wrote {args.out}")

if __name__ == "__main__":
    main()