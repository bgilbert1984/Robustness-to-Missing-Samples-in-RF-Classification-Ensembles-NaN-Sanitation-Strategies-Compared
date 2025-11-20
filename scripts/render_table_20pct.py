#!/usr/bin/env python3
"""
20% Burst Corruption Table Generator
Paper 13: NaN/Padding/Interpolation Robustness

Renders a focused table showing performance at 20% burst corruption.
"""

import json
import statistics as st
from pathlib import Path

def main():
    """Generate the 20% burst corruption table."""
    data_file = Path("data/robustness_metrics.json")
    if not data_file.exists():
        print("❌ No robustness metrics found - run evaluation first")
        return
    
    j = json.loads(data_file.read_text())
    focus = [r for r in j if abs(r.get("ratio", -1) - 0.20) < 1e-9]
    
    # Helper to pull per-mode summaries
    def summarize(mode):
        rows = [r for r in focus if r["mode"] == mode]
        if not rows:
            return None
        acc = 100 * st.median([r["accuracy"] for r in rows])
        p50 = st.median([r["latency_p50_ms"] for r in rows])
        p95 = st.median([r["latency_p95_ms"] for r in rows])
        kl = st.median([r.get("psd_kl_median", float('nan')) for r in rows 
                       if r.get("psd_kl_median") is not None])
        return dict(acc=acc, p50=p50, p95=p95, kl=kl)
    
    modes = ["baseline_clean", "none", "nan_to_num", "zero_pad", "interp_lin"]
    S = {m: summarize(m) for m in modes}
    
    def fmt(v, places=1):
        if v is None or (isinstance(v, float) and (v != v)):  # NaN check
            return "\\textemdash{}"
        return f"{v:.{places}f}" if isinstance(v, (int, float)) else "\\textemdash{}"
    
    def get_stat(mode_dict, stat, default=None):
        """Safely get a stat from a mode dict."""
        if mode_dict is None:
            return default
        return mode_dict.get(stat, default)
    
    # Build values first
    values = {
        "b_acc": fmt(get_stat(S.get("baseline_clean"), "acc")),
        "b_p95": fmt(get_stat(S.get("baseline_clean"), "p95")),
        "n2n_acc": fmt(get_stat(S.get("nan_to_num"), "acc")),
        "n2n_p95": fmt(get_stat(S.get("nan_to_num"), "p95")),
        "n2n_kl": fmt(get_stat(S.get("nan_to_num"), "kl"), 3),
        "zp_acc": fmt(get_stat(S.get("zero_pad"), "acc")),
        "zp_p95": fmt(get_stat(S.get("zero_pad"), "p95")),
        "zp_kl": fmt(get_stat(S.get("zero_pad"), "kl"), 3),
        "il_acc": fmt(get_stat(S.get("interp_lin"), "acc")),
        "il_p95": fmt(get_stat(S.get("interp_lin"), "p95")),
        "il_kl": fmt(get_stat(S.get("interp_lin"), "kl"), 3),
    }
    
    tex = r"""\begin{table}[t]
\centering
\caption{Performance at 20%% burst corruption on RadioML 2018.01A (8-model ensemble)}
\begin{tabular}{lccc}
\toprule
Sanitizer & Top-1 Acc. (\%%) & p95 Latency (ms) & PSD KL Div. \\
\midrule
Clean baseline  & %s & %s & 0.000 \\
None (crash)    & ---       & ---       & ---   \\
\texttt{nan\_to\_num} & %s & %s & %s \\
\texttt{zero\_pad}    & %s  & %s  & %s  \\
\textbf{\texttt{interp\_lin}} & \textbf{%s} & %s & \textbf{%s} \\
\bottomrule
\end{tabular}
\label{tab:burst-20pct}
\end{table}
""" % (values["b_acc"], values["b_p95"],
       values["n2n_acc"], values["n2n_p95"], values["n2n_kl"],
       values["zp_acc"], values["zp_p95"], values["zp_kl"],
       values["il_acc"], values["il_p95"], values["il_kl"])
    
    Path("tables").mkdir(exist_ok=True)
    Path("tables/table_20pct.tex").write_text(tex)
    print("🧾 wrote tables/table_20pct.tex")

if __name__ == "__main__":
    main()