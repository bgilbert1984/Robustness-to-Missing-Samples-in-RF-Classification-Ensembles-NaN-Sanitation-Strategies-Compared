#!/usr/bin/env python3
import json, argparse
from pathlib import Path
import numpy as np, matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--global-json", default="data/robustness_metrics.json")
ap.add_argument("--out", default="figs/latency_vs_corruption.pdf")
args = ap.parse_args()

rows = json.loads(Path(args.global_json).read_text())
# group by ratio (modes collapsed: take median)
from statistics import median
by_ratio = {}
for r in rows:
    by_ratio.setdefault(r["ratio"], []).append(r["latency_p50_ms"])
x = sorted(by_ratio.keys())
y = [median(by_ratio[k]) for k in x]

plt.figure()
plt.plot([100*t for t in x], y, marker='o')
plt.xlabel("Corruption ratio (%)")
plt.ylabel("Latency P50 (ms)")
plt.title("Latency vs. Corruption")
plt.tight_layout()
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
plt.savefig(args.out)
print(f"✅ wrote {args.out}")