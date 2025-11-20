#!/usr/bin/env python3
import json, argparse
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, select_autoescape

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snr-json", default="data/robustness_metrics_snr.json")
    ap.add_argument("--templates", default="templates")
    ap.add_argument("--out", default="tables/robustness_tables_snr.tex")
    args = ap.parse_args()

    jpath = Path(args.snr_json)
    rows = json.loads(jpath.read_text()) if jpath.exists() else []
    # sort for stable table: (mode, ratio, snr_bin)
    rows = sorted(rows, key=lambda r: (r["mode"], r["ratio"], r["snr_bin"]))

    env = Environment(loader=FileSystemLoader(args.templates),
                      autoescape=select_autoescape([]))
    tpl = env.get_template("robustness_tables_snr_simple.tex.j2")

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(tpl.render(rows=rows))
    print(f"🧾 wrote {outp}")

if __name__ == "__main__":
    main()