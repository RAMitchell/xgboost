"""Render current Python experiment outputs as CSV, Markdown and LaTeX."""

# Numerical state and orchestration are kept explicit for reproduction.
# pylint: disable=too-many-locals

import argparse
import csv
import json
from pathlib import Path


def main():
    """Run the documented command-line entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    a = p.parse_args()
    manifest = json.loads((a.input / "manifest.json").read_text())
    methods = manifest["methods"]
    models = manifest["models"]
    rows = []
    md = [
        "| Benchmark | " + " | ".join(methods) + " |",
        "|---|" + "---:|" * len(methods),
    ]
    tex = [
        r"\begin{tabular}{l" + "r" * len(methods) + "}",
        r"\toprule",
        "Benchmark & " + " & ".join(methods) + r" \\",
        r"\midrule",
    ]
    for model in models:
        cells = []
        latex = []
        for method in methods:
            path = a.input / model / method / "results.json"
            r = (
                json.loads(path.read_text())
                if path.exists()
                else {"status": "pending", "mean_s": None, "sample_sd_s": None}
            )
            rows.append(
                {
                    "model": model,
                    "variant": method,
                    "status": r["status"],
                    "mean_s": r["mean_s"],
                    "sample_sd_s": r["sample_sd_s"],
                }
            )
            if r["status"] == "complete":
                cells.append(
                    f"{r['mean_s']:.5g}"
                    if r["sample_sd_s"] is None
                    else f"{r['mean_s']:.5g} ± {r['sample_sd_s']:.2g}"
                )
                latex.append(
                    f"${r['mean_s']:.5g}$"
                    if r["sample_sd_s"] is None
                    else f"${r['mean_s']:.5g} \\pm {r['sample_sd_s']:.2g}$"
                )
            else:
                cells.append("--")
                latex.append("--")
        md.append("| " + model + " | " + " | ".join(cells) + " |")
        tex.append(model.replace("_", r"\_") + " & " + " & ".join(latex) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "runtimes.md").write_text("\n".join(md) + "\n")
    (a.out_dir / "runtimes.tex").write_text("\n".join(tex) + "\n")
    with (a.out_dir / "runtimes.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
