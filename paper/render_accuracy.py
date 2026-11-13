"""Render float64 per-feature, efficiency and conservative SAE appendix tables."""

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
    rows = []
    for path in sorted(a.input.glob("*/metrics.json")):
        m = json.loads(path.read_text())
        r = {
            "dataset": m["dataset"],
            "seed": m["seed"],
            "depth": m["requested_depth"],
            "actual_depth": m["max_depth"],
            "distinct_depth": m["max_unique_depth"],
        }
        for n in [8, 16]:
            v = m["methods"][f"standard{n}_float64"]
            r.update(
                {
                    f"{n}_max_error": v["attribution_abs_error"]["max"],
                    f"{n}_efficiency": v["efficiency_abs_error"]["max"],
                    f"{n}_sae_max": v["sum_abs_errs_max"],
                    f"{n}_sae_mean": v["sum_abs_errs_mean"],
                }
            )
        r["reference_efficiency"] = m["reference_efficiency_max_error"]
        rows.append(r)
    if not rows:
        raise ValueError("No completed accuracy cases")
    rows.sort(key=lambda r: (r["dataset"], r["seed"], r["depth"]))
    a.out_dir.mkdir(parents=True, exist_ok=True)
    with (a.out_dir / "float64_accuracy.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for filename, columns in [
        (
            "float64_accuracy",
            [
                "depth",
                "8_max_error",
                "8_efficiency",
                "16_max_error",
                "16_efficiency",
                "reference_efficiency",
            ],
        ),
        (
            "sae",
            [
                "depth",
                "actual_depth",
                "8_sae_max",
                "8_sae_mean",
                "16_sae_max",
                "16_sae_mean",
            ],
        ),
    ]:
        text = [
            r"\begin{tabular}{" + "r" * len(columns) + "}",
            r"\toprule",
            " & ".join(c.replace("_", r"\_") for c in columns) + r" \\",
            r"\midrule",
        ]
        for r in rows:
            text.append(
                " & ".join(
                    str(r[c]) if c in ["depth", "actual_depth"] else f"${r[c]:.2e}$"
                    for c in columns
                )
                + r" \\"
            )
        text += [r"\bottomrule", r"\end{tabular}"]
        (a.out_dir / f"{filename}.tex").write_text("\n".join(text) + "\n")


if __name__ == "__main__":
    main()
