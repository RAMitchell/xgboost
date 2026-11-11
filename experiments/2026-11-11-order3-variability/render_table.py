"""Export validated repeated third-order runtimes in the paper's table layout."""
# pylint: disable=duplicate-code
# pylint: disable=missing-function-docstring,too-many-locals

import argparse
import csv
import json
import math
from pathlib import Path

from adapters import ROOT


def cell(result):
    if result["status"] == "pending":
        return r"\textit{pending}"
    if result["status"] != "complete":
        return "--"
    mean = result["mean_s"] * 1000
    sd = result["sample_sd_s"] * 1000
    if sd == 0:
        return f"${mean:.4g} \\pm 0$"
    unit = 10 ** math.floor(math.log10(sd))
    sd = round(sd / unit) * unit
    unit = 10 ** math.floor(math.log10(sd))
    digits = max(0, -int(math.floor(math.log10(unit))))
    mean = round(mean / unit) * unit
    return f"${mean:.{digits}f} \\pm {sd:.{digits}f}$"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    out = args.directory
    data = json.loads((out / "results.json").read_text(encoding="utf-8"))
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    old = ROOT / "experiments/2026-11-10-runtime-variability/full-suite-10"
    identity = json.loads((old / "manifest.json").read_text(encoding="utf-8"))[
        "identity"
    ]
    assert all(
        v == identity["model_sha256"][k] for k, v in manifest["model_sha256"].items()
    )
    models = {
        r["model"]: r
        for r in json.loads((old / "first-order.json").read_text(encoding="utf-8"))[
            "models_table"
        ]
    }
    rows = []
    latex = [
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Benchmark & $F$ & $D$ & $d$ & TreeSHAP-IQ & \methodshort{} & Speedup \\",
        r"\midrule",
    ]
    labels = {"cal_housing": "CalHousing", "covtype": "CovType"}
    for r in data["results"]:
        model = r["model"]
        ds, setting = model.rsplit("-", 1)
        stats = models[model]
        a = r["methods"]["treeshapiq"]
        b = r["methods"]["quadrature"]
        speed = r["speedup"]
        speedcell = "--" if speed is None else f"{speed:.2f}"
        latex.append(
            " & ".join(
                [
                    labels[ds] + "-" + setting,
                    str(stats["num_features"])
                    if "num_features" in stats
                    else ("8" if ds == "cal_housing" else "54"),
                    str(stats["max_max_depth"]),
                    str(stats["max_unique_feature_depth"]),
                    cell(a),
                    cell(b),
                    speedcell,
                ]
            )
            + r" \\"
        )
        rows.append(
            {
                "model": model,
                "treeshapiq_mean_s": a.get("mean_s"),
                "treeshapiq_sd_s": a.get("sample_sd_s"),
                "quadrature_mean_s": b.get("mean_s"),
                "quadrature_sd_s": b.get("sample_sd_s"),
                "speedup": speed,
                "treeshapiq_status": a["status"],
                "quadrature_status": b["status"],
                "treeshapiq_error": a.get("error"),
                "quadrature_error": b.get("error"),
            }
        )
    latex.extend([r"\bottomrule", r"\end{tabular}"])
    (out / "table4-mean-sd-ms.tex").write_text("\n".join(latex) + "\n")
    with (out / "table4-summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(out / "table4-summary.csv")


if __name__ == "__main__":
    main()
