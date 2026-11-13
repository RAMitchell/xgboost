"""Render the paper's four-dataset normalized depth-error comparison."""

# Numerical state and orchestration are kept explicit for reproduction.
# pylint: disable=wrong-import-position,too-many-locals

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

DATASETS = {
    "adult": ("Adult", "#0072B2"),
    "cal_housing": ("CalHousing", "#D55E00"),
    "covtype": ("CovType", "#697A24"),
    "fashion_mnist": ("Fashion-MNIST", "#B34F8C"),
}
METHODS = {
    "cpu_standard8_float32": ("Q-TreeSHAP-8", "-", "o"),
    "cpu_treeshap_float32": ("TreeSHAP", "--", "^"),
}


def main():
    """Run the documented command-line entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    a = p.parse_args()
    rows = []
    for path in sorted(a.input.glob("*/metrics.json")):
        m = json.loads(path.read_text())
        for method in METHODS:
            value = m["methods"][method]["attribution_nrmse"]
            if value is None:
                raise ValueError(f"Undefined normalization: {path}")
            rows.append(
                {
                    "dataset": m["dataset"],
                    "seed": m["seed"],
                    "requested_depth": m["requested_depth"],
                    "max_distinct_depth": m["max_unique_depth"],
                    "method": method,
                    "nrmse": value,
                }
            )
    if not rows:
        raise ValueError("No depth-study results found")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    with (a.out_dir / "depth_scores.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    plt.rcdefaults()
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for dataset, (_, color) in DATASETS.items():
        for method, (_, style, marker) in METHODS.items():
            selected = [
                r for r in rows if r["dataset"] == dataset and r["method"] == method
            ]
            depths = sorted({r["requested_depth"] for r in selected})
            if not depths:
                continue
            values = [
                np.asarray([r["nrmse"] for r in selected if r["requested_depth"] == d])
                for d in depths
            ]
            ax.plot(
                depths,
                [np.median(v) for v in values],
                color=color,
                linestyle=style,
                marker=marker,
                linewidth=2,
                markersize=5,
            )
            ax.fill_between(
                depths,
                [min(v) for v in values],
                [max(v) for v in values],
                color=color,
                alpha=0.045,
                linewidth=0,
            )
    ax.set_yscale("log")
    ax.set_xlabel("Requested max_depth")
    ax.set_ylabel("Normalized attribution RMSE")
    ax.set_xticks([4, 8, 12, 16, 24, 32, 40, 50])
    ax.grid(True, which="major", alpha=0.25)
    handles = [
        Line2D([0], [0], color=color, linewidth=2, label=name)
        for name, color in DATASETS.values()
    ]
    handles += [
        Line2D(
            [0],
            [0],
            color="#444444",
            linestyle=style,
            marker=marker,
            linewidth=2,
            label=name,
        )
        for name, style, marker in METHODS.values()
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        ncol=2,
        fontsize=8.5,
        columnspacing=1.5,
        handlelength=2.8,
    )
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(a.out_dir / f"depth-normalized-error.{ext}", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
