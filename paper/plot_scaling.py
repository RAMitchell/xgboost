"""Plot CovType-sparse explanation runtime for SII orders two through six."""

# Numerical state and orchestration are kept explicit for reproduction.
# pylint: disable=wrong-import-position,too-many-locals

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    """Run the documented command-line entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", nargs="+", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    a = p.parse_args()
    rows = []
    for directory in a.inputs:
        manifest = json.loads((directory / "manifest.json").read_text())
        order = manifest["order"]
        row = {"order": order}
        for method in ["quadrature", "treeshapiq"]:
            r = json.loads(
                (directory / "covtype-sparse" / method / "results.json").read_text()
            )
            row[method] = r["mean_s"]
        rows.append(row)
    rows.sort(key=lambda r: r["order"])
    a.out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for method, label, marker in [
        ("quadrature", "Q-TreeSHAP", "o"),
        ("treeshapiq", "TreeSHAP-IQ", "s"),
    ]:
        valid = [r for r in rows if r[method] is not None]
        ax.plot(
            [r["order"] for r in valid],
            [r[method] for r in valid],
            marker=marker,
            linewidth=2,
            label=label,
        )
    ax.set_yscale("log")
    ax.set_xlabel("Interaction order")
    ax.set_ylabel("Wall-clock seconds")
    ax.set_xticks([2, 3, 4, 5, 6])
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(a.out_dir / f"higher-order-scaling.{ext}", dpi=200)
    plt.close(fig)
    with (a.out_dir / "order_scaling.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
