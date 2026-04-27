"""Build order-scaling tables and charts from benchmark JSON outputs."""
# pylint: disable=missing-function-docstring

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_rows(paths: list[Path], model: str) -> list[dict[str, object]]:
    rows = []
    for path in paths:
        payload = json.loads(path.read_text())
        for row in payload["results"]:
            if row.get("model") == model and "error" not in row:
                rows.append(row)
    return sorted(rows, key=lambda row: int(row["order"]))


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "| order | quadrature s | TreeSHAP-IQ s | speedup | max abs diff | mean abs diff |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['order']} | {row['quadrature_total_s']:.3f} | "
            f"{row['treeshapiq_total_s']:.3f} | {row['speedup']:.3f} | "
            f"{row['max_abs_diff']:.3e} | {row['mean_abs_diff']:.3e} |"
        )
    path.write_text("\n".join(lines) + "\n")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    columns = [
        "order",
        "quadrature_total_s",
        "treeshapiq_total_s",
        "speedup",
        "max_abs_diff",
        "mean_abs_diff",
        "quadrature_nnz_max",
        "treeshapiq_nnz_max",
    ]
    with path.open("w", newline="") as fd:
        writer = csv.DictWriter(fd, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def plot(path: Path, rows: list[dict[str, object]]) -> None:
    orders = [int(row["order"]) for row in rows]
    speedups = [float(row["speedup"]) for row in rows]

    plt.figure(figsize=(5.8, 3.8))
    plt.plot(orders, speedups, marker="o", linewidth=2.4, color="#1f77b4")
    for order, speedup in zip(orders, speedups):
        plt.annotate(
            f"{speedup:.2f}x",
            (order, speedup),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
        )
    plt.xlabel("Interaction order")
    plt.ylabel("TreeSHAP-IQ / Quadrature runtime")
    plt.xticks(orders)
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_time(
    path: Path, rows: list[dict[str, object]], log_scale: bool = False
) -> None:
    orders = [int(row["order"]) for row in rows]
    quadrature = [float(row["quadrature_total_s"]) for row in rows]
    treeshapiq = [float(row["treeshapiq_total_s"]) for row in rows]

    plt.figure(figsize=(5.8, 3.8))
    plt.plot(orders, quadrature, marker="o", linewidth=2.4, label="Quadrature")
    plt.plot(orders, treeshapiq, marker="o", linewidth=2.4, label="TreeSHAP-IQ")
    plt.xlabel("Interaction order")
    plt.ylabel("Runtime (seconds)")
    plt.xticks(orders)
    if log_scale:
        plt.yscale("log")
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="cal_housing-sparse")
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-chart", type=Path, required=True)
    parser.add_argument("--out-time-chart", type=Path, default=None)
    parser.add_argument("--out-log-time-chart", type=Path, default=None)
    args = parser.parse_args()

    rows = load_rows(args.inputs, args.model)
    write_markdown(args.out_md, rows)
    write_csv(args.out_csv, rows)
    plot(args.out_chart, rows)
    if args.out_time_chart is not None:
        plot_time(args.out_time_chart, rows)
    if args.out_log_time_chart is not None:
        plot_time(args.out_log_time_chart, rows, log_scale=True)


if __name__ == "__main__":
    main()
