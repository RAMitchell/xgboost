"""Build generic order-scaling tables and charts from benchmark JSON outputs."""
# pylint: disable=missing-function-docstring

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_rows(paths: list[Path]) -> list[dict[str, object]]:
    rows = []
    for path in paths:
        payload = json.loads(path.read_text())
        for row in payload["results"]:
            if "error" not in row:
                rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    columns = [
        "model",
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


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "| model | order | quadrature s | TreeSHAP-IQ s | speedup | max abs diff | mean abs diff |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in sorted(rows, key=lambda r: (str(r["model"]), int(r["order"]))):
        lines.append(
            f"| {row['model']} | {row['order']} | {row['quadrature_total_s']:.3f} | "
            f"{row['treeshapiq_total_s']:.3f} | {row['speedup']:.3f} | "
            f"{row['max_abs_diff']:.3e} | {row['mean_abs_diff']:.3e} |"
        )
    path.write_text("\n".join(lines) + "\n")


def plot_speedup(path: Path, rows: list[dict[str, object]]) -> None:
    by_model = {}
    for row in rows:
        by_model.setdefault(row["model"], []).append(row)

    plt.figure(figsize=(7.2, 4.2))
    for model, model_rows in sorted(by_model.items()):
        model_rows = sorted(model_rows, key=lambda r: r["order"])
        plt.plot(
            [r["order"] for r in model_rows],
            [r["speedup"] for r in model_rows],
            marker="o",
            linewidth=2.2,
            label=model.replace("_", "\\_"),
        )

    plt.xlabel("Interaction order")
    plt.ylabel("TreeSHAP-IQ / Quadrature runtime")
    plt.xticks(sorted({int(row["order"]) for row in rows}))
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_time(path: Path, rows: list[dict[str, object]]) -> None:
    models = sorted({str(row["model"]) for row in rows})
    orders = sorted({int(row["order"]) for row in rows})
    _, axes = plt.subplots(1, len(models), figsize=(8.4, 3.8), sharey=False)
    if len(models) == 1:
        axes = [axes]
    for ax, model in zip(axes, models):
        model_rows = {int(row["order"]): row for row in rows if row["model"] == model}
        ax.plot(
            orders,
            [model_rows[o]["quadrature_total_s"] for o in orders],
            marker="o",
            label="Quadrature",
        )
        ax.plot(
            orders,
            [model_rows[o]["treeshapiq_total_s"] for o in orders],
            marker="o",
            label="TreeSHAP-IQ",
        )
        ax.set_title(model.replace("_", "\\_"))
        ax.set_xlabel("Order")
        ax.set_xticks(orders)
        ax.grid(True, axis="y", alpha=0.25)
    axes[0].set_ylabel("Runtime s")
    axes[-1].legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-speedup", type=Path, required=True)
    parser.add_argument("--out-time", type=Path, required=True)
    args = parser.parse_args()

    rows = load_rows(args.inputs)
    write_csv(args.out_csv, rows)
    write_markdown(args.out_md, rows)
    plot_speedup(args.out_speedup, rows)
    plot_time(args.out_time, rows)


if __name__ == "__main__":
    main()
