"""Summarize order-3 benchmark JSON output as Markdown and CSV."""
# pylint: disable=missing-function-docstring,duplicate-code

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL_TABLE = (
    ROOT
    / "experiments/2026-04-10-rapids-style-shap-benchmark/models-only-unique-depth.json"
)


def markdown_table(rows: list[dict[str, object]]) -> str:
    columns = [
        ("model", "model"),
        ("n_features", "features"),
        ("num_trees", "trees"),
        ("max_max_depth", "max depth"),
        ("max_unique_feature_depth", "max unique depth"),
        ("quadrature_total_s", "quadrature s"),
        ("treeshapiq_total_s", "TreeSHAP-IQ s"),
        ("speedup", "speedup"),
        ("max_abs_diff", "max abs diff"),
        ("mean_abs_diff", "mean abs diff"),
    ]
    lines = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        values = []
        for key, _ in columns:
            value = row.get(key)
            if isinstance(value, float):
                if key.endswith("_s") or key == "speedup":
                    values.append(f"{value:.3f}")
                else:
                    values.append(f"{value:.3e}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def csv_table(rows: list[dict[str, object]]) -> str:
    columns = [
        "model",
        "n_features",
        "num_trees",
        "max_max_depth",
        "max_unique_feature_depth",
        "quadrature_total_s",
        "treeshapiq_total_s",
        "speedup",
        "max_abs_diff",
        "mean_abs_diff",
    ]
    lines = [",".join(columns)]
    for row in rows:
        lines.append(",".join(str(row.get(column, "")) for column in columns))
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text())
    model_payload = json.loads(MODEL_TABLE.read_text())
    stats = {row["model"]: row for row in model_payload["models_table"]}

    rows = []
    for row in payload["results"]:
        if "error" in row:
            rows.append(row)
            continue
        rows.append({**row, **stats.get(row["model"], {})})

    args.out_md.write_text(markdown_table(rows))
    args.out_csv.write_text(csv_table(rows))


if __name__ == "__main__":
    main()
