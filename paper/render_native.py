"""Render the current native runtime tables from recorded repeated-run samples."""

# pylint: disable=too-many-locals

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

METHODS = (
    ("cpu", "treeshap", "CPU TreeSHAP"),
    ("cpu", "quadratureshap", "CPU Q-TreeSHAP"),
    ("cuda", "treeshap", "GPU TreeSHAP"),
    ("cuda", "quadratureshap", "GPU Q-TreeSHAP"),
)


def sample_sd(row, payload):
    """Validate sample moments from the current repeated-run protocol."""
    import statistics

    if payload.get("std_ddof") != 1:
        raise ValueError("Expected current sample-SD results")
    samples = row.get("samples_s", [])
    if row.get("error") is not None:
        if samples:
            raise ValueError("Failed entries must discard all samples")
        return 0, None
    if not samples or len(samples) != payload["niter"]:
        raise ValueError("Complete entries require all raw timing samples")
    return len(samples), statistics.stdev(samples) if len(samples) > 1 else None


def summarize(payload, source):
    """Flatten saved timing entries without changing measured means."""
    rows = []
    seen = set()
    for detail in payload["details"]:
        key = (detail["model"], detail["device"], detail["algorithm"])
        if key in seen:
            raise ValueError(f"Duplicate benchmark case: {key}")
        seen.add(key)
        n, sd = sample_sd(detail, payload)
        mean = detail["mean_time_s"]
        rows.append(
            {
                "model": detail["model"],
                "device": detail["device"],
                "algorithm": detail["algorithm"],
                "n": n if mean is not None else 0,
                "mean_s": mean,
                "sample_sd_s": sd,
                "cv_percent": 100 * sd / mean if sd is not None and mean else None,
                "error": detail.get("error"),
                "source": str(source),
            }
        )
    return rows


def render(payload, rows):
    """Render a mean plus sample-SD table and its protocol notes."""
    lookup = {(r["model"], r["device"], r["algorithm"]): r for r in rows}
    models = list(dict.fromkeys(r["model"] for r in rows))
    header = ["Model"] + [m[2] for m in METHODS] + ["CPU speedup", "GPU speedup"]
    table = []
    tex = []
    for model in models:
        cells = [model]
        tex_cells = [model.replace("_", r"\_")]
        for device, algorithm, _ in METHODS:
            row = lookup.get((model, device, algorithm))
            if row is None:
                cell, tex_cell = "--", "--"
            elif row["mean_s"] is None:
                cell, tex_cell = "--", "--"
            elif row["sample_sd_s"] is None:
                cell = f"{row['mean_s']:.4g} (SD unavailable)"
                tex_cell = f"${row['mean_s']:.4g}$"
            else:
                cell = f"{row['mean_s']:.4g} ± {row['sample_sd_s']:.3g}"
                tex_cell = f"${row['mean_s']:.4g} \\pm {row['sample_sd_s']:.3g}$"
            cells.append(cell)
            tex_cells.append(tex_cell)
        for device in ("cpu", "cuda"):
            baseline = lookup.get((model, device, "treeshap"))
            candidate = lookup.get((model, device, "quadratureshap"))
            if baseline and candidate and baseline["mean_s"] and candidate["mean_s"]:
                ratio = baseline["mean_s"] / candidate["mean_s"]
                cells.append(f"{ratio:.2f}×")
                tex_cells.append(f"${ratio:.2f}\\times$")
            else:
                cells.append("--")
                tex_cells.append("--")
        table.append("| " + " | ".join(cells) + " |")
        tex.append(" & ".join(tex_cells) + r" \\")
    title = "Pairwise interactions" if payload["interactions"] else "First-order SHAP"
    notes = (
        f"{title}; {payload['nrows']} explained rows; {payload['niter']} repetitions. "
        "Times are mean ± sample standard deviation in wall-clock seconds."
    )
    markdown = (
        notes
        + "\n\n| "
        + " | ".join(header)
        + " |\n| "
        + " | ".join(["---"] * len(header))
        + " |\n"
        + "\n".join(table)
        + "\n"
    )
    latex = (
        "% "
        + notes
        + "\n\\begin{tabular}{lrrrrrr}\n\\toprule\n"
        + " & ".join(header)
        + r" \\"
        + "\n\\midrule\n"
        + "\n".join(tex)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
    return markdown, latex


def main():
    """Write paper-facing timing summaries from saved benchmark JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for source in args.inputs:
        payload = json.loads(source.read_text())
        rows = summarize(payload, source.resolve())
        md, tex = render(payload, rows)
        stem = args.out_dir / source.stem
        stem.with_suffix(".md").write_text(md)
        stem.with_suffix(".tex").write_text(tex)
        stem.with_suffix(".json").write_text(json.dumps(rows, indent=2) + "\n")
        with stem.with_suffix(".csv").open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(stem.with_suffix(".md"))


if __name__ == "__main__":
    main()
