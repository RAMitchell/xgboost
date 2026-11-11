"""Export completed matched-count cases, leaving unsupported/failed states explicit."""
# pylint: disable=duplicate-code
# pylint: disable=line-too-long,missing-function-docstring,redefined-outer-name

import argparse
import csv
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("directory", type=Path)
a = p.parse_args()
root = a.directory
rows = []
lines = [
    "# Matched-count first-order comparison",
    "",
    "Both implementations use float64 standard Gauss–Legendre rules. Times are mean ± sample SD of full batches of ten rows; each trial reuses the same input rows. Native counts follow TreeGrad per-tree ceil(min(D,F)/2); distinct counts use per-tree ceil(d/2). Construction, warm-up, and JIT are excluded. All class outputs are kept separate.",
    "",
    "| Model | Native matched speedup | Distinct matched speedup | Native matched, TG nodes cached | Fixed8 vs native TG |",
    "|---|---:|---:|---:|---:|",
]
for folder in sorted(root.iterdir()):
    if not folder.is_dir():
        continue
    done = folder / "complete.json"
    failed = folder / "failure.json"
    if not done.exists():
        if failed.exists():
            lines.append(f"| {folder.name} | Failed | Failed | Failed | Failed |")
        continue
    marker = json.loads(done.read_text(encoding="utf-8"))
    if marker["status"] == "unsupported":
        lines.append(f"| {folder.name} | Unsupported categories | — | — | — |")
        continue
    m = json.loads((folder / "results.json").read_text(encoding="utf-8"))
    s = m["speedups"]
    lines.append(
        "| "
        + folder.name
        + " | "
        + " | ".join(
            f"{s[k]:.3f}×"
            for k in [
                "matched_native",
                "matched_distinct",
                "cached_native",
                "fixed8_vs_native",
            ]
        )
        + " |"
    )
    for method, r in m["methods"].items():
        rows.append(
            {
                "model": folder.name,
                "method": method,
                "rows": m["rows"],
                "niter": m["niter"],
                "mean_s": r["mean_s"],
                "sample_sd_s": r["sample_sd_s"],
                "max_abs_diff_vs_tg_distinct": r["max_abs_diff_vs_tg_distinct"],
                "nrmse_vs_tg_distinct": r["nrmse_vs_tg_distinct"],
                "samples_s": json.dumps(r["samples_s"]),
            }
        )
(root / "summary.md").write_text("\n".join(lines) + "\n")
if rows:
    with (root / "summary.csv").open("w", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
print("Rendered", len(rows), "completed method/model cases")

for policy, tg, q in [
    ("native", "tg_native", "q_native_count"),
    ("distinct", "tg_distinct", "q_distinct_count"),
]:
    lines = [
        f"# Table 5: matched {policy} counts, float64",
        "",
        "Both methods use the same count per tree. The table shows the maximum count per tree; per-tree distributions are retained in JSON. Times are seconds, mean ± sample SD across repeated ten-row batches.",
        "",
        "| Model | Max points/tree | TreeGrad | QuadratureTreeSHAP | Speedup |",
        "|---|---:|---:|---:|---:|",
    ]
    for done in sorted(root.glob("*/complete.json")):
        folder = done.parent
        marker = json.loads(done.read_text(encoding="utf-8"))
        if marker["status"] == "unsupported":
            lines.append(f"| {folder.name} | — | Unsupported categories | — | — |")
            continue
        m = json.loads((folder / "results.json").read_text(encoding="utf-8"))
        counts = m[policy + "_counts"]

        def cell(method):
            r = m["methods"][method]
            return (
                f"{r['mean_s']:.5g} ± {r['sample_sd_s']:.3g}"
                if r["sample_sd_s"] is not None
                else f"{r['mean_s']:.5g} (SD unavailable)"
            )

        lines.append(
            f"| {folder.name} | {max(counts)} | {cell(tg)} | {cell(q)} | {m['methods'][tg]['mean_s'] / m['methods'][q]['mean_s']:.3f}× |"
        )
    (root / f"table5-matched-{policy}.md").write_text("\n".join(lines) + "\n")
