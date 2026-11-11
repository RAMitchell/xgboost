"""Resume the two runtime tables, checkpointing each model before continuing."""
# pylint: disable=too-many-locals,protected-access

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "experiments/2026-04-10-rapids-style-shap-benchmark"
MODELS = [
    f"{dataset}-{size}"
    for size in ("small", "sparse", "large")
    for dataset in ("cal_housing", "adult", "covtype", "fashion_mnist")
]


def sha256(path):
    """Hash a source, model or library for resume provenance."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    """Atomically checkpoint one suite output."""
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def main():
    """Run or resume both runtime tables under the recorded configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    from xgboost import core

    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    identity = {
        "revision": revision,
        "library": str(Path(core._LIB._name).resolve()),
        "library_sha256": sha256(Path(core._LIB._name)),
        "benchmark_sha256": sha256(BENCH / "quadratureshap_rapids_benchmark.py"),
        "model_sha256": {
            name: sha256(BENCH / "model-cache" / f"{name}.ubj") for name in MODELS
        },
        "niter": 10,
        "warmup": 3,
        "nthread": 16,
        "seed": 432,
        "prediction_timeout_seconds": 600,
        "case_timeout_seconds": None,
        "quadrature_rule": "standard_gauss_legendre",
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "environment": {
            key: os.environ.get(key)
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "PYTHONPATH",
            )
        },
    }
    manifest = out / "manifest.json"
    if manifest.exists():
        assert json.loads(manifest.read_text())["identity"] == identity, (
            "Run configuration changed; use a new output directory"
        )
    else:
        save(
            manifest,
            {
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "identity": identity,
            },
        )
        (out / "working.diff").write_text(
            subprocess.check_output(["git", "diff", "HEAD"], cwd=ROOT, text=True)
        )
    for mode, nrows in (("first-order", 1000), ("pairwise", 100)):
        folder = out / mode
        folder.mkdir(exist_ok=True)
        completed = []
        for model in MODELS:
            target = folder / f"{model}.json"
            if not target.exists():
                command = [
                    sys.executable,
                    "-u",
                    str(BENCH / "quadratureshap_rapids_benchmark.py"),
                    "--model",
                    model,
                    "--model-dir",
                    str(BENCH / "model-cache"),
                    "--nrows",
                    str(nrows),
                    "--niter",
                    "10",
                    "--warmup",
                    "3",
                    "--nthread",
                    "16",
                    "--prediction-timeout-seconds",
                    "600",
                    "--output",
                    str(target),
                ]
                if mode == "pairwise":
                    command.append("--interactions")
                (folder / f"{model}-command.json").write_text(
                    json.dumps(command, indent=2) + "\n"
                )
                print(
                    datetime.now(timezone.utc).isoformat(),
                    "START",
                    mode,
                    model,
                    flush=True,
                )
                with (folder / f"{model}.log").open("w") as log:
                    subprocess.run(
                        command,
                        cwd=ROOT,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
            payload = json.loads(target.read_text())
            assert payload["niter"] == 10 and payload["nrows"] == nrows
            assert len(payload["details"]) == 4
            completed.append(payload)
            merged = dict(completed[0])
            for key in ("models_table", "results_table", "details"):
                merged[key] = [row for item in completed for row in item[key]]
            merged["model_filter"] = (
                "all" if len(completed) == len(MODELS) else "partial"
            )
            merged["provenance_manifest"] = str(manifest)
            merged["completed_models"] = [item["model_filter"] for item in completed]
            save(out / f"{mode}.json", merged)
            subprocess.run(
                [
                    sys.executable,
                    str(BENCH / "summarize_runtime_variability.py"),
                    "--inputs",
                    str(out / f"{mode}.json"),
                    "--out-dir",
                    str(out / "tables"),
                ],
                cwd=ROOT,
                check=True,
                stdout=subprocess.DEVNULL,
            )
            failures = [
                f"{r['device']}/{r['algorithm']}: {r['error']}"
                for r in payload["details"]
                if r["error"]
            ]
            print(
                datetime.now(timezone.utc).isoformat(),
                "DONE",
                mode,
                model,
                failures,
                flush=True,
            )
    save(
        out / "complete.json", {"finished_utc": datetime.now(timezone.utc).isoformat()}
    )
    print("SUITE COMPLETE", flush=True)


if __name__ == "__main__":
    main()
