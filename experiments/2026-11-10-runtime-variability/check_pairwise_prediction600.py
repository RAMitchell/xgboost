"""Resolve two pairwise aggregate failures with a 600s limit on each call."""
# pylint: disable=duplicate-code
# pylint: disable=cell-var-from-loop,missing-function-docstring,protected-access

import hashlib
import importlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import xgboost as xgb
from pairwise_prediction_timeout import run_case

ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
OUT = EXP / "pairwise-per-prediction600"
BENCH = ROOT / "experiments/2026-04-10-rapids-style-shap-benchmark"
MODELS = ["fashion_mnist-large", "fashion_mnist-sparse"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, data):
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(p)


def main():
    OUT.mkdir(exist_ok=True)
    prior = json.loads(
        (EXP / "full-suite-10/manifest.json").read_text(encoding="utf-8")
    )["identity"]
    identity = {
        "models": MODELS,
        "interactions": True,
        "nrows": 100,
        "niter": 10,
        "warmup": 3,
        "nthread": 16,
        "prediction_timeout_seconds": 600,
        "seed": 432,
        "revision": "7ee42815b",
        "library_path": str(xgb.core._LIB._name),
        "library_sha256": sha(xgb.core._LIB._name),
        "benchmark_sha256": sha(BENCH / "quadratureshap_rapids_benchmark.py"),
        "driver_sha256": sha(__file__),
        "watchdog_sha256": sha(EXP / "pairwise_prediction_timeout.py"),
        "model_sha256": {
            name: sha(BENCH / "model-cache" / f"{name}.ubj") for name in MODELS
        },
        "affinity": sorted(os.sched_getaffinity(0)),
        "environment": {
            k: os.environ.get(k)
            for k in [
                "PYTHONPATH",
                "CUDA_VISIBLE_DEVICES",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
            ]
        },
    }
    assert identity["library_sha256"] == prior["library_sha256"]
    assert identity["benchmark_sha256"] == prior["benchmark_sha256"]
    assert (
        identity["affinity"] == prior["cpu_affinity"]
        and identity["environment"] == prior["environment"]
    )
    assert all(identity["model_sha256"][m] == prior["model_sha256"][m] for m in MODELS)
    if (OUT / "manifest.json").exists():
        assert (
            json.loads((OUT / "manifest.json").read_text(encoding="utf-8")) == identity
        )
    else:
        save(OUT / "manifest.json", identity)
    (OUT / "sources").mkdir(exist_ok=True)
    for p in [Path(__file__), EXP / "pairwise_prediction_timeout.py"]:
        (OUT / "sources" / p.name).write_bytes(p.read_bytes())
    sys.path.insert(0, str(BENCH))
    bench = importlib.import_module("quadratureshap_rapids_benchmark")
    results = []
    for name in MODELS:
        target = OUT / f"{name}-cpu-treeshap.json"
        if target.exists():
            results.append(json.loads(target.read_text(encoding="utf-8")))
            continue
        dataset = next(
            d for d in bench.get_test_datasets() if d.name == name.rsplit("-", 1)[0]
        )
        b = xgb.Booster()
        b.load_model(BENCH / "model-cache" / f"{name}.ubj")
        b.set_param({"device": "cpu", "nthread": 16})
        x = dataset.test_input(100, 432)

        def progress(event, calls):
            save(
                OUT / "status.json",
                {
                    "updated_utc": datetime.now(timezone.utc).isoformat(),
                    "model": name,
                    "last_event": event,
                    "completed_prediction_calls": calls,
                },
            )
            if event["event"] == "end":
                print(
                    "CALL",
                    name,
                    event["phase"],
                    event["index"],
                    event["elapsed_s"],
                    flush=True,
                )

        print("START", name, flush=True)
        started = time.monotonic()
        result = run_case(
            b,
            x,
            "cpu",
            "treeshap",
            10,
            None,
            warmup=3,
            nthread=16,
            limit=600,
            progress=progress,
        )
        result.update(
            model=name,
            device="cpu",
            algorithm="treeshap",
            whole_experiment_elapsed_s=time.monotonic() - started,
        )
        save(target, result)
        results.append(result)
        print("RESULT", name, result["error"], result["mean_time_s"], flush=True)
    save(
        OUT / "complete.json",
        {"finished_utc": datetime.now(timezone.utc).isoformat(), "results": results},
    )
    print("PAIRWISE PER-CALL CHECK COMPLETE", flush=True)


if __name__ == "__main__":
    main()
