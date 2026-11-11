"""Reuse individually bounded first-order calls; replay old whole-case failures."""
# pylint: disable=duplicate-code
# pylint: disable=cell-var-from-loop,line-too-long,missing-function-docstring,protected-access,too-many-branches,too-many-locals,too-many-statements

import copy
import hashlib
import importlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import xgboost as xgb
from per_prediction_timeout import run_case

ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
SOURCE = EXP / "full-suite-10"
OUT = EXP / "first-order-per-prediction600"
BENCH = ROOT / "experiments/2026-04-10-rapids-style-shap-benchmark"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def update_table(data):
    bykey = {(r["model"], r["device"], r["algorithm"]): r for r in data["details"]}
    for row in data["results_table"]:
        for dev, a, b, s in [
            ("cpu", "TreeSHAP", "QuadratureTreeSHAP", "QuadratureTreeSHAP Speedup"),
            (
                "cuda",
                "GPUTreeShap",
                "QuadratureTreeSHAP (GPU)",
                "QuadratureTreeSHAP (GPU) Speedup",
            ),
        ]:
            first = bykey[row["model"], dev, "treeshap"]["mean_time_s"]
            second = bykey[row["model"], dev, "quadratureshap"]["mean_time_s"]
            row[a] = first
            row[b] = second
            row[s] = first / second if first and second else None
    save(OUT / "first-order.json", data)


def main():
    OUT.mkdir(exist_ok=True)
    original = json.loads((SOURCE / "first-order.json").read_text(encoding="utf-8"))
    prior = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))[
        "identity"
    ]
    identity = {
        "source_results_sha256": sha(SOURCE / "first-order.json"),
        "library_sha256": sha(xgb.core._LIB._name),
        "benchmark_sha256": sha(BENCH / "quadratureshap_rapids_benchmark.py"),
        "prediction_timeout_seconds": 600,
        "script_sha256": sha(__file__),
        "watchdog_sha256": sha(Path(__file__).with_name("per_prediction_timeout.py")),
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
    if (OUT / "manifest.json").exists():
        assert (
            json.loads((OUT / "manifest.json").read_text(encoding="utf-8")) == identity
        )
    else:
        save(OUT / "manifest.json", identity)
    starts = {}
    for line in (SOURCE / "runner.log").read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) >= 4 and fields[1:3] == ["START", "first-order"]:
            starts[fields[3]] = datetime.fromisoformat(fields[0]).timestamp()
    data = copy.deepcopy(original)
    audit = []
    pending = []
    for model in original["completed_models"]:
        path = SOURCE / "first-order" / f"{model}.json"
        source = json.loads(path.read_text(encoding="utf-8"))
        model_wall = path.stat().st_mtime - starts[model]
        timed_total = sum(sum(r.get("samples_s", [])) for r in source["details"])
        failed_lower = 0.0
        for r in source["details"]:
            match = re.search(r"exceeded ([0-9.]+)s", r.get("error") or "")
            if match:
                failed_lower += float(match[1])
        # After removing recorded timed calls and guaranteed failed-case time,
        # the remainder bounds any one successful setup/accuracy/warm-up call.
        untimed_bound = model_wall - timed_total - failed_lower
        assert untimed_bound >= -0.01, (model, untimed_bound)
        for index, row in enumerate(data["details"]):
            if row["model"] != model:
                continue
            item = {
                "model": model,
                "device": row["device"],
                "algorithm": row["algorithm"],
                "model_result_sha256": sha(path),
                "model_wall_upper_s": model_wall,
                "successful_timed_total_s": timed_total,
                "failed_case_lower_s": failed_lower,
                "untimed_call_upper_s": max(0.0, untimed_bound),
            }
            if row["error"] is None:
                item["max_timed_call_s"] = max(row["samples_s"])
                assert (
                    item["max_timed_call_s"] <= 600
                    and item["untimed_call_upper_s"] <= 600
                ), item
                item["action"] = (
                    "reuse: every timed call and conservative untimed-call bound within 600s"
                )
            elif "exceeded" in row["error"]:
                item["action"] = (
                    "replay: whole-case timeout gives no individual-call outcome"
                )
                row["error"] = "Pending: remeasurement with 600s per prediction"
                row["status"] = "pending"
                pending.append((index, item))
            else:
                item["action"] = "retain unsupported case"
            audit.append(item)
    data.update(
        prediction_timeout_seconds=600,
        timeout_scope="one explanation call; any over-limit call voids the entire algorithm/device entry",
        source_results=str(SOURCE / "first-order.json"),
        prediction_timeout_audit=str(OUT / "audit.json"),
        pending_cases=[
            {k: e[k] for k in ["model", "device", "algorithm"]} for i, e in pending
        ],
    )
    save(OUT / "audit.json", audit)
    update_table(data)
    print("Replaying", data["pending_cases"], flush=True)
    sys.path.insert(0, str(BENCH))
    bench = importlib.import_module("quadratureshap_rapids_benchmark")
    for index, item in pending:
        name = item["model"]
        target = OUT / f"{name}-{item['device']}-{item['algorithm']}.json"
        if target.exists():
            result = json.loads(target.read_text(encoding="utf-8"))
        else:
            dataset = next(
                d for d in bench.get_test_datasets() if d.name == name.rsplit("-", 1)[0]
            )
            b = xgb.Booster()
            b.load_model(BENCH / "model-cache" / f"{name}.ubj")
            assert (
                sha(BENCH / "model-cache" / f"{name}.ubj")
                == prior["model_sha256"][name]
            )
            b.set_param({"device": "cpu", "nthread": 16})
            x = dataset.test_input(1000, 432)
            dm = xgb.DMatrix(x, enable_categorical=True, nthread=16)
            margin = b.predict(dm, output_margin=True)

            def progress(event, calls):
                save(
                    OUT / "status.json",
                    {
                        "updated_utc": datetime.now(timezone.utc).isoformat(),
                        "model": name,
                        "device": item["device"],
                        "algorithm": item["algorithm"],
                        "last_event": event,
                        "completed_prediction_calls": calls,
                    },
                )
                if event["event"] == "end":
                    print(
                        "CALL",
                        event["phase"],
                        event["index"],
                        event["elapsed_s"],
                        flush=True,
                    )

            started = time.monotonic()
            result = run_case(
                b,
                x,
                item["device"],
                item["algorithm"],
                10,
                margin,
                warmup=3,
                nthread=16,
                limit=600,
                progress=progress,
            )
            result["whole_experiment_elapsed_s"] = time.monotonic() - started
            save(target, result)
        data["details"][index] = {
            "model": name,
            "device": item["device"],
            "algorithm": item["algorithm"],
            **result,
        }
        data["pending_cases"] = []
        update_table(data)
        print("RESULT", name, result["error"], result.get("mean_time_s"), flush=True)
    save(
        OUT / "complete.json", {"finished_utc": datetime.now(timezone.utc).isoformat()}
    )
    print("PER-PREDICTION LIMIT EXPORT COMPLETE", flush=True)


if __name__ == "__main__":
    main()
