"""Repeated Table 4 timings with independent per-call and setup deadlines."""
# pylint: disable=duplicate-code
# pylint: disable=broad-exception-caught,cell-var-from-loop,line-too-long,missing-function-docstring,protected-access,too-many-arguments,too-many-branches,too-many-locals,too-many-positional-arguments,too-many-statements,unnecessary-lambda-assignment

import argparse
import hashlib
import importlib.metadata
import json
import multiprocessing as mp
import os
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from adapters import (
    CACHE,
    OLD,
    ROOT,
    canonical_trees,
    explain,
    load_old,
    make_explainer,
    pack,
)

MODELS = [
    "cal_housing-small",
    "cal_housing-sparse",
    "covtype-small",
    "covtype-large",
    "covtype-sparse",
    "cal_housing-large",
]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, data):
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(p)


def worker(conn, model, method, row, folder, niter, warmup, delays):
    try:
        conn.send(
            {
                "event": "start",
                "phase": "setup",
                "index": 0,
                "started_monotonic": time.monotonic(),
            }
        )
        started = time.perf_counter()
        if delays is None:
            import xgboost as xgb
            from numba import set_num_threads

            set_num_threads(1)
            old = load_old()
            b = xgb.Booster()
            b.load_model(CACHE / f"{model}.ubj")
            b.set_param({"nthread": 1, "device": "cpu"})
            trees = canonical_trees(old, b)
            obj = make_explainer(old, trees, b.num_features(), method)
            predict = lambda: explain(old, obj, row, method)
            metadata = {"num_trees": len(trees), "num_features": b.num_features()}
        else:
            counter = [0]

            def predict():
                delay = delays[min(counter[0], len(delays) - 1)]
                counter[0] += 1
                time.sleep(delay)
                return {(0, 1, 2): 0.125}

            metadata = {}
        setup_s = time.perf_counter() - started
        conn.send({"event": "end", "phase": "setup", "index": 0, "elapsed_s": setup_s})
        samples = []
        for index in range(warmup + niter):
            phase = "warmup" if index < warmup else "timed"
            conn.send(
                {
                    "event": "start",
                    "phase": phase,
                    "index": index,
                    "started_monotonic": time.monotonic(),
                }
            )
            started = time.perf_counter()
            values = predict()
            elapsed = time.perf_counter() - started
            conn.send(
                {"event": "end", "phase": phase, "index": index, "elapsed_s": elapsed}
            )
            keys, array = pack(values)
            if not np.isfinite(array).all():
                raise ValueError("Nonfinite third-order interaction")
            np.savez_compressed(
                folder / f"{phase}-{index:02d}.npz", keys=keys, values=array
            )
            if phase == "timed":
                samples.append(elapsed)
        result = {
            "model": model,
            "method": method,
            "status": "complete",
            "error": None,
            "setup_s": setup_s,
            "samples_s": samples,
            "n_samples": len(samples),
            "mean_s": statistics.mean(samples),
            "sample_sd_s": statistics.stdev(samples) if niter > 1 else None,
            "min_s": min(samples),
            "max_s": max(samples),
            **metadata,
        }
        conn.send({"event": "result", "result": result})
    except Exception as error:
        conn.send(
            {
                "event": "result",
                "result": {
                    "status": "failed",
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                },
            }
        )
    finally:
        conn.close()


def run_case(
    model,
    method,
    row,
    folder,
    niter=10,
    warmup=3,
    limit=600,
    setup_limit=600,
    progress=None,
    delays=None,
):
    folder.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")
    recv, send = ctx.Pipe(duplex=False)
    proc = ctx.Process(
        target=worker, args=(send, model, method, row, folder, niter, warmup, delays)
    )
    proc.start()
    send.close()
    active = None
    calls = []
    result = None
    try:
        while result is None:
            deadline = setup_limit if active and active["phase"] == "setup" else limit
            if (
                active
                and time.monotonic() > active["started_monotonic"] + deadline + 0.1
                and not recv.poll()
            ):
                result = {
                    "status": "timeout",
                    "error": f"{active['phase']} call exceeded {deadline:g}s",
                    "timed_out_call": {
                        **active,
                        "observed_elapsed_s": time.monotonic()
                        - active["started_monotonic"],
                    },
                }
                break
            if recv.poll(0.05):
                try:
                    event = recv.recv()
                except EOFError:
                    result = {
                        "status": "failed",
                        "error": f"Worker exited without result ({proc.exitcode})",
                    }
                    break
                if event["event"] == "start":
                    active = event
                elif event["event"] == "end":
                    calls.append({k: event[k] for k in ["phase", "index", "elapsed_s"]})
                    deadline = setup_limit if event["phase"] == "setup" else limit
                    if event["elapsed_s"] > deadline:
                        result = {
                            "status": "timeout",
                            "error": f"{event['phase']} call exceeded {deadline:g}s",
                            "timed_out_call": event,
                        }
                    active = None
                elif event["event"] == "result":
                    result = event["result"]
                else:
                    raise ValueError(event)
                if progress:
                    progress(event, calls)
            elif not proc.is_alive() and not recv.poll():
                result = {
                    "status": "failed",
                    "error": f"Worker exited without result ({proc.exitcode})",
                }
    finally:
        if proc.is_alive():
            proc.join(timeout=1 if result and result.get("error") is None else 0)
            if proc.is_alive():
                proc.terminate()
        proc.join()
        recv.close()
    result.update(
        model=model,
        method=method,
        prediction_calls=calls,
        prediction_timeout_seconds=limit,
        setup_timeout_seconds=setup_limit,
    )
    if result.get("error") is not None:
        result.update(
            samples_s=[],
            n_samples=0,
            mean_s=None,
            sample_sd_s=None,
            min_s=None,
            max_s=None,
        )
    return result


def summarize(out, models):
    results = []
    text = [
        "| Model | TreeSHAP-IQ (s) | Quadrature (s) | Speedup |",
        "|---|---:|---:|---:|",
    ]
    for model in models:
        methods = {}
        for method in ["quadrature", "treeshapiq"]:
            p = out / model / method / "results.json"
            methods[method] = (
                json.loads(p.read_text(encoding="utf-8"))
                if p.exists()
                else {"status": "pending", "mean_s": None}
            )
        a = methods["treeshapiq"]["mean_s"]
        b = methods["quadrature"]["mean_s"]
        speedup = a / b if a and b else None
        result = {"model": model, "methods": methods, "speedup": speedup}
        results.append(result)

        def cell(r):
            return (
                f"{r['mean_s']:.6g} ± {r['sample_sd_s']:.3g}"
                if r["status"] == "complete"
                else r["status"]
            )

        text.append(
            f"| {model} | {cell(methods['treeshapiq'])} | {cell(methods['quadrature'])} | {speedup:.2f}× |"
            if speedup
            else f"| {model} | {cell(methods['treeshapiq'])} | {cell(methods['quadrature'])} | -- |"
        )
    save(out / "results.json", {"results": results})
    (out / "summary.md").write_text("\n".join(text) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--niter", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--limit", type=float, default=600)
    parser.add_argument("--setup-limit", type=float, default=600)
    args = parser.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.niter < 2 or args.warmup < 1 or min(args.limit, args.setup_limit) <= 0:
        parser.error("Need >=2 repeats, >=1 warm-up and positive limits")
    import xgboost as xgb

    identity = {
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "models": args.models,
        "niter": args.niter,
        "warmup": args.warmup,
        "rows": 1,
        "seed": 432,
        "order": 3,
        "index": "SII",
        "points": 8,
        "quadrature_rule": "standard_gauss_legendre",
        "precision": "float64",
        "output_scope": "sum of all tree outputs, including all multiclass groups, matching original Table4",
        "prediction_timeout_seconds": args.limit,
        "setup_timeout_seconds": args.setup_limit,
        "affinity": sorted(os.sched_getaffinity(0)),
        "environment": {
            k: os.environ.get(k)
            for k in [
                "PYTHONPATH",
                "CUDA_VISIBLE_DEVICES",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMBA_NUM_THREADS",
                "NUMBA_CACHE_DIR",
            ]
        },
        "versions": {
            name: importlib.metadata.version(name)
            for name in ["shapiq", "numpy", "numba", "xgboost"]
        },
        "source_sha256": {
            p.name: sha(p)
            for p in [
                Path(__file__),
                Path(__file__).with_name("adapters.py"),
                OLD,
                ROOT
                / "experiments/2026-04-10-rapids-style-shap-benchmark/quadratureshap_rapids_benchmark.py",
            ]
        },
        "model_sha256": {m: sha(CACHE / f"{m}.ubj") for m in args.models},
        "library": str(xgb.core._LIB._name),
        "library_sha256": sha(xgb.core._LIB._name),
    }
    if (out / "manifest.json").exists():
        assert (
            json.loads((out / "manifest.json").read_text(encoding="utf-8")) == identity
        ), "Changed protocol/source: use a fresh output directory"
    else:
        save(out / "manifest.json", identity)
    sources = out / "sources"
    sources.mkdir(exist_ok=True)
    for p in [Path(__file__), Path(__file__).with_name("adapters.py"), OLD]:
        (sources / p.name).write_bytes(p.read_bytes())
    sys.path.insert(0, str(ROOT / "experiments/2026-04-10-rapids-style-shap-benchmark"))
    from quadratureshap_rapids_benchmark import get_test_datasets

    datasets = {d.name: d for d in get_test_datasets()}
    for model in args.models:
        folder = out / model
        folder.mkdir(exist_ok=True)
        path = folder / "row.npy"
        if path.exists():
            row = np.load(path)
        else:
            data = datasets[model.rsplit("-", 1)[0]].test_input(1, 432)
            row = np.asarray(data, dtype=np.float32).astype(np.float64).reshape(-1)
            assert np.isfinite(row).all()
            np.save(path, row)
        order = ["quadrature", "treeshapiq"]
        np.random.default_rng(432 + args.models.index(model)).shuffle(order)
        for method in order:
            case = folder / method
            target = case / "results.json"
            if target.exists():
                continue

            def progress(event, calls):
                save(
                    out / "status.json",
                    {
                        "updated_utc": datetime.now(timezone.utc).isoformat(),
                        "model": model,
                        "method": method,
                        "last_event": event,
                        "completed_calls": calls,
                    },
                )
                if event["event"] == "end":
                    print(
                        "CALL",
                        model,
                        method,
                        event["phase"],
                        event["index"],
                        event["elapsed_s"],
                        flush=True,
                    )

            print("START", model, method, flush=True)
            result = run_case(
                model,
                method,
                row,
                case,
                args.niter,
                args.warmup,
                args.limit,
                args.setup_limit,
                progress,
            )
            save(target, result)
            summarize(out, args.models)
            print(
                "DONE",
                model,
                method,
                result["status"],
                result.get("mean_s"),
                flush=True,
            )
    save(
        out / "complete.json", {"finished_utc": datetime.now(timezone.utc).isoformat()}
    )
    print("THIRD-ORDER REPEATS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
