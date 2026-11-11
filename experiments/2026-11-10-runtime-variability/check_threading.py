"""Compare CPU threading configurations on fixed cached paper models."""
# pylint: disable=duplicate-code
# pylint: disable=missing-function-docstring

import argparse
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "experiments/2026-04-10-rapids-style-shap-benchmark"
CONFIGS = {
    "pinned16": ("20-35", "16", 16),
    "wide16": ("0-79", "80", 16),
    "wide40": ("0-79", "80", 40),
    "wide80": ("0-79", "80", 80),
    "automatic": ("0-79", None, None),
}
MODELS = ("cal_housing-small", "adult-small", "cal_housing-sparse")


def worker(config, output):
    from threadpoolctl import threadpool_info

    spec = importlib.util.spec_from_file_location(
        "thread_bench", BENCH / "quadratureshap_rapids_benchmark.py"
    )
    bench = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = bench
    spec.loader.exec_module(bench)
    nthread = CONFIGS[config][2]
    payload = {
        "niter": 10,
        "warmup": 3,
        "nrows": 1000,
        "std_ddof": 1,
        "nthread": nthread,
        "config": config,
        "interactions": False,
        "affinity": sorted(os.sched_getaffinity(0)),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "threadpools": threadpool_info(),
        "details": [],
        "results_table": [],
    }
    for name in MODELS:
        model = bench.get_models(name, BENCH / "model-cache", False)[0]
        if nthread is not None:
            model.booster.set_param({"nthread": nthread})
        x = model.dataset.test_input(1000, 432)
        data = bench.xgb.DMatrix(x, enable_categorical=True, nthread=nthread)
        result, details = bench.benchmark_model(
            model, x, data, 10, False, None, 3, nthread, ("cpu",)
        )
        payload["results_table"].append(result)
        payload["details"].extend(details)
        output.write_text(json.dumps(payload, indent=2) + "\n")
        print(config, name, result["QuadratureTreeSHAP Speedup"], flush=True)


def active_children(pid):
    children = Path(f"/proc/{pid}/task/{pid}/children")
    if not children.exists():
        return []
    alive = []
    for child in children.read_text(encoding="utf-8").split():
        try:
            state = (
                Path(f"/proc/{child}/stat")
                .read_text(encoding="utf-8")
                .split(") ", 1)[1]
                .split()[0]
            )
            if state != "Z":
                alive.append(child)
        except FileNotFoundError:
            pass
    return alive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--worker", choices=CONFIGS)
    parser.add_argument("--suite-pid", type=int)
    args = parser.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(args.worker, out / f"{args.worker}.json")
        return
    pid = args.suite_pid
    # Interruptions still release the scheduler; existing worker timing is untouched.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    try:
        if pid:
            os.kill(pid, signal.SIGSTOP)
            while active_children(pid):
                print(
                    "Waiting for active suite model:", active_children(pid), flush=True
                )
                time.sleep(30)
        for name, (cpus, omp, _) in CONFIGS.items():
            env = dict(
                os.environ,
                PYTHONPATH=str(ROOT / "python-package"),
                OPENBLAS_NUM_THREADS="1",
            )
            env.pop("OMP_NUM_THREADS", None)
            if omp is not None:
                env["OMP_NUM_THREADS"] = omp
            cmd = [
                "taskset",
                "-c",
                cpus,
                sys.executable,
                "-u",
                str(Path(__file__).resolve()),
                "--worker",
                name,
                "--out-dir",
                str(out),
            ]
            (out / f"{name}-command.json").write_text(
                json.dumps({"command": cmd, "OMP_NUM_THREADS": omp}, indent=2) + "\n"
            )
            print("START", name, flush=True)
            with (out / f"{name}.log").open("w", encoding="utf-8") as log:
                subprocess.run(
                    cmd,
                    env=env,
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            print("DONE", name, flush=True)
    finally:
        if pid:
            try:
                os.kill(pid, signal.SIGCONT)
                print("Resumed suite scheduler", pid, flush=True)
            except ProcessLookupError:
                pass


if __name__ == "__main__":
    main()
