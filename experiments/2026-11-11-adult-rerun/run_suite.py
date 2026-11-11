"""Sequential, resumable rerun of every paper experiment involving canonical Adult."""

# pylint: disable=missing-function-docstring,invalid-name,line-too-long,too-many-locals,too-many-statements
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUN = HERE / "full"
BENCH = (
    ROOT
    / "experiments/2026-04-10-rapids-style-shap-benchmark/quadratureshap_rapids_benchmark.py"
)
CACHE = BENCH.parent / "model-cache"
PYTHON = "/home/nfs/rorym/anaconda3/envs/xgboost/bin/python"
CPU_RUNTIME = (
    ROOT / "experiments/2026-11-10-high-dimensional-accuracy/python-runtime-standard"
)
MODELS = [f"adult-{size}" for size in ("small", "large", "sparse")]


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    archive = RUN / "historical-model-cache"
    if not archive.exists():
        archive.mkdir()
        for model in MODELS:
            shutil.copy2(CACHE / f"{model}.ubj", archive / f"{model}.ubj")
        save(
            archive / "manifest.json",
            {p.name: digest(p) for p in archive.glob("*.ubj")},
        )
    base = os.environ.copy()
    base.update(OPENBLAS_NUM_THREADS="1", NUMBA_CACHE_DIR=str(RUN / "numba-cache"))
    gpu = base | {
        "PYTHONPATH": str(ROOT / "python-package"),
        "CUDA_VISIBLE_DEVICES": "1",
        "OMP_NUM_THREADS": "16",
        "NUMBA_NUM_THREADS": "16",
    }
    cpu = base | {
        "PYTHONPATH": str(CPU_RUNTIME),
        "CUDA_VISIBLE_DEVICES": "-1",
        "OMP_NUM_THREADS": "16",
        "NUMBA_NUM_THREADS": "16",
    }
    single = cpu | {"OMP_NUM_THREADS": "1", "NUMBA_NUM_THREADS": "1"}
    jobs = [
        (
            "validate-encoding",
            gpu,
            "20-35",
            [str(HERE / "validate_encoding.py"), str(archive)],
        )
    ]
    jobs.append(
        (
            "train-models",
            gpu,
            "20-35",
            [
                str(BENCH),
                "--model",
                "adult",
                "--model-dir",
                str(CACHE),
                "--force-retrain",
                "--models-only",
                "--output",
                str(RUN / "models.json"),
            ],
        )
    )
    for order, rows in ((1, 1000), (2, 100)):
        for model in MODELS:
            name = f"order{order}-{model}"
            arguments = [
                str(BENCH),
                "--model",
                model,
                "--model-dir",
                str(CACHE),
                "--nrows",
                str(rows),
                "--niter",
                "10",
                "--warmup",
                "3",
                "--nthread",
                "16",
                "--devices",
                "cpu",
                "cuda",
                "--prediction-timeout-seconds",
                "600",
                "--output",
                str(RUN / f"{name}.json"),
            ]
            if order == 2:
                arguments.append("--interactions")
            jobs.append((name, gpu, "20-35", arguments))
    accuracy = ROOT / "experiments/2026-11-10-high-dimensional-accuracy"
    jobs.append(
        (
            "depth-study",
            cpu,
            "20-35",
            [
                str(accuracy / "collect_accuracy.py"),
                "--datasets",
                "adult",
                "--seeds",
                "432",
                "433",
                "434",
                "--out-dir",
                str(RUN / "depth-study"),
            ],
        )
    )
    third = ROOT / "experiments/2026-11-11-order3-variability"
    jobs.append(
        (
            "order3",
            single,
            "20",
            [
                str(third / "run_repeated.py"),
                "--models",
                *MODELS,
                "--out-dir",
                str(RUN / "order3"),
                "--niter",
                "10",
                "--warmup",
                "3",
                "--limit",
                "600",
                "--setup-limit",
                "600",
            ],
        )
    )
    treegrad = ROOT / "experiments/2026-11-10-treegrad-matched"
    jobs.append(
        (
            "treegrad",
            single,
            "20",
            [
                str(treegrad / "benchmark_matched.py"),
                "--models",
                *MODELS,
                "--out-dir",
                str(RUN / "treegrad"),
                "--treegrad-root",
                str(treegrad / "dependencies/TreeGrad"),
                "--rows",
                "10",
                "--niter",
                "3",
                "--warmup",
                "1",
            ],
        )
    )
    identity = {
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "sources": {
            str(p.relative_to(ROOT)): digest(p)
            for p in [BENCH, Path(__file__), HERE / "validate_encoding.py"]
        },
        "jobs": [
            {
                "name": name,
                "command": ["taskset", "-c", cores, PYTHON, "-u", *args],
                "environment": {
                    k: env[k]
                    for k in [
                        "PYTHONPATH",
                        "CUDA_VISIBLE_DEVICES",
                        "OMP_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS",
                        "NUMBA_NUM_THREADS",
                        "NUMBA_CACHE_DIR",
                    ]
                },
            }
            for name, env, cores, args in jobs
        ],
    }
    manifest = RUN / "manifest.json"
    if manifest.exists():
        assert json.loads(manifest.read_text(encoding="utf-8")) == identity, (
            "Source or run configuration changed"
        )
    else:
        save(manifest, identity)
        (RUN / "sources").mkdir()
        for p in [BENCH, Path(__file__), HERE / "validate_encoding.py"]:
            shutil.copy2(p, RUN / "sources" / p.name)
        (RUN / "source.diff").write_text(
            subprocess.check_output(["git", "diff", "HEAD"], cwd=ROOT, text=True),
            encoding="utf-8",
        )
    for name, env, cores, args in jobs:
        marker = RUN / f"{name}.complete.json"
        if marker.exists():
            continue
        command = ["taskset", "-c", cores, PYTHON, "-u", *args]
        started = time.time()
        save(
            RUN / "status.json",
            {"phase": name, "started": started, "pid": os.getpid(), "command": command},
        )
        print(f"START {name}", flush=True)
        with (RUN / f"{name}.log").open("a", encoding="utf-8") as log:
            result = subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        outcome = {
            "phase": name,
            "returncode": result.returncode,
            "elapsed_seconds": time.time() - started,
        }
        if result.returncode:
            save(RUN / "status.json", outcome | {"status": "failed"})
            sys.exit(result.returncode)
        save(marker, outcome)
        print(f"DONE {name}: {outcome['elapsed_seconds']:.1f}s", flush=True)
    save(
        RUN / "status.json",
        {
            "status": "collected",
            "next": "Validate raw outputs, summarize and update paper; collection alone is not numerical validation.",
        },
    )


if __name__ == "__main__":
    main()
