"""Matched float64/count comparison for Appendix A Table 5; checkpoint every trial."""
# pylint: disable=duplicate-code
# pylint: disable=consider-using-enumerate,line-too-long,missing-function-docstring,protected-access,too-many-branches,too-many-locals,too-many-statements

import argparse
import hashlib
import importlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numba
import numpy as np
import xgboost as xgb
from quadrature import Quadrature

ROOT = Path(__file__).resolve().parents[2]
OLD = (
    ROOT / "experiments/2026-04-30-treegrad-appendix/benchmark_treegrad_first_order.py"
)
CACHE = ROOT / "experiments/2026-04-10-rapids-style-shap-benchmark/model-cache"
MODELS = [
    f"{dataset}-{size}"
    for size in ["small", "sparse", "large"]
    for dataset in ["adult", "cal_housing", "covtype", "fashion_mnist"]
]
METHODS = [
    "q_fixed8",
    "q_native_count",
    "q_distinct_count",
    "tg_native",
    "tg_distinct",
    "tg_native_cached",
]


def digest(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def load_old():
    spec = importlib.util.spec_from_file_location("appendix_base", OLD)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def depths(tree):
    def walk(node, seen, d):
        if tree.feature[node] < 0:
            return d, len(seen)
        seen = seen | {int(tree.feature[node])}
        a = walk(tree.left[node], seen, d + 1)
        b = walk(tree.right[node], seen, d + 1)
        return max(a[0], b[0]), max(a[1], b[1])

    return walk(0, set(), 0)


class NodesProxy:
    """Override only TreeGrad's node provider; cached arrays are copied before mutation."""

    def __init__(self, cache=None, audit=False):
        # Match direct NumPy attribute lookup cost; override only polynomial.leggauss.
        self.__dict__.update(vars(np))
        self.cache = cache
        self.calls = []
        self.audit = audit
        self.polynomial = SimpleNamespace(
            legendre=SimpleNamespace(leggauss=self.leggauss)
        )

    def __getattr__(self, name):
        return getattr(np, name)

    def leggauss(self, n):
        if self.audit:
            self.calls.append(int(n))
        if self.cache is None:
            return np.polynomial.legendre.leggauss(n)
        a, b = self.cache[n]
        return a.copy(), b.copy()


def grouped(trees, info, groups):
    return [[t for t, g in zip(trees, info) if g == group] for group in range(groups)]


def make_objects(old, trees, info, nfeatures, groups):
    ds = [depths(tree) for tree in trees]
    native = [max(1, math.ceil(min(d, nfeatures) / 2)) for d, u in ds]
    distinct = [max(1, math.ceil(u / 2)) for d, u in ds]
    # Constant trees have exactly zero feature attributions; omit them for both methods.
    gtrees = [
        [t for t, g in zip(trees, info) if g == group and t.feature[0] >= 0]
        for group in range(groups)
    ]
    q = {
        name: [
            Quadrature(
                ts,
                nfeatures,
                [
                    counts[i]
                    for i, g in enumerate(info)
                    if g == group and trees[i].feature[0] >= 0
                ],
            )
            for group, ts in enumerate(gtrees)
        ]
        for name, counts in [
            ("q_fixed8", [8] * len(trees)),
            ("q_native_count", native),
            ("q_distinct_count", distinct),
        ]
    }
    native_models = []
    distinct_models = []
    expected_native = []
    expected_distinct = []
    for group, ts in enumerate(gtrees):
        a = old.xgboost_to_treegrad_model(ts)
        b = old.xgboost_to_treegrad_model(ts)
        active = [
            i for i, g in enumerate(info) if g == group and trees[i].feature[0] >= 0
        ]
        for stage, index in zip(b.estimators_, active):
            # Upstream uses max_depth only for point selection, never traversal/storage.
            stage[0].tree_.max_depth = ds[index][1]
        native_models.append(a)
        distinct_models.append(b)
        expected_native.extend(native[i] for i in active)
        expected_distinct.extend(distinct[i] for i in active)
    return (
        q,
        native_models,
        distinct_models,
        native,
        distinct,
        ds,
        expected_native,
        expected_distinct,
    )


def worker(args):
    case = args.out_dir / args.worker
    case.mkdir(parents=True, exist_ok=True)
    old = load_old()
    tg = old.load_treegrad(args.treegrad_root)
    tgmodule = importlib.import_module("TreeGrad.treegrad_shap")
    booster = xgb.Booster()
    booster.load_model(CACHE / f"{args.worker}.ubj")
    trees = old.parse_trees(booster)
    # Restore exact float32 model parameters before promoting both inputs to float64.
    for tree in trees:
        for key in ["threshold", "cover", "value"]:
            setattr(tree, key, getattr(tree, key).astype(np.float32).astype(np.float64))
        for parent in range(len(tree.feature)):
            if tree.feature[parent] >= 0:
                for child in [tree.left[parent], tree.right[parent]]:
                    tree.child_weight[child] = tree.cover[child] / tree.cover[parent]
    if old.has_categorical_splits(trees):
        save(
            case / "complete.json",
            {
                "model": args.worker,
                "status": "unsupported",
                "reason": "Categorical XGBoost splits unsupported by the existing TreeGrad adapter",
            },
        )
        return
    dataset_name = args.worker.rsplit("-", 1)[0]
    dataset = next(
        d
        for d in old.load_benchmark_module().get_test_datasets()
        if d.name == dataset_name
    )
    x = (
        old.to_numeric_matrix(dataset.test_input(args.rows, args.seed))
        .astype(np.float32)
        .astype(np.float64)
    )
    if not np.isfinite(x).all():
        raise ValueError("Upstream TreeGrad does not implement missing-value routing")
    learner = json.loads(booster.save_raw(raw_format="json"))["learner"]
    model = learner["gradient_booster"]["model"]
    info = model["tree_info"]
    groups = max(1, int(learner["learner_model_param"]["num_class"]))
    t0 = time.perf_counter()
    (
        q,
        native_models,
        distinct_models,
        native,
        distinct,
        ds,
        expected_native,
        expected_distinct,
    ) = make_objects(old, trees, info, x.shape[1], groups)
    construction = time.perf_counter() - t0
    cache = {n: np.polynomial.legendre.leggauss(n) for n in set(native)}
    metadata = {
        "model": args.worker,
        "rows": len(x),
        "niter": args.niter,
        "warmup_rows": args.warmup,
        "count_audit_rows": 1,
        "precision": "float64",
        "rule": "standard Gauss-Legendre on [0,1]",
        "num_features": x.shape[1],
        "num_trees": len(trees),
        "groups": groups,
        "constant_trees_skipped_both": int(sum(t.feature[0] < 0 for t in trees)),
        "max_raw_depth": max(d for d, u in ds),
        "max_distinct_depth": max(u for d, u in ds),
        "native_count_histogram": dict(Counter(native)),
        "distinct_count_histogram": dict(Counter(distinct)),
        "native_counts": native,
        "distinct_counts": distinct,
        "model_sha256": digest(CACHE / f"{args.worker}.ubj"),
        "construction_s": construction,
        "methods": {},
        "trial_order": [],
    }
    np.save(case / "input.npy", x)

    def select_method(method):
        if method.startswith("q_"):
            return lambda row: np.stack([obj.explain(row) for obj in q[method]])
        objs = distinct_models if method == "tg_distinct" else native_models
        tgmodule.np = NodesProxy(cache) if method == "tg_native_cached" else np
        return lambda row: np.stack(
            [
                tg(obj, row, (1, 1), class_index=None)
                if obj.estimators_.size
                else np.zeros(len(row), dtype=np.float64)
                for obj in objs
            ]
        )

    # Verify actual upstream per-tree counts, without instrumentation in timed calls.
    for name, objs, expected in [
        ("native", native_models, expected_native),
        ("distinct", distinct_models, expected_distinct),
    ]:
        proxy = NodesProxy(audit=True)
        tgmodule.np = proxy
        for obj in objs:
            if obj.estimators_.size:
                tg(obj, x[0], (1, 1), class_index=None)
        assert proxy.calls == expected, (name, proxy.calls, expected)
    tgmodule.np = np
    for method in METHODS:
        save(case / "status.json", {"phase": "warmup", "method": method})
        explain = select_method(method)
        start = time.perf_counter()
        for row in x[: args.warmup]:
            value = explain(row)
            assert value.dtype == np.float64
        metadata["methods"][method] = {
            "samples_s": [],
            "row_samples_s": [],
            "warmup_s": time.perf_counter() - start,
        }
    save(case / "results.json", metadata)
    rng = np.random.default_rng(args.seed)
    try:
        for repeat in range(args.niter):
            order = [METHODS[int(i)] for i in rng.permutation(len(METHODS))]
            metadata["trial_order"].append(order)
            for method in order:
                save(
                    case / "status.json",
                    {"phase": "timing", "method": method, "repeat": repeat},
                )
                explain = select_method(method)
                outputs = []
                times = []
                for row in x:
                    start = time.perf_counter()
                    value = explain(row)
                    times.append(time.perf_counter() - start)
                    outputs.append(value)
                values = np.stack(outputs)
                if not np.isfinite(values).all():
                    raise ValueError(f"Nonfinite attribution: {method}")
                if values.dtype != np.float64:
                    raise ValueError(f"Unexpected output dtype: {method}")
                np.save(case / f"{method}-repeat{repeat}.npy", values)
                r = metadata["methods"][method]
                r["samples_s"].append(float(sum(times)))
                r["row_samples_s"].append(times)
                r["mean_s"] = float(np.mean(r["samples_s"]))
                r["sample_sd_s"] = (
                    float(np.std(r["samples_s"], ddof=1))
                    if len(r["samples_s"]) > 1
                    else None
                )
                save(case / "results.json", metadata)
                print(
                    "TRIAL", args.worker, repeat, method, r["samples_s"][-1], flush=True
                )
    finally:
        tgmodule.np = np
    baseline = np.load(case / "tg_distinct-repeat0.npy")
    scale = float(np.sqrt(np.mean(baseline**2)))
    for method in METHODS:
        values = np.load(case / f"{method}-repeat0.npy")
        diff = values - baseline
        metadata["methods"][method]["max_abs_diff_vs_tg_distinct"] = float(
            np.max(np.abs(diff))
        )
        metadata["methods"][method]["nrmse_vs_tg_distinct"] = (
            float(np.sqrt(np.mean(diff**2)) / scale) if scale else None
        )
    metadata["speedups"] = {
        "matched_native": metadata["methods"]["tg_native"]["mean_s"]
        / metadata["methods"]["q_native_count"]["mean_s"],
        "matched_distinct": metadata["methods"]["tg_distinct"]["mean_s"]
        / metadata["methods"]["q_distinct_count"]["mean_s"],
        "cached_native": metadata["methods"]["tg_native_cached"]["mean_s"]
        / metadata["methods"]["q_native_count"]["mean_s"],
        "fixed8_vs_native": metadata["methods"]["tg_native"]["mean_s"]
        / metadata["methods"]["q_fixed8"]["mean_s"],
    }
    save(case / "results.json", metadata)
    save(case / "complete.json", {"model": args.worker, "status": "complete"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--treegrad-root",
        type=Path,
        default=Path(__file__).resolve().parent / "dependencies/TreeGrad",
    )
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--rows", type=int, default=10)
    parser.add_argument("--niter", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=432)
    parser.add_argument("--timeout", type=float, default=21600)
    parser.add_argument("--worker")
    args = parser.parse_args()
    args.out_dir = args.out_dir.resolve()
    args.treegrad_root = args.treegrad_root.resolve()
    if min(args.rows, args.niter, args.warmup) < 1:
        parser.error("rows, niter and warmup must be positive")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(args)
        return
    identity = {
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "treegrad_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=args.treegrad_root, text=True
        ).strip(),
        "treegrad_source_sha256": digest(
            args.treegrad_root / "TreeGrad/treegrad_shap.py"
        ),
        "treegrad_dispatch_sha256": digest(args.treegrad_root / "TreeGrad/treestab.py"),
        "python": sys.version,
        "numpy": np.__version__,
        "numba": numba.__version__,
        "xgboost": xgb.__version__,
        "library": str(Path(xgb.core._LIB._name).resolve()),
        "library_sha256": digest(xgb.core._LIB._name),
        "script_sha256": digest(__file__),
        "quadrature_sha256": digest(Path(__file__).with_name("quadrature.py")),
        "appendix_adapter_sha256": digest(OLD),
        "model_sha256": {m: digest(CACHE / f"{m}.ubj") for m in args.models},
        "affinity": sorted(os.sched_getaffinity(0)),
        "environment": {
            k: os.environ.get(k)
            for k in [
                "PYTHONPATH",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMBA_NUM_THREADS",
                "CUDA_VISIBLE_DEVICES",
            ]
        },
        "rows": args.rows,
        "niter": args.niter,
        "warmup": args.warmup,
        "seed": args.seed,
        "models": args.models,
        "timeout_per_model_seconds": args.timeout,
        "methods": METHODS,
        "timing_scope": "sum of row explanation call times, all output groups; conversion/construction/warmup/JIT excluded",
    }
    manifest = args.out_dir / "manifest.json"
    if manifest.exists():
        assert json.loads(manifest.read_text(encoding="utf-8")) == identity, (
            "Run identity changed; choose a new output directory"
        )
    else:
        save(manifest, identity)
        sources = args.out_dir / "sources"
        sources.mkdir()
        for path in [
            Path(__file__),
            Path(__file__).with_name("quadrature.py"),
            Path(__file__).with_name("validate_results.py"),
            Path(__file__).with_name("validate_algorithms.py"),
            Path(__file__).with_name("summarize.py"),
            OLD,
            args.treegrad_root / "TreeGrad/treegrad_shap.py",
            args.treegrad_root / "TreeGrad/treestab.py",
        ]:
            (sources / path.name).write_bytes(path.read_bytes())
    for model in args.models:
        folder = args.out_dir / model
        if (folder / "complete.json").exists():
            continue
        folder.mkdir(exist_ok=True)
        if (folder / "failure.json").exists():
            (folder / "failure.json").unlink()
        # Incomplete model cases are rerun; trial files stay available for inspection.
        command = [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "--worker",
            model,
            "--out-dir",
            str(args.out_dir),
            "--treegrad-root",
            str(args.treegrad_root),
            "--rows",
            str(args.rows),
            "--niter",
            str(args.niter),
            "--warmup",
            str(args.warmup),
            "--seed",
            str(args.seed),
        ]
        save(folder / "command.json", command)
        print("START", model, flush=True)
        try:
            with (folder / "worker.log").open("w", encoding="utf-8") as log:
                subprocess.run(
                    command,
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=args.timeout,
                )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            save(folder / "failure.json", {"model": model, "error": str(e)})
            print("FAILED", model, str(e), flush=True)
            continue
        print(
            "DONE",
            model,
            json.loads((folder / "complete.json").read_text(encoding="utf-8"))[
                "status"
            ],
            flush=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("summarize.py")),
                str(args.out_dir),
            ],
            check=True,
        )
    save(
        args.out_dir / "finished.json",
        {
            "attempted_models": args.models,
            "completed": sum(
                (args.out_dir / m / "complete.json").exists() for m in args.models
            ),
            "failed": sum(
                (args.out_dir / m / "failure.json").exists() for m in args.models
            ),
        },
    )
    print("SUITE FINISHED", flush=True)
    if any((args.out_dir / m / "failure.json").exists() for m in args.models):
        sys.exit(1)


if __name__ == "__main__":
    main()
