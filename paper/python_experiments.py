"""Shared execution of the first-order ablation and higher-order Python timings."""

# Numerical state and orchestration are kept explicit for reproduction.
# pylint: disable=too-many-locals,too-many-branches,too-many-statements

import argparse
import hashlib
import importlib.metadata
import json
import os
import statistics
import time
import traceback
from pathlib import Path

import numpy as np
import xgboost as xgb

from .models import (
    MODEL_DIR,
    MODELS,
    counts,
    input_rows,
    load_treegrad,
    parse_trees,
    xgboost_to_treegrad_model,
)
from .numba_quadrature import QuadratureSHAPNumba
from .numpy_quadrature import BatchedNumpyQuadrature, ScalarNumericQuadrature
from .timing import run_case, save


def first_order(booster, variant, treegrad_root):
    """Construct one of the paper’s five first-order implementations."""
    if variant in ["D", "E"]:
        obj = QuadratureSHAPNumba(booster, fixed_eight=variant == "E")
        return obj.explain
    trees, groups = parse_trees(booster)
    features = booster.num_features()
    if variant == "C":
        return BatchedNumpyQuadrature(
            trees, groups, features, counts(trees, features)
        ).explain
    if variant == "A":
        reference = load_treegrad(treegrad_root)
        models = [
            xgboost_to_treegrad_model([t for t, g in zip(trees, groups) if g == group])
            for group in range(max(groups) + 1)
        ]

        def predict_treegrad(x):
            return np.asarray(
                [
                    [
                        (
                            reference(m, row, (1, 1), class_index=None)
                            if m.estimators_.size
                            else np.zeros(features)
                        )
                        for m in models
                    ]
                    for row in x
                ]
            )

        return predict_treegrad
    rules = {}
    native = counts(trees, features)
    for n in set(native):
        t, w = np.polynomial.legendre.leggauss(n)
        rules[n] = ((t + 1) / 2, w / 2)

    def predict(x):
        if not np.isfinite(x).all():
            raise ValueError("Scalar B requires finite numeric benchmark rows")
        result = np.zeros((len(x), max(groups) + 1, features))
        obj = ScalarNumericQuadrature()
        for r, row in enumerate(x):
            for tree, group, n in zip(trees, groups, native):
                if tree.feature[0] < 0:
                    continue
                obj.nodes, obj.weights = rules[n]
                obj._dfs(  # pylint: disable=protected-access
                    tree,
                    row,
                    0,
                    np.ones(n),
                    1.0,
                    np.full(features, np.nan),
                    result[r, group],
                )
        return result

    return predict


class OrderedFeatureSet(set):
    """Keep TreeSHAP-IQ’s feature remapping in original feature-ID order."""

    def __iter__(self):
        return iter(sorted(set.__iter__(self)))


def higher_order(booster, method, order):
    """Construct the requested-order SII explainer."""
    from .interactions import OrderQuadrature, dict_from_interaction_values

    trees, _ = parse_trees(booster)
    trees = [t for t in trees if t.feature[0] >= 0]
    if method == "quadrature":
        return OrderQuadrature(trees, booster.num_features(), order, 8).explain
    from shapiq.explainer.tree import TreeExplainer
    from shapiq.explainer.tree.base import TreeModel

    models = [
        TreeModel(
            children_left=t.left.copy(),
            children_right=t.right.copy(),
            features=t.feature.copy(),
            thresholds=np.nextafter(t.threshold, -np.inf),
            values=t.value.copy(),
            node_sample_weight=t.cover.copy(),
            original_output_type="raw",
            feature_ids=OrderedFeatureSet(int(f) for f in t.feature if f >= 0),
        )
        for t in trees
    ]
    obj = TreeExplainer(models, max_order=order, min_order=order, index="SII")
    return lambda row: dict_from_interaction_values(obj.explain(row), order)


def worker(pipe, case, args):
    """Report setup and individual prediction durations from an isolated process."""

    def start(phase, index):
        pipe.send(
            {
                "event": "start",
                "phase": phase,
                "index": index,
                "started_monotonic": time.monotonic(),
            }
        )

    def end(phase, index, elapsed):
        pipe.send(
            {"event": "end", "phase": phase, "index": index, "elapsed_s": elapsed}
        )

    try:
        if args.cpu is not None:
            os.sched_setaffinity(0, {args.cpu})
        start("setup", 0)
        t = time.perf_counter()
        booster = xgb.Booster(model_file=args.model_dir / f"{args.model}.ubj")
        booster.set_param({"device": "cpu", "nthread": 1})
        x = np.load(case / "input.npy")
        features = booster.num_features()
        if x.shape[1] != features:
            raise ValueError(
                "Model and canonical input feature schemas differ; rebuild the models"
            )
        groups = max(
            1,
            int(
                json.loads(booster.save_config())["learner"]["learner_model_param"][
                    "num_class"
                ]
            ),
        )
        predict = (
            first_order(booster, args.method, args.treegrad_root)
            if args.experiment == "ablation"
            else higher_order(booster, args.method, args.order)
        )
        end("setup", 0, time.perf_counter() - t)
        samples = []
        for index in range(args.warmup + args.niter):
            phase = "warmup" if index < args.warmup else "timed"
            start(phase, index)
            t = time.perf_counter()
            data = x[:1] if phase == "warmup" and args.method in ["A", "B"] else x
            value = predict(data) if args.experiment == "ablation" else predict(data[0])
            elapsed = time.perf_counter() - t
            end(phase, index, elapsed)
            if args.experiment == "ablation":
                assert value.shape == (len(data), groups, features)
                if not np.isfinite(value).all():
                    raise ValueError("Nonfinite attributions")
                if phase == "timed":
                    np.save(case / f"repeat{index - args.warmup}.npy", value)
            else:
                keys = sorted(value)
                array = np.asarray([value[k] for k in keys])
                assert np.isfinite(array).all()
                if phase == "timed":
                    np.savez_compressed(
                        case / f"repeat{index - args.warmup}.npz",
                        keys=np.asarray(keys, dtype=np.int32).reshape(-1, args.order),
                        values=array,
                    )
            if phase == "timed":
                samples.append(elapsed)
        pipe.send(
            {
                "event": "result",
                "result": {
                    "status": "complete",
                    "error": None,
                    "samples_s": samples,
                    "mean_s": statistics.mean(samples),
                    "sample_sd_s": (
                        statistics.stdev(samples) if len(samples) > 1 else None
                    ),
                },
            }
        )
    except BaseException:  # pylint: disable=broad-exception-caught
        pipe.send(
            {
                "event": "result",
                "result": {
                    "status": "error",
                    "error": traceback.format_exc(),
                    "samples_s": [],
                    "mean_s": None,
                    "sample_sd_s": None,
                },
            }
        )
    finally:
        pipe.close()


def main(experiment):
    """Run or resume the selected paper experiment with frozen inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    default = (
        MODELS
        if experiment == "ablation"
        else [m for m in MODELS if not m.startswith("fashion_mnist")]
    )
    parser.add_argument("--models", nargs="+", default=default)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=(
            list("ABCDE") if experiment == "ablation" else ["quadrature", "treeshapiq"]
        ),
        default=(
            list("ABCDE") if experiment == "ablation" else ["quadrature", "treeshapiq"]
        ),
    )
    parser.add_argument(
        "--niter", type=int, default=3 if experiment == "ablation" else 10
    )
    parser.add_argument(
        "--warmup", type=int, default=1 if experiment == "ablation" else 3
    )
    parser.add_argument("--limit", type=float, default=600)
    parser.add_argument("--setup-limit", type=float, default=600)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--seed", type=int, default=432)
    if experiment == "higher_order":
        parser.add_argument("--order", type=int, choices=[2, 3, 4, 5, 6], default=3)
        parser.set_defaults(treegrad_root=None)
    else:
        parser.add_argument(
            "--treegrad-root",
            type=Path,
            default=Path(__file__).parent / "dependencies/TreeGrad",
        )
        parser.set_defaults(order=1)
    args = parser.parse_args()
    args.experiment = experiment
    args.rows = 128 if experiment == "ablation" else 1
    if args.niter < 1 or args.warmup < 0 or min(args.limit, args.setup_limit) <= 0:
        parser.error("Use positive repetition/limit values and nonnegative warmup")
    if experiment == "ablation" and any(m in args.methods for m in ["D", "E"]):
        from numba import config

        if not config.SLP_VECTORIZE:
            parser.error("Set NUMBA_SLP_VECTORIZE=1 before Python starts")
    if experiment == "ablation" and "A" in args.methods:
        import subprocess

        from .fetch_treegrad import REVISION

        if not args.treegrad_root.exists():
            parser.error(
                "Run python -m paper.fetch_treegrad before selecting variant A"
            )
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=args.treegrad_root, text=True
        ).strip()
        if revision != REVISION:
            parser.error(
                "TreeGrad checkout does not match the pinned reference revision"
            )
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    def sha(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()

    manifest = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    manifest.update(
        timeout_scope=f"{args.limit:g} seconds per complete explanation call, no aggregate cutoff",
        model_sha256={m: sha(args.model_dir / f"{m}.ubj") for m in args.models},
        source_sha256={p.name: sha(p) for p in Path(__file__).parent.glob("*.py")},
        versions={
            name: importlib.metadata.version(name)
            for name in ["numpy", "numba", "xgboost", "shapiq"]
        },
    )
    if (out / "manifest.json").exists():
        assert json.loads((out / "manifest.json").read_text()) == manifest, (
            "Configuration changed; use a new output directory"
        )
    else:
        save(out / "manifest.json", manifest)
    for model in args.models:
        x = input_rows(model, args.rows, args.seed)
        for method in args.methods:
            case = out / model / method
            case.mkdir(parents=True, exist_ok=True)
            if (case / "results.json").exists():
                continue
            np.save(case / "input.npy", x)
            args.model = model
            args.method = method

            def progress(event, _calls, model=model, method=method):
                save(
                    out / "status.json",
                    {"model": model, "method": method, "event": event},
                )

            print("START", model, method, flush=True)
            result = run_case(case, args, progress, target=worker)
            print("DONE", model, method, result["status"], flush=True)
    save(out / "complete.json", {"status": "collection_finished"})
