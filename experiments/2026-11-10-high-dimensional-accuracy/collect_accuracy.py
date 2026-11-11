"""Collect depth/scale/attribution-error data before choosing a combined plot."""
# pylint: disable=duplicate-code
# pylint: disable=line-too-long,missing-function-docstring,protected-access,too-many-branches,too-many-locals,too-many-statements

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import xgboost as xgb
from numba import set_num_threads
from reference import flatten, predict_reference
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
BENCH = (
    ROOT
    / "experiments/2026-04-10-rapids-style-shap-benchmark/quadratureshap_rapids_benchmark.py"
)


def atomic_json(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def numeric(x):
    if hasattr(x, "iloc"):
        x = x.copy()
        for col in x.columns:
            if str(x[col].dtype) == "category":
                codes = x[col].cat.codes.astype(float)
                x[col] = codes.where(codes >= 0, np.nan)
        return x.to_numpy(dtype=np.float32)
    return np.asarray(x, dtype=np.float32)


def select(x, idx):
    return x.iloc[idx] if hasattr(x, "iloc") else x[idx]


def distribution(v):
    v = np.asarray(v)
    finite = v[np.isfinite(v)]
    return {
        "mean": float(np.mean(finite)) if finite.size else None,
        "p99": float(np.quantile(finite, 0.99)) if finite.size else None,
        "max": float(np.max(finite)) if finite.size else None,
        "finite_count": int(finite.size),
        "nonfinite_count": int(v.size - finite.size),
    }


def ratio(a, b):
    return float(a / b) if b > 0 and np.isfinite(a) and np.isfinite(b) else None


def measure(pred, margin, reference):
    # Reduce in float64, so these metrics do not add float32 summation error.
    pred = np.asarray(pred, dtype=np.float64)
    err = pred[: len(reference), :, :-1] - reference[:, :, :-1]
    truth = reference[:, :, :-1]
    rms = float(np.sqrt(np.mean(truth**2)))
    ma = float(np.mean(np.abs(truth)))
    maxtruth = float(np.max(np.abs(truth)))
    residual = np.abs(pred.sum(axis=-1) - margin)
    centered = margin - reference[0, :, -1]
    scale = float(np.sqrt(np.mean(centered**2)))
    rownorm = np.sqrt(np.sum(truth**2, axis=-1))
    valid = rownorm > max(float(np.max(rownorm)) * 1e-12, 1e-300)
    relative = np.sqrt(np.sum(err**2, axis=-1))[valid] / rownorm[valid]
    return {
        "attribution_abs_error": distribution(np.abs(err)),
        "attribution_nrmse": ratio(np.sqrt(np.mean(err**2)), rms),
        "attribution_normalized_mae": ratio(np.mean(np.abs(err)), ma),
        "attribution_max_error_over_max_reference": ratio(
            np.max(np.abs(err)), maxtruth
        ),
        "row_relative_l2": distribution(relative) if len(relative) else None,
        "row_relative_l2_valid_fraction": float(np.mean(valid)),
        "reference_attribution_rms": rms,
        "reference_attribution_mean_abs": ma,
        "reference_attribution_max_abs": maxtruth,
        "efficiency_abs_error": distribution(residual),
        "prediction_centered_rms": scale,
        "efficiency_max_over_prediction_rms": ratio(np.max(residual), scale),
        "bias_abs_error": distribution(
            np.abs(pred[: len(reference), :, -1] - reference[:, :, -1])
        ),
        "finite": bool(np.all(np.isfinite(pred))),
        "nonfinite_prediction_count": int(np.sum(~np.isfinite(pred))),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["adult", "cal_housing", "covtype", "fashion_mnist"],
    )
    p.add_argument(
        "--depths", type=int, nargs="+", default=[4, 8, 12, 16, 24, 32, 40, 50]
    )
    p.add_argument("--seeds", type=int, nargs="+", default=[432])
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--max-leaves", type=int, default=512)
    p.add_argument("--eval-rows", type=int, default=256)
    p.add_argument("--reference-rows", type=int, default=16)
    p.add_argument("--nthread", type=int, default=16)
    args = p.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        p.error(
            "Set CUDA_VISIBLE_DEVICES=-1: GPU predictor configuration rejects CPU point counts other than 8"
        )
    if not 1 <= args.reference_rows <= args.eval_rows:
        p.error("reference rows must be in [1, eval rows]")
    if min(args.depths) < 1 or min(args.rounds, args.max_leaves, args.nthread) < 1:
        p.error("positive training settings required")
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    set_num_threads(args.nthread)
    identity = {
        k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
    }
    identity.update(
        {
            "revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "library": str(Path(xgb.core._LIB._name).resolve()),
            "library_sha256": digest(Path(xgb.core._LIB._name)),
            "dataset_loader_sha256": digest(BENCH),
            "collector_sha256": digest(Path(__file__)),
            "reference_sha256": digest(Path(__file__).with_name("reference.py")),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "environment": {
                k: os.environ.get(k)
                for k in [
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "CUDA_VISIBLE_DEVICES",
                    "PYTHONPATH",
                ]
            },
            "xgboost": xgb.__version__,
            "numpy": np.__version__,
        }
    )
    manifest = out / "manifest.json"
    if manifest.exists():
        assert json.loads(manifest.read_text(encoding="utf-8")) == identity, (
            "configuration/source changed; use a new directory"
        )
    else:
        atomic_json(manifest, identity)
        sources = out / "sources"
        sources.mkdir(exist_ok=True)
        (sources / "dataset_loader.py").write_bytes(BENCH.read_bytes())
        for source in [
            "collect_accuracy.py",
            "reference.py",
            "validate_reference.py",
            "validate_collection.py",
            "prepare_runtime.py",
        ]:
            (sources / source).write_bytes(
                Path(__file__).with_name(source).read_bytes()
            )
    spec = importlib.util.spec_from_file_location("accuracy_datasets", BENCH)
    bench = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = bench
    spec.loader.exec_module(bench)
    datasets = {d.name: d for d in bench.get_test_datasets()}
    unknown = set(args.datasets) - datasets.keys()
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")
    for seed in args.seeds:
        for name in args.datasets:
            dataset = datasets[name]
            indices = np.arange(len(dataset.y))
            train_idx, test_idx = train_test_split(
                indices,
                test_size=0.2,
                random_state=seed,
                stratify=dataset.y if dataset.objective != "reg:squarederror" else None,
            )
            rng = np.random.default_rng(seed)
            eval_idx = rng.choice(
                test_idx, size=min(args.eval_rows, len(test_idx)), replace=False
            )
            train = select(dataset.X, train_idx)
            test = select(dataset.X, eval_idx)
            dtrain = xgb.QuantileDMatrix(
                train,
                dataset.y[train_idx],
                enable_categorical=True,
                nthread=args.nthread,
            )
            dtest = xgb.DMatrix(test, enable_categorical=True, nthread=args.nthread)
            raw = numeric(test)
            for depth in args.depths:
                case = out / f"{name}-seed{seed}-depth{depth}"
                if (case / "metrics.json").exists():
                    continue
                case.mkdir(exist_ok=True)
                params = dataset.set_params(
                    {
                        "tree_method": "hist",
                        "device": "cpu",
                        "grow_policy": "lossguide",
                        "max_depth": depth,
                        "max_leaves": args.max_leaves,
                        "eta": 0.05,
                        "seed": seed,
                        "nthread": args.nthread,
                        "base_score": 0.5,
                    }
                )
                print("START", name, seed, depth, flush=True)
                t0 = time.monotonic()
                if (case / "model.ubj").exists():
                    model = xgb.Booster()
                    model.load_model(case / "model.ubj")
                    model.set_param({"nthread": args.nthread, "device": "cpu"})
                else:
                    model = xgb.train(params, dtrain, args.rounds)
                    model.save_model(case / "model.ubj")
                arrays, base, stats = flatten(model)
                n = max(16, (stats["max_unique_depth"] + 1) // 2)
                ref = predict_reference(raw[: args.reference_rows], arrays, base, n)
                check = predict_reference(
                    raw[: args.reference_rows], arrays, base, 2 * n
                )
                gap = float(np.max(np.abs(ref - check)))
                scale = max(float(np.max(np.abs(check))), 1.0)
                if gap > 1e-10 * scale:
                    raise ValueError(f"Reference did not converge: {gap}")
                margin = np.asarray(
                    model.predict(dtest, output_margin=True, strict_shape=True),
                    dtype=np.float64,
                )
                ref_eff = float(np.max(np.abs(ref.sum(axis=-1) - margin[: len(ref)])))
                if not np.all(np.isfinite(ref)) or not np.all(np.isfinite(check)):
                    raise ValueError("Nonfinite reference")
                data = {
                    "x_eval": raw,
                    "eval_indices": eval_idx,
                    "train_indices": train_idx,
                    "x_reference": raw[: args.reference_rows],
                    "margin": margin,
                    "reference": ref,
                    "reference_double_points": check,
                }
                metrics = {
                    "dataset": name,
                    "seed": seed,
                    "requested_depth": depth,
                    "training_params": params,
                    "rounds": args.rounds,
                    "eval_rows": len(raw),
                    "reference_rows": len(ref),
                    **stats,
                    "standard8_theoretical_exact": stats["max_unique_depth"] <= 16,
                    "production_standard8_theoretical_exact": stats["max_unique_depth"]
                    <= 16,
                    "reference_points": n,
                    "reference_double_points_max_gap": gap,
                    "reference_finite": bool(
                        np.all(np.isfinite(ref)) and np.all(np.isfinite(check))
                    ),
                    "reference_efficiency_max_error": ref_eff,
                    "model_sha256": digest(case / "model.ubj"),
                    "methods": {},
                }
                for label, nodes, endpoint in [("standard8_float64", 8, False)]:
                    result = predict_reference(
                        raw[: args.reference_rows], arrays, base, nodes, endpoint
                    )
                    data[label] = result
                    metrics["methods"][label] = measure(result, margin[: len(ref)], ref)
                for points in [4, 6, 8, 16]:
                    model.set_param(
                        {
                            "device": "cpu",
                            "shap_algorithm": "quadratureshap",
                            "quadratureshap_points": points,
                        }
                    )
                    result = model.predict(dtest, pred_contribs=True, strict_shape=True)
                    label = f"cpu_standard{points}_float32"
                    data[label] = result
                    metrics["methods"][label] = measure(result, margin, ref)
                # TreeSHAP is a comparator, never the assumed reference on deep trees.
                model.set_param({"device": "cpu", "shap_algorithm": "treeshap"})
                result = model.predict(dtest, pred_contribs=True, strict_shape=True)
                data["cpu_treeshap_float32"] = result
                metrics["methods"]["cpu_treeshap_float32"] = measure(
                    result, margin, ref
                )
                implementation_error = (
                    data["cpu_standard8_float32"][: len(ref)].astype(np.float64)
                    - data["standard8_float64"]
                )
                metrics["standard8_implementation_abs_error"] = distribution(
                    np.abs(implementation_error)
                )
                metrics["standard8_implementation_nrmse"] = ratio(
                    np.sqrt(np.mean(implementation_error[:, :, :-1] ** 2)),
                    np.sqrt(np.mean(ref[:, :, :-1] ** 2)),
                )
                metrics["elapsed_seconds"] = time.monotonic() - t0
                np.savez_compressed(case / "raw.npz", **data)
                atomic_json(case / "metrics.json", metrics)
                combined = [
                    json.loads(path.read_text(encoding="utf-8"))
                    for path in sorted(out.glob("*/metrics.json"))
                ]
                atomic_json(
                    out / "results.json", {"manifest": str(manifest), "cases": combined}
                )
                print(
                    "DONE",
                    name,
                    seed,
                    depth,
                    "realized d",
                    stats["max_unique_depth"],
                    "reference gap",
                    gap,
                    "standard8 nrmse",
                    metrics["methods"]["standard8_float64"]["attribution_nrmse"],
                    "production8 nrmse",
                    metrics["methods"]["cpu_standard8_float32"]["attribution_nrmse"],
                    flush=True,
                )
    atomic_json(
        out / "complete.json",
        {"completed_cases": len(list(out.glob("*/metrics.json")))},
    )
    print("COLLECTION COMPLETE", flush=True)


if __name__ == "__main__":
    main()
