"""RAPIDS-style SHAP benchmark for TreeSHAP and QuadratureSHAP.

This benchmark keeps the basic structure of the RAPIDS GPUTreeShap benchmark while
benchmarking four explanation paths from the current XGBoost worktree:

- CPU TreeSHAP
- CPU QuadratureTreeSHAP
- GPU TreeSHAP
- GPU QuadratureTreeSHAP

It supports additive SHAP values and SHAP interactions and emits both a model-metadata table
and a timing table.
"""

# pylint: disable=missing-class-docstring,missing-function-docstring,too-many-instance-attributes,too-many-arguments,too-many-positional-arguments,too-many-locals,broad-exception-caught,no-member,too-many-statements,duplicate-code

from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import os
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import datasets


@dataclass(frozen=True)
class TestDataset:
    name: str
    objective: str
    X: object
    y: np.ndarray

    def set_params(self, params: dict[str, object]) -> dict[str, object]:
        params["objective"] = self.objective
        if self.objective == "multi:softmax":
            params["num_class"] = int(np.max(self.y) + 1)
        return params

    def train_dmatrix(self) -> xgb.DMatrix:
        return xgb.QuantileDMatrix(self.X, self.y, enable_categorical=True)

    def test_input(self, num_rows: int, seed: int) -> object:
        rs = np.random.RandomState(seed)
        row_idx = rs.randint(0, self.X.shape[0], size=num_rows)
        if hasattr(self.X, "iloc"):
            return self.X.iloc[row_idx, :]
        return self.X[row_idx, :]

    def test_dmatrix(self, num_rows: int, seed: int) -> xgb.DMatrix:
        return xgb.DMatrix(self.test_input(num_rows, seed), enable_categorical=True)


@dataclass(frozen=True)
class ModelSpec:
    suffix: str
    num_rounds: int
    max_depth: int
    grow_policy: str | None = None
    max_leaves: int | None = None

    def training_params(self) -> dict[str, object]:
        params: dict[str, object] = {
            "tree_method": "hist",
            "device": "cuda",
            "eta": 0.01,
            "max_depth": self.max_depth,
        }
        if self.grow_policy is not None:
            params["grow_policy"] = self.grow_policy
        if self.max_leaves is not None:
            params["max_leaves"] = self.max_leaves
        return params


MODEL_SPECS = {
    "small": ModelSpec("small", 10, 6),
    "large": ModelSpec("large", 1000, 16),
    # "sparse" here means a LightGBM-style leaf-wise tree shape rather than sparse input storage.
    "sparse": ModelSpec("sparse", 100, 0, grow_policy="lossguide", max_leaves=512),
}


@dataclass(frozen=True)
class Model:
    name: str
    dataset: TestDataset
    spec: ModelSpec
    booster: xgb.Booster
    trees: int
    leaves: int
    average_depth: float
    mean_max_depth: float
    max_max_depth: int
    max_unique_feature_depth: int
    mean_nodes: float
    mean_leaves: float


@lru_cache(maxsize=1)
def fetch_adult() -> tuple[object, np.ndarray]:
    x, y = datasets.fetch_openml("adult", return_X_y=True)
    y_binary = np.array([y_i != "<=50K" for y_i in y])
    return x, y_binary


@lru_cache(maxsize=1)
def fetch_fashion_mnist() -> tuple[object, np.ndarray]:
    x, y = datasets.fetch_openml("Fashion-MNIST", return_X_y=True)
    return x, y.astype(np.int64)


@lru_cache(maxsize=1)
def get_test_datasets() -> tuple[TestDataset, ...]:
    cov_x, cov_y = datasets.fetch_covtype(return_X_y=True)
    cal_x, cal_y = datasets.fetch_california_housing(return_X_y=True)
    return (
        TestDataset("adult", "binary:logistic", *fetch_adult()),
        TestDataset("covtype", "multi:softmax", cov_x, cov_y.astype(np.int64)),
        TestDataset(
            "cal_housing",
            "reg:squarederror",
            cal_x.astype(np.float32),
            cal_y.astype(np.float32),
        ),
        TestDataset("fashion_mnist", "multi:softmax", *fetch_fashion_mnist()),
    )


def train_model(dataset: TestDataset, spec: ModelSpec) -> xgb.Booster:
    dtrain = dataset.train_dmatrix()
    params = spec.training_params()
    params = dataset.set_params(params)
    return xgb.train(
        params,
        dtrain,
        spec.num_rounds,
        evals=[(dtrain, "train")],
        verbose_eval=False,
    )


def train_or_load_model(
    dataset: TestDataset,
    spec: ModelSpec,
    model_name: str,
    model_dir: Path | None,
    force_retrain: bool,
) -> xgb.Booster:
    if model_dir is None:
        print(f"Training {model_name}")
        return train_model(dataset, spec)

    model_path = model_dir / f"{model_name}.ubj"
    if model_path.exists() and not force_retrain:
        print(f"Loading {model_name} from {model_path}")
        booster = xgb.Booster()
        booster.load_model(model_path)
        return booster

    print(f"Training {model_name}")
    booster = train_model(dataset, spec)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(model_path)
    return booster


def tree_stats(model: xgb.Booster) -> dict[str, float]:
    dump = model.get_dump(dump_format="json", with_stats=True)

    def walk(
        node: dict, depth: int = 0, path_features: frozenset[str] = frozenset()
    ) -> tuple[int, int, int, int]:
        children = node.get("children", [])
        if not children:
            return depth, len(path_features), 1, 1
        feature = node["split"]
        child_path_features = path_features | {feature}
        max_depth = depth
        max_unique_feature_depth = len(child_path_features)
        node_count = 1
        leaf_count = 0
        for child in children:
            child_depth, child_unique_feature_depth, child_nodes, child_leaves = walk(
                child, depth + 1, child_path_features
            )
            max_depth = max(max_depth, child_depth)
            max_unique_feature_depth = max(
                max_unique_feature_depth, child_unique_feature_depth
            )
            node_count += child_nodes
            leaf_count += child_leaves
        return max_depth, max_unique_feature_depth, node_count, leaf_count

    max_depths: list[int] = []
    unique_feature_depths: list[int] = []
    node_counts: list[int] = []
    leaf_counts: list[int] = []
    for tree_json in dump:
        tree = json.loads(tree_json)
        max_depth, unique_feature_depth, nodes, leaves = walk(tree)
        max_depths.append(max_depth)
        unique_feature_depths.append(unique_feature_depth)
        node_counts.append(nodes)
        leaf_counts.append(leaves)

    return {
        "trees": len(dump),
        "leaves": int(sum(leaf_counts)),
        "average_depth": float(statistics.mean(max_depths)),
        "mean_max_depth": float(statistics.mean(max_depths)),
        "max_max_depth": int(max(max_depths)),
        "max_unique_feature_depth": int(max(unique_feature_depths)),
        "mean_nodes": float(statistics.mean(node_counts)),
        "mean_leaves": float(statistics.mean(leaf_counts)),
    }


def get_models(
    model_filter: str, model_dir: Path | None, force_retrain: bool
) -> list[Model]:
    models: list[Model] = []
    for dataset in get_test_datasets():
        for spec in MODEL_SPECS.values():
            model_name = f"{dataset.name}-{spec.suffix}"
            if model_filter not in {"all", spec.suffix} and model_filter != model_name:
                continue
            booster = train_or_load_model(
                dataset, spec, model_name, model_dir, force_retrain
            )
            stats = tree_stats(booster)
            models.append(
                Model(
                    name=model_name,
                    dataset=dataset,
                    spec=spec,
                    booster=booster,
                    trees=int(stats["trees"]),
                    leaves=int(stats["leaves"]),
                    average_depth=float(stats["average_depth"]),
                    mean_max_depth=float(stats["mean_max_depth"]),
                    max_max_depth=int(stats["max_max_depth"]),
                    max_unique_feature_depth=int(stats["max_unique_feature_depth"]),
                    mean_nodes=float(stats["mean_nodes"]),
                    mean_leaves=float(stats["mean_leaves"]),
                )
            )
    return models


def predict_with_algorithm(
    booster: xgb.Booster,
    dtest: xgb.DMatrix,
    device: str,
    algorithm: str,
    interactions: bool,
) -> np.ndarray:
    params: dict[str, object] = {"device": device}
    if algorithm == "quadratureshap":
        params["shap_algorithm"] = "quadratureshap"
        params["quadratureshap_points"] = 8
    else:
        params["shap_algorithm"] = "treeshap"
    booster.set_param(params)
    if interactions:
        return np.asarray(booster.predict(dtest, pred_interactions=True))
    return np.asarray(booster.predict(dtest, pred_contribs=True))


def _benchmark_case_worker(
    queue: mp.Queue,
    booster_raw: bytes,
    x_test: object,
    device: str,
    algorithm: str,
    interactions: bool,
    niter: int,
    margin: np.ndarray | None,
    warmup: int = 3,
    nthread: int | None = None,
) -> None:
    try:
        booster = xgb.Booster()
        booster.load_model(bytearray(booster_raw))
        if nthread is not None:
            booster.set_param({"nthread": nthread})
        dtest = xgb.DMatrix(x_test, enable_categorical=True, nthread=nthread)
        pred = predict_with_algorithm(booster, dtest, device, algorithm, interactions)
        if interactions:
            additive = predict_with_algorithm(
                booster, dtest, device, algorithm, interactions=False
            )
            row_sums = np.sum(pred, axis=pred.ndim - 1)
            metrics = {
                "max_row_sum_err": float(np.max(np.abs(row_sums - additive))),
                "mean_row_sum_err": float(np.mean(np.abs(row_sums - additive))),
                "max_asymmetry": float(
                    np.max(np.abs(pred - np.swapaxes(pred, -1, -2)))
                ),
            }
        else:
            assert margin is not None
            summed = np.sum(pred, axis=pred.ndim - 1)
            metrics = {
                "max_additivity_err": float(np.max(np.abs(summed - margin))),
                "mean_additivity_err": float(np.mean(np.abs(summed - margin))),
            }

        # Accuracy calls above and these explicit warm-ups are never timed.
        for _ in range(warmup):
            predict_with_algorithm(booster, dtest, device, algorithm, interactions)
        samples = []
        for _ in range(niter):
            t0 = time.perf_counter()
            predict_with_algorithm(booster, dtest, device, algorithm, interactions)
            samples.append(time.perf_counter() - t0)
        queue.put(
            {
                "mean_time_s": float(np.mean(samples)),
                "std_time_s": float(np.std(samples, ddof=1)) if niter > 1 else None,
                "median_time_s": float(np.median(samples)),
                "q25_time_s": float(np.quantile(samples, 0.25)),
                "q75_time_s": float(np.quantile(samples, 0.75)),
                "min_time_s": float(min(samples)),
                "max_time_s": float(max(samples)),
                "samples_s": samples,
                "n_samples": len(samples),
                "error": None,
                **metrics,
            }
        )
    except Exception as err:  # noqa: BLE001
        queue.put(
            {
                "mean_time_s": None,
                "std_time_s": None,
                "max_additivity_err": None,
                "mean_additivity_err": None,
                "max_row_sum_err": None,
                "mean_row_sum_err": None,
                "max_asymmetry": None,
                "error": str(err).splitlines()[0],
            }
        )


def run_case_with_timeout(
    booster: xgb.Booster,
    x_test: object,
    device: str,
    algorithm: str,
    interactions: bool,
    niter: int,
    margin: np.ndarray | None,
    timeout_seconds: float | None,
    warmup: int = 3,
    nthread: int | None = None,
    prediction_timeout_seconds: float | None = 600.0,
) -> dict[str, object]:
    if prediction_timeout_seconds is not None or timeout_seconds is not None:
        # Keep the two deadlines distinct: repeated in-limit predictions may take
        # longer than the per-call limit in total.
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from prediction_timeout import run_case

        return run_case(
            booster,
            x_test,
            device,
            algorithm,
            niter,
            margin,
            warmup=warmup,
            nthread=nthread,
            limit=prediction_timeout_seconds,
            interactions=interactions,
            aggregate_limit=timeout_seconds,
        )
    queue: mp.Queue = mp.Queue()
    _benchmark_case_worker(
        queue,
        bytes(booster.save_raw()),
        x_test,
        device,
        algorithm,
        interactions,
        niter,
        margin,
        warmup,
        nthread,
    )
    result = queue.get()
    queue.close()
    return result


def check_accuracy(
    booster: xgb.Booster,
    dtest: xgb.DMatrix,
    device: str,
    algorithm: str,
    pred: np.ndarray,
    margin: np.ndarray,
    interactions: bool,
) -> dict[str, float]:
    if interactions:
        additive = predict_with_algorithm(
            booster, dtest, device, algorithm, interactions=False
        )
        row_sums = np.sum(pred, axis=pred.ndim - 1)
        return {
            "max_row_sum_err": float(np.max(np.abs(row_sums - additive))),
            "mean_row_sum_err": float(np.mean(np.abs(row_sums - additive))),
            "max_asymmetry": float(np.max(np.abs(pred - np.swapaxes(pred, -1, -2)))),
        }

    summed = np.sum(pred, axis=pred.ndim - 1)
    return {
        "max_additivity_err": float(np.max(np.abs(summed - margin))),
        "mean_additivity_err": float(np.mean(np.abs(summed - margin))),
    }


def benchmark_model(
    model: Model,
    x_test: object,
    dtest: xgb.DMatrix,
    niter: int,
    interactions: bool,
    timeout_seconds: float | None,
    warmup: int = 3,
    nthread: int | None = None,
    devices: tuple[str, ...] = ("cpu", "cuda"),
    prediction_timeout_seconds: float | None = 600.0,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    margin = model.booster.predict(dtest, output_margin=True)
    details: list[dict[str, object]] = []
    result_row = {
        "model": model.name,
        "test_rows": dtest.num_row(),
        "TreeSHAP": None,
        "QuadratureTreeSHAP": None,
        "GPUTreeShap": None,
        "QuadratureTreeSHAP (GPU)": None,
        "QuadratureTreeSHAP Speedup": None,
        "QuadratureTreeSHAP (GPU) Speedup": None,
    }

    for algorithm in ["treeshap", "quadratureshap"]:
        for device in devices:
            result = run_case_with_timeout(
                model.booster,
                x_test,
                device,
                algorithm,
                interactions,
                niter,
                margin if not interactions else None,
                timeout_seconds,
                warmup,
                nthread,
                prediction_timeout_seconds,
            )
            details.append(
                {
                    "model": model.name,
                    "algorithm": algorithm,
                    "device": device,
                    **result,
                }
            )
            if result["mean_time_s"] is not None:
                if algorithm == "treeshap" and device == "cpu":
                    result_row["TreeSHAP"] = float(result["mean_time_s"])
                elif algorithm == "quadratureshap" and device == "cpu":
                    result_row["QuadratureTreeSHAP"] = float(result["mean_time_s"])
                elif algorithm == "treeshap" and device == "cuda":
                    result_row["GPUTreeShap"] = float(result["mean_time_s"])
                elif algorithm == "quadratureshap" and device == "cuda":
                    result_row["QuadratureTreeSHAP (GPU)"] = float(
                        result["mean_time_s"]
                    )
        gc.collect()

    if (
        result_row["TreeSHAP"] is not None
        and result_row["QuadratureTreeSHAP"] is not None
    ):
        result_row["QuadratureTreeSHAP Speedup"] = (
            result_row["TreeSHAP"] / result_row["QuadratureTreeSHAP"]
        )
    if (
        result_row["GPUTreeShap"] is not None
        and result_row["QuadratureTreeSHAP (GPU)"] is not None
    ):
        result_row["QuadratureTreeSHAP (GPU) Speedup"] = (
            result_row["GPUTreeShap"] / result_row["QuadratureTreeSHAP (GPU)"]
        )
    return result_row, details


def markdown_table(df: pd.DataFrame, float_fmt: str = ".6f") -> str:
    headers = [str(c) for c in df.columns]
    rows = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        formatted = []
        for value in row:
            if value is None or (isinstance(value, float) and pd.isna(value)):
                formatted.append("NA")
            elif isinstance(value, float):
                formatted.append(format(value, float_fmt))
            else:
                formatted.append(str(value))
        rows.append("| " + " | ".join(formatted) + " |")
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAPIDS-style benchmark adapted for XGBoost TreeSHAP and QuadratureSHAP."
    )
    parser.add_argument("--output", type=Path, required=True, help="JSON summary path")
    parser.add_argument(
        "--out-models", type=Path, default=None, help="CSV path for model table"
    )
    parser.add_argument(
        "--out-results", type=Path, default=None, help="CSV path for timing table"
    )
    parser.add_argument(
        "--out-markdown", type=Path, default=None, help="Markdown table path"
    )
    parser.add_argument("--nrows", type=int, default=1000)
    parser.add_argument("--niter", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--nthread", type=int, default=None)
    parser.add_argument(
        "--devices", nargs="+", choices=["cpu", "cuda"], default=["cpu", "cuda"]
    )
    parser.add_argument(
        "--out-details",
        type=Path,
        default=None,
        help="CSV with timing spread and raw samples",
    )
    parser.add_argument("--seed", type=int, default=432)
    parser.add_argument(
        "--prediction-timeout-seconds",
        type=float,
        default=600.0,
        help="Limit for each accuracy, warm-up or timed prediction (default: 600s). "
        "One over-limit call invalidates the whole entry. Use 0 to disable.",
    )
    parser.add_argument(
        "--case-timeout-seconds",
        type=float,
        default=None,
        help=(
            "Optional aggregate case budget including all repeats; "
            "independent of the per-call limit."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        help="Model filter: all, small, large, sparse, or a specific dataset-size name",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Optional directory for persisted trained model cache.",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Retrain and overwrite persisted models even when cached models exist.",
    )
    parser.add_argument(
        "--models-only",
        action="store_true",
        help="Only train/load models and emit the model metadata table.",
    )
    parser.add_argument("--interactions", action="store_true")
    args = parser.parse_args()
    if args.prediction_timeout_seconds < 0 or not np.isfinite(
        args.prediction_timeout_seconds
    ):
        parser.error("prediction timeout must be finite and nonnegative")
    if args.case_timeout_seconds is not None and (
        args.case_timeout_seconds <= 0 or not np.isfinite(args.case_timeout_seconds)
    ):
        parser.error("aggregate case timeout must be finite and positive")
    args.prediction_timeout_seconds = args.prediction_timeout_seconds or None
    if args.niter < 1 or args.warmup < 0 or args.nrows < 1:
        parser.error("niter and nrows must be positive; warmup must be nonnegative")
    if args.nthread is not None and args.nthread < 1:
        parser.error("nthread must be positive")

    models = get_models(args.model, args.model_dir, args.force_retrain)
    model_rows = [
        {
            "model": model.name,
            "num_rounds": model.spec.num_rounds,
            "requested_max_depth": model.spec.max_depth,
            "grow_policy": model.spec.grow_policy or "depthwise",
            "max_leaves": model.spec.max_leaves,
            "num_trees": model.trees,
            "num_leaves": model.leaves,
            "average_depth": model.average_depth,
            "mean_max_depth": model.mean_max_depth,
            "max_max_depth": model.max_max_depth,
            "max_unique_feature_depth": model.max_unique_feature_depth,
            "mean_nodes": model.mean_nodes,
            "mean_leaves_per_tree": model.mean_leaves,
        }
        for model in models
    ]
    results_rows: list[dict[str, object]] = []
    details_rows: list[dict[str, object]] = []
    if not args.models_only:
        for model in models:
            if args.nthread is not None:
                model.booster.set_param({"nthread": args.nthread})
            x_test = model.dataset.test_input(args.nrows, args.seed)
            dtest = xgb.DMatrix(x_test, enable_categorical=True)
            result_row, details = benchmark_model(
                model,
                x_test,
                dtest,
                args.niter,
                args.interactions,
                args.case_timeout_seconds,
                args.warmup,
                args.nthread,
                tuple(dict.fromkeys(args.devices)),
                args.prediction_timeout_seconds,
            )
            results_rows.append(result_row)
            details_rows.extend(details)
            print(
                pd.DataFrame(results_rows).to_string(
                    index=False, float_format=lambda x: f"{x:.6f}"
                )
            )

    models_df = pd.DataFrame(model_rows)
    results_df = pd.DataFrame(results_rows)
    payload = {
        "nrows": args.nrows,
        "niter": args.niter,
        "warmup": args.warmup,
        "nthread": args.nthread,
        "devices": args.devices,
        "std_ddof": 1,
        "prediction_timeout_seconds": args.prediction_timeout_seconds,
        "case_timeout_seconds": args.case_timeout_seconds,
        "timeout_scope": "each prediction call; any over-limit call voids the entire entry",
        "quadrature_rule": "standard_gauss_legendre",
        "seed": args.seed,
        "timing_scope": (
            "wall-clock predict call including parameter setting and host output; "
            "excludes model/DMatrix construction and accuracy calls"
        ),
        "speedup_statistic": "ratio of arithmetic mean runtimes",
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "xgboost": xgb.__version__,
            "build_info": xgb.build_info(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        },
        "interactions": args.interactions,
        "models_only": args.models_only,
        "model_filter": args.model,
        "model_dir": str(args.model_dir) if args.model_dir is not None else None,
        "model_specs": {
            name: {
                "num_rounds": spec.num_rounds,
                "max_depth": spec.max_depth,
                "grow_policy": spec.grow_policy,
                "max_leaves": spec.max_leaves,
            }
            for name, spec in MODEL_SPECS.items()
        },
        "models_table": models_df.to_dict(orient="records"),
        "results_table": results_df.to_dict(orient="records"),
        "details": details_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    if args.out_models is not None:
        args.out_models.parent.mkdir(parents=True, exist_ok=True)
        models_df.to_csv(args.out_models, index=False)
    if args.out_results is not None:
        args.out_results.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(args.out_results, index=False)
    if args.out_details is not None:
        args.out_details.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(details_rows).to_csv(args.out_details, index=False)
    if args.out_markdown is not None:
        args.out_markdown.parent.mkdir(parents=True, exist_ok=True)
        text = "## Models\n\n" + markdown_table(models_df, ".3f") + "\n"
        if not args.models_only:
            text += "\n## Results\n\n" + markdown_table(results_df, ".6f") + "\n"
            spread = pd.DataFrame(
                [
                    {
                        key: row.get(key)
                        for key in (
                            "model",
                            "device",
                            "algorithm",
                            "n_samples",
                            "mean_time_s",
                            "std_time_s",
                            "median_time_s",
                            "q25_time_s",
                            "q75_time_s",
                            "error",
                        )
                    }
                    for row in details_rows
                ]
            )
            text += (
                "\n## Runtime variability\n\n"
                "Times are seconds. SD is sample standard deviation (ddof=1), "
                "not a confidence interval. Repetitions reuse the same model and rows. "
                "Speedups are ratios of means.\n\n"
                + markdown_table(spread, ".6g")
                + "\n"
            )
        args.out_markdown.write_text(text)

    print("Models:")
    print(models_df.to_string(index=False))
    print("Results:")
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()
