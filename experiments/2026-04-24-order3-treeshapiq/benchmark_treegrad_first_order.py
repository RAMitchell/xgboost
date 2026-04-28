"""Compare TreeGrad-Shap against the Python quadrature first-order implementation."""
# pylint: disable=missing-function-docstring,missing-class-docstring,too-many-arguments,too-many-locals,broad-exception-caught,cell-var-from-loop,too-many-instance-attributes,too-many-positional-arguments,duplicate-code

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import multiprocessing as mp
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import xgboost as xgb
from numba import njit

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "demo/guide-python/quadratureshap_rapids_benchmark.py"
MODEL_CACHE = ROOT / "experiments/2026-04-10-rapids-style-shap-benchmark/model-cache"
DEFAULT_MODELS = [
    "adult-small",
    "cal_housing-small",
    "covtype-small",
    "fashion_mnist-small",
]


@dataclass
class PyTree:
    left: np.ndarray
    right: np.ndarray
    missing: np.ndarray
    feature: np.ndarray
    threshold: np.ndarray
    categories: list[set[int] | None]
    value: np.ndarray
    cover: np.ndarray
    child_weight: np.ndarray


def load_benchmark_module():
    spec = importlib.util.spec_from_file_location("qbench", BENCH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_treegrad(treegrad_root: Path):
    sys.path.insert(0, str(treegrad_root))
    from TreeGrad import treestab  # pylint: disable=import-outside-toplevel

    return treestab


def convert_feature_column(booster: xgb.Booster, frame):
    names = booster.feature_names or [f"f{i}" for i in range(booster.num_features())]
    feature_map = {name: i for i, name in enumerate(names)}
    feature_map["Leaf"] = -2
    frame["Feature"] = frame["Feature"].astype(object).replace(feature_map)


def parse_trees(booster: xgb.Booster) -> list[PyTree]:
    frame = booster.trees_to_dataframe()
    convert_feature_column(booster, frame)
    trees = []
    for _, tree_df in frame.groupby("Tree"):
        tree_df = tree_df.sort_values("Node")
        node_lookup = {node_id: i for i, node_id in enumerate(tree_df["ID"])}

        def child_index(value):
            if not isinstance(value, str) or value == "":
                return -1
            if isinstance(value, float) and math.isnan(value):
                return -1
            return node_lookup.get(value, -1)

        left = np.array([child_index(v) for v in tree_df["Yes"]], dtype=np.int32)
        right = np.array([child_index(v) for v in tree_df["No"]], dtype=np.int32)
        cover = tree_df["Cover"].to_numpy(dtype=np.float64)
        child_weight = np.ones_like(cover)
        for parent, child in enumerate(left):
            if child >= 0:
                child_weight[child] = cover[child] / cover[parent]
        for parent, child in enumerate(right):
            if child >= 0:
                child_weight[child] = cover[child] / cover[parent]

        trees.append(
            PyTree(
                left=left,
                right=right,
                missing=np.array(
                    [child_index(v) for v in tree_df["Missing"]], dtype=np.int32
                ),
                feature=tree_df["Feature"].to_numpy(dtype=np.int32),
                threshold=tree_df["Split"].to_numpy(dtype=np.float64),
                categories=[
                    set(int(item) for item in value)
                    if isinstance(value, list)
                    else None
                    for value in tree_df["Category"]
                ],
                value=tree_df["Gain"].to_numpy(dtype=np.float64),
                cover=cover,
                child_weight=child_weight,
            )
        )
    return trees


def has_categorical_splits(trees: list[PyTree]) -> bool:
    return any(category is not None for tree in trees for category in tree.categories)


def max_depth(tree: PyTree) -> int:
    def walk(node: int, depth: int) -> int:
        if tree.feature[node] < 0:
            return depth
        return max(
            walk(int(tree.left[node]), depth + 1),
            walk(int(tree.right[node]), depth + 1),
        )

    return walk(0, 0)


def xgboost_to_treegrad_model(trees: list[PyTree]):
    estimators = []
    for tree in trees:
        if tree.feature[0] < 0:
            continue
        threshold = tree.threshold.astype(np.float64).copy()
        split_mask = tree.feature >= 0
        # XGBoost sends values left for x < split; sklearn-style trees use
        # x <= threshold. Nudge split thresholds down so TreeGrad traverses the
        # converted tree with XGBoost's strict-left semantics.
        threshold[split_mask] = np.nextafter(threshold[split_mask], -np.inf)
        value = tree.value.reshape(-1, 1, 1).astype(np.float64)
        sklearn_tree = SimpleNamespace(
            children_left=tree.left.astype(np.int64),
            children_right=tree.right.astype(np.int64),
            feature=tree.feature.astype(np.int64),
            threshold=threshold,
            n_node_samples=tree.cover.astype(np.float64),
            value=value,
            max_depth=max_depth(tree),
        )
        estimators.append([SimpleNamespace(tree_=sklearn_tree)])
    return SimpleNamespace(
        estimators_=np.asarray(estimators, dtype=object), learning_rate=1.0
    )


@njit
def child_basis8(nodes, basis, p_edge, old, old_seen, child_basis):
    alpha_edge = p_edge - 1.0
    alpha_old = old - 1.0
    for i in range(8):
        value = basis[i] * (1.0 + alpha_edge * nodes[i])
        if old_seen and old != 1.0:
            value /= 1.0 + alpha_old * nodes[i]
        child_basis[i] = value


@njit
def leaf8(basis, leaf_value, weight_prod, out):
    scale = leaf_value * weight_prod
    for i in range(8):
        out[i] = basis[i] * scale


@njit
def add8(lhs, rhs):
    for i in range(8):
        lhs[i] += rhs[i]


@njit
def extract_order1_8(nodes, weights, h_child, p_edge, p_up):
    alpha_edge = p_edge - 1.0
    alpha_up = p_up - 1.0
    acc = 0.0
    for i in range(8):
        edge_delta = alpha_edge / (1.0 + alpha_edge * nodes[i])
        edge_delta -= alpha_up / (1.0 + alpha_up * nodes[i])
        acc += weights[i] * h_child[i] * edge_delta
    return acc


class OrderOneQuadrature:
    def __init__(self, trees: list[PyTree], n_features: int, points: int):
        self.trees = trees
        self.n_features = n_features
        self.nodes, self.weights = np.polynomial.legendre.leggauss(points)
        self.nodes = 0.5 * (self.nodes + 1.0)
        self.weights = 0.5 * self.weights
        self.use_numba = points == 8
        if self.use_numba:
            one = np.ones(8, dtype=np.float64)
            tmp = np.empty(8, dtype=np.float64)
            child_basis8(self.nodes, one, 1.0, 1.0, False, tmp)
            leaf8(one, 1.0, 1.0, tmp)
            add8(tmp, one)
            extract_order1_8(self.nodes, self.weights, tmp, 1.0, 1.0)

    def explain(self, row: np.ndarray) -> np.ndarray:
        out = np.zeros(self.n_features, dtype=np.float64)
        for tree in self.trees:
            path_prob = np.full(self.n_features, np.nan, dtype=np.float64)
            if self.use_numba:
                self._dfs_numba(
                    tree, row, 0, np.ones(8, dtype=np.float64), 1.0, path_prob, out
                )
                continue
            self._dfs(tree, row, 0, np.ones_like(self.nodes), 1.0, path_prob, out)
        return out

    def _dfs(self, tree, row, node, basis, weight_prod, path_prob, out):
        feature = int(tree.feature[node])
        if feature < 0:
            return basis * tree.value[node] * weight_prod

        left = int(tree.left[node])
        right = int(tree.right[node])
        fvalue = row[feature]
        if tree.categories[node] is not None:
            if np.isnan(fvalue) or fvalue < 0:
                hot_child = int(tree.missing[node])
            else:
                hot_child = left if int(fvalue) in tree.categories[node] else right
        elif np.isnan(fvalue):
            hot_child = int(tree.missing[node])
        else:
            hot_child = left if fvalue < tree.threshold[node] else right

        total = np.zeros_like(self.nodes)
        for child in (left, right):
            child_weight = tree.child_weight[child]
            satisfies = child == hot_child
            old = path_prob[feature]
            if np.isnan(old):
                p_edge = (1.0 / child_weight) if satisfies else 0.0
                p_up = 1.0
            elif old == 0.0:
                p_edge = 0.0
                p_up = 0.0
            else:
                p_edge = (old / child_weight) if satisfies else 0.0
                p_up = old

            child_basis = basis * (1.0 + (p_edge - 1.0) * self.nodes)
            if not np.isnan(old) and old != 1.0:
                child_basis = child_basis / (1.0 + (old - 1.0) * self.nodes)

            path_prob[feature] = p_edge
            h_child = self._dfs(
                tree,
                row,
                child,
                child_basis,
                weight_prod * child_weight,
                path_prob,
                out,
            )
            self._extract(feature, p_edge, p_up, h_child, out)
            path_prob[feature] = old
            total += h_child
        return total

    def _extract(self, feature, p_edge, p_up, h_child, out):
        alpha_edge = p_edge - 1.0
        alpha_up = p_up - 1.0
        edge_delta = alpha_edge / (1.0 + alpha_edge * self.nodes)
        edge_delta -= alpha_up / (1.0 + alpha_up * self.nodes)
        out[feature] += float(np.sum(self.weights * h_child * edge_delta))

    def _dfs_numba(self, tree, row, node, basis, weight_prod, path_prob, out):
        feature = int(tree.feature[node])
        h_out = np.empty(8, dtype=np.float64)
        if feature < 0:
            leaf8(basis, tree.value[node], weight_prod, h_out)
            return h_out

        left = int(tree.left[node])
        right = int(tree.right[node])
        fvalue = row[feature]
        if tree.categories[node] is not None:
            if np.isnan(fvalue) or fvalue < 0:
                hot_child = int(tree.missing[node])
            else:
                hot_child = left if int(fvalue) in tree.categories[node] else right
        elif np.isnan(fvalue):
            hot_child = int(tree.missing[node])
        else:
            hot_child = left if fvalue < tree.threshold[node] else right

        total = np.zeros(8, dtype=np.float64)
        for child in (left, right):
            child_weight = tree.child_weight[child]
            satisfies = child == hot_child
            old = path_prob[feature]
            old_seen = not np.isnan(old)
            if not old_seen:
                p_edge = (1.0 / child_weight) if satisfies else 0.0
                p_up = 1.0
            elif old == 0.0:
                p_edge = 0.0
                p_up = 0.0
            else:
                p_edge = (old / child_weight) if satisfies else 0.0
                p_up = old

            child_basis = np.empty(8, dtype=np.float64)
            child_basis8(
                self.nodes,
                basis,
                p_edge,
                old if old_seen else 1.0,
                old_seen,
                child_basis,
            )
            path_prob[feature] = p_edge
            h_child = self._dfs_numba(
                tree,
                row,
                child,
                child_basis,
                weight_prod * child_weight,
                path_prob,
                out,
            )
            out[feature] += extract_order1_8(
                self.nodes, self.weights, h_child, p_edge, p_up
            )
            path_prob[feature] = old
            add8(total, h_child)
        return total


def default_points(trees: list[PyTree], n_features: int) -> int:
    return max(
        1, math.ceil(min(max(max_depth(tree) for tree in trees), n_features) / 2)
    )


def to_numeric_matrix(x_test) -> np.ndarray:
    if hasattr(x_test, "select_dtypes"):
        converted = x_test.copy()
        for column in converted.select_dtypes(include=["category"]).columns:
            converted[column] = converted[column].cat.codes
        return converted.to_numpy(dtype=np.float64)
    return np.asarray(x_test, dtype=np.float64)


def run_model(
    model_name: str, rows: int, quadrature_points: int, seed: int, treegrad_root: Path
):
    qbench = load_benchmark_module()
    treestab = load_treegrad(treegrad_root)
    dataset_name = model_name.rsplit("-", 1)[0]
    dataset = next(ds for ds in qbench.get_test_datasets() if ds.name == dataset_name)

    booster = xgb.Booster()
    booster.load_model(MODEL_CACHE / f"{model_name}.ubj")
    n_features = booster.num_features()

    t0 = time.perf_counter()
    trees = parse_trees(booster)

    x_test = dataset.test_input(rows, seed)
    x_np = to_numeric_matrix(x_test)
    x_np = np.atleast_2d(x_np)
    treegrad_points = default_points(trees, n_features)
    quadrature = OrderOneQuadrature(trees, n_features, quadrature_points)
    q_construct = time.perf_counter() - t0

    supports_treegrad = not has_categorical_splits(trees)
    treegrad_model = None
    tg_construct = None
    if supports_treegrad:
        t0 = time.perf_counter()
        treegrad_model = xgboost_to_treegrad_model(trees)
        tg_construct = time.perf_counter() - t0

    q_times = []
    tg_times = []
    max_abs_diffs = []
    mean_abs_diffs = []
    for row in x_np:
        t0 = time.perf_counter()
        q_values = quadrature.explain(row)
        q_times.append(time.perf_counter() - t0)

        if supports_treegrad:
            t0 = time.perf_counter()
            tg_values = treestab(treegrad_model, row, (1, 1), class_index=None)
            tg_times.append(time.perf_counter() - t0)

            diffs = np.abs(q_values - tg_values)
            max_abs_diffs.append(float(np.max(diffs)))
            mean_abs_diffs.append(float(np.mean(diffs)))

    result = {
        "model": model_name,
        "rows": rows,
        "quadrature_points": quadrature_points,
        "treegrad_points": treegrad_points,
        "has_categorical_splits": has_categorical_splits(trees),
        "n_features": n_features,
        "num_trees": len(trees),
        "quadrature_construct_s": q_construct,
        "treegrad_construct_s": tg_construct,
        "quadrature_total_s": float(sum(q_times)),
        "treegrad_total_s": float(sum(tg_times)) if supports_treegrad else None,
        "quadrature_mean_row_s": float(np.mean(q_times)),
        "treegrad_mean_row_s": float(np.mean(tg_times)) if supports_treegrad else None,
        "speedup_treegrad_over_quadrature": (
            float(sum(tg_times) / sum(q_times))
            if supports_treegrad and sum(q_times)
            else None
        ),
        "max_abs_diff": (
            float(max(max_abs_diffs, default=0.0)) if supports_treegrad else None
        ),
        "mean_abs_diff": (
            float(np.mean(mean_abs_diffs))
            if supports_treegrad and mean_abs_diffs
            else None
        ),
    }
    if not supports_treegrad:
        result["treegrad_error"] = (
            "categorical XGBoost splits are not supported by the TreeGrad adapter"
        )
    return result


def worker(queue, model_name, rows, quadrature_points, seed, treegrad_root):
    try:
        queue.put(
            {
                "ok": True,
                "result": run_model(
                    model_name, rows, quadrature_points, seed, treegrad_root
                ),
            }
        )
    except Exception as exc:  # noqa: BLE001
        queue.put(
            {
                "ok": False,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "model": model_name,
            }
        )


def run_with_timeout(model_name: str, args):
    queue = mp.Queue()
    proc = mp.Process(
        target=worker,
        args=(
            queue,
            model_name,
            args.rows,
            args.quadrature_points,
            args.seed,
            args.treegrad_root,
        ),
    )
    proc.start()
    proc.join(args.timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        return {"model": model_name, "error": f"timeout after {args.timeout}s"}
    if queue.empty():
        return {
            "model": model_name,
            "error": f"worker exited with code {proc.exitcode}",
        }
    payload = queue.get()
    if not payload["ok"]:
        return {
            "model": model_name,
            "error": payload["error"],
            "traceback": payload.get("traceback"),
        }
    return payload["result"]


def markdown_table(rows: list[dict[str, object]]) -> str:
    columns = [
        ("model", "model"),
        ("rows", "rows"),
        ("quadrature_points", "quadrature points"),
        ("treegrad_points", "TreeGrad points"),
        ("has_categorical_splits", "categorical"),
        ("quadrature_total_s", "quadrature s"),
        ("treegrad_total_s", "TreeGrad s"),
        ("speedup_treegrad_over_quadrature", "TreeGrad / quadrature"),
        ("max_abs_diff", "max abs diff"),
        ("mean_abs_diff", "mean abs diff"),
        ("treegrad_error", "TreeGrad error"),
        ("error", "error"),
    ]
    lines = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        values = []
        for key, _ in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                values.append(f"{value:.3e}" if "diff" in key else f"{value:.3f}")
            elif value is None:
                values.append("")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def write_outputs(args, results: list[dict[str, object]]) -> None:
    payload = {
        "rows": args.rows,
        "quadrature_points": args.quadrature_points,
        "models": args.models,
        "treegrad_root": str(args.treegrad_root),
        "complete": len(results) == len(args.models),
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.out_md.write_text(markdown_table(results), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--rows", type=int, default=100)
    parser.add_argument("--quadrature-points", type=int, default=8)
    parser.add_argument("--seed", type=int, default=432)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--treegrad-root", type=Path, default=Path("/tmp/TreeGrad"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).with_name("results-treegrad-first-order-row100.json"),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path(__file__).with_name("treegrad-first-order-row100.md"),
    )
    args = parser.parse_args()

    results = []
    for model_name in args.models:
        print(f"Running {model_name}", flush=True)
        result = run_with_timeout(model_name, args)
        results.append(result)
        print(json.dumps(result, indent=2), flush=True)
        write_outputs(args, results)


if __name__ == "__main__":
    main()
