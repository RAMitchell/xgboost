"""Explore Numba JIT for constant-8 quadrature vector operations."""
# pylint: disable=missing-function-docstring,missing-class-docstring,too-many-arguments,too-many-locals,protected-access,cell-var-from-loop,too-many-instance-attributes,too-many-positional-arguments,duplicate-code

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xgboost as xgb
from numba import njit

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "demo/guide-python/quadratureshap_rapids_benchmark.py"
MODEL_CACHE = ROOT / "experiments/2026-04-10-rapids-style-shap-benchmark/model-cache"
HERE = Path(__file__).resolve().parent


@dataclass
class PyTree:
    left: np.ndarray
    right: np.ndarray
    missing: np.ndarray
    feature: np.ndarray
    threshold: np.ndarray
    value: np.ndarray
    cover: np.ndarray
    child_weight: np.ndarray


def load_benchmark_module():
    spec = importlib.util.spec_from_file_location("qbench", BENCH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def convert_feature_column(booster: xgb.Booster, frame):
    names = booster.feature_names or [f"f{i}" for i in range(booster.num_features())]
    feature_map = {name: i for i, name in enumerate(names)}
    feature_map["Leaf"] = -2
    frame["Feature"] = frame["Feature"].astype(object).replace(feature_map)


def parse_trees(booster: xgb.Booster) -> list[PyTree]:
    frame = booster.trees_to_dataframe()
    if frame["Category"].notna().any():
        raise ValueError("categorical XGBoost splits are not supported by this harness")
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
                value=tree_df["Gain"].to_numpy(dtype=np.float64),
                cover=cover,
                child_weight=child_weight,
            )
        )
    return trees


class NumpyOrderQuadrature:
    def __init__(self, trees: list[PyTree], n_features: int, order: int, points: int):
        self.trees = trees
        self.n_features = n_features
        self.order = order
        self.nodes, self.weights = np.polynomial.legendre.leggauss(points)
        self.nodes = 0.5 * (self.nodes + 1.0)
        self.weights = 0.5 * self.weights
        self._subset_cache = {}

    def explain(self, row: np.ndarray) -> dict[tuple[int, ...], float]:
        out = {}
        for tree in self.trees:
            path_prob = np.full(self.n_features, np.nan, dtype=np.float64)
            active_mask = np.zeros(self.n_features, dtype=np.bool_)
            active = []
            self._dfs(
                tree,
                row,
                0,
                np.ones_like(self.nodes),
                1.0,
                path_prob,
                active_mask,
                active,
                out,
            )
        return out

    def _dfs(
        self, tree, row, node, basis, weight_prod, path_prob, active_mask, active, out
    ):
        feature = int(tree.feature[node])
        if feature < 0:
            return basis * tree.value[node] * weight_prod

        left = int(tree.left[node])
        right = int(tree.right[node])
        fvalue = row[feature]
        if np.isnan(fvalue):
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
                was_active = bool(active_mask[feature])
            elif old == 0.0:
                p_edge = 0.0
                p_up = 0.0
                was_active = True
            else:
                p_edge = (old / child_weight) if satisfies else 0.0
                p_up = old
                was_active = True

            child_basis = basis * (1.0 + (p_edge - 1.0) * self.nodes)
            if not np.isnan(old) and old != 1.0:
                child_basis = child_basis / (1.0 + (old - 1.0) * self.nodes)

            path_prob[feature] = p_edge
            if not was_active:
                active_mask[feature] = True
                active.append(feature)
            h_child = self._dfs(
                tree,
                row,
                child,
                child_basis,
                weight_prod * child_weight,
                path_prob,
                active_mask,
                active,
                out,
            )
            self._extract(feature, p_edge, p_up, h_child, path_prob, active, out)
            if not was_active:
                active.pop()
                active_mask[feature] = False
            path_prob[feature] = old
            total += h_child
        return total

    def _extract(self, feature, p_edge, p_up, h_child, path_prob, active, out):
        subset_data = self._subsets_for_active(feature, active)
        if not subset_data:
            return
        alpha_edge = p_edge - 1.0
        alpha_up = p_up - 1.0
        edge_delta = alpha_edge / (1.0 + alpha_edge * self.nodes)
        edge_delta -= alpha_up / (1.0 + alpha_up * self.nodes)
        base = self.weights * h_child * edge_delta
        for subset, key in subset_data:
            gamma = np.ones_like(self.nodes)
            skip = False
            for partner in subset:
                p_partner = path_prob[partner]
                if p_partner == 1.0:
                    skip = True
                    break
                alpha_partner = p_partner - 1.0
                gamma *= alpha_partner / (1.0 + alpha_partner * self.nodes)
            if not skip:
                out[key] = out.get(key, 0.0) + float(np.sum(base * gamma))

    def _subsets_for_active(self, feature, active):
        cache_key = (feature, tuple(active))
        cached = self._subset_cache.get(cache_key)
        if cached is not None:
            return cached
        partners = tuple(f for f in active if f != feature)
        if len(partners) < self.order - 1:
            cached = []
        else:
            cached = [
                (subset, tuple(sorted((feature, *subset))))
                for subset in itertools.combinations(partners, self.order - 1)
            ]
        self._subset_cache[cache_key] = cached
        return cached


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


@njit
def extract_subset8(nodes, weights, h_child, path_prob, subset, p_edge, p_up):
    alpha_edge = p_edge - 1.0
    alpha_up = p_up - 1.0
    acc = 0.0
    for i in range(8):
        edge_delta = alpha_edge / (1.0 + alpha_edge * nodes[i])
        edge_delta -= alpha_up / (1.0 + alpha_up * nodes[i])
        gamma = 1.0
        for j in range(subset.shape[0]):
            p_partner = path_prob[subset[j]]
            if p_partner == 1.0:
                return 0.0
            alpha_partner = p_partner - 1.0
            gamma *= alpha_partner / (1.0 + alpha_partner * nodes[i])
        acc += weights[i] * h_child[i] * edge_delta * gamma
    return acc


class NumbaOrderQuadrature(NumpyOrderQuadrature):
    def __init__(self, trees, n_features: int, order: int, points: int):
        if points != 8:
            raise ValueError(
                "Numba constant-size variant currently requires --points 8"
            )
        super().__init__(trees, n_features, order, points)
        one = np.ones(8, dtype=np.float64)
        tmp = np.empty(8, dtype=np.float64)
        child_basis8(self.nodes, one, 1.0, 1.0, False, tmp)
        leaf8(one, 1.0, 1.0, tmp)
        add8(tmp, one)
        extract_order1_8(self.nodes, self.weights, tmp, 1.0, 1.0)
        extract_subset8(
            self.nodes,
            self.weights,
            tmp,
            np.ones(n_features),
            np.array([0], dtype=np.int64),
            1.0,
            1.0,
        )

    def explain(self, row: np.ndarray) -> dict[tuple[int, ...], float]:
        out = {}
        for tree in self.trees:
            path_prob = np.full(self.n_features, np.nan, dtype=np.float64)
            active_mask = np.zeros(self.n_features, dtype=np.bool_)
            active = []
            self._dfs_numba(
                tree,
                row,
                0,
                np.ones(8, dtype=np.float64),
                1.0,
                path_prob,
                active_mask,
                active,
                out,
            )
        return out

    def _dfs_numba(
        self, tree, row, node, basis, weight_prod, path_prob, active_mask, active, out
    ):
        feature = int(tree.feature[node])
        h_out = np.empty(8, dtype=np.float64)
        if feature < 0:
            leaf8(basis, tree.value[node], weight_prod, h_out)
            return h_out

        left = int(tree.left[node])
        right = int(tree.right[node])
        fvalue = row[feature]
        if np.isnan(fvalue):
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
                was_active = bool(active_mask[feature])
            elif old == 0.0:
                p_edge = 0.0
                p_up = 0.0
                was_active = True
            else:
                p_edge = (old / child_weight) if satisfies else 0.0
                p_up = old
                was_active = True

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
            if not was_active:
                active_mask[feature] = True
                active.append(feature)
            h_child = self._dfs_numba(
                tree,
                row,
                child,
                child_basis,
                weight_prod * child_weight,
                path_prob,
                active_mask,
                active,
                out,
            )
            self._extract_numba(feature, p_edge, p_up, h_child, path_prob, active, out)
            if not was_active:
                active.pop()
                active_mask[feature] = False
            path_prob[feature] = old
            add8(total, h_child)
        return total

    def _subsets_for_active(self, feature, active):
        cache_key = (feature, tuple(active), "numba")
        cached = self._subset_cache.get(cache_key)
        if cached is not None:
            return cached
        partners = tuple(f for f in active if f != feature)
        if len(partners) < self.order - 1:
            cached = []
        else:
            cached = [
                (np.asarray(subset, dtype=np.int64), tuple(sorted((feature, *subset))))
                for subset in itertools.combinations(partners, self.order - 1)
            ]
        self._subset_cache[cache_key] = cached
        return cached

    def _extract_numba(self, feature, p_edge, p_up, h_child, path_prob, active, out):
        subset_data = self._subsets_for_active(feature, active)
        if not subset_data:
            return
        if self.order == 1:
            out[(feature,)] = out.get((feature,), 0.0) + extract_order1_8(
                self.nodes, self.weights, h_child, p_edge, p_up
            )
            return
        for subset, key in subset_data:
            delta = extract_subset8(
                self.nodes, self.weights, h_child, path_prob, subset, p_edge, p_up
            )
            if delta != 0.0:
                out[key] = out.get(key, 0.0) + delta


def load_rows(model_name: str, rows: int, seed: int):
    qbench = load_benchmark_module()
    dataset_name = model_name.rsplit("-", 1)[0]
    dataset = next(ds for ds in qbench.get_test_datasets() if ds.name == dataset_name)
    x_test = dataset.test_input(rows, seed)
    x_np = (
        x_test.to_numpy(dtype=np.float64)
        if hasattr(x_test, "to_numpy")
        else np.asarray(x_test, dtype=np.float64)
    )
    return np.atleast_2d(x_np)


def compare_dicts(lhs, rhs):
    keys = set(lhs) | set(rhs)
    if not keys:
        return 0.0, 0.0
    diffs = np.array([abs(lhs.get(key, 0.0) - rhs.get(key, 0.0)) for key in keys])
    return float(np.max(diffs)), float(np.mean(diffs))


def run(model_name: str, rows: int, order: int, seed: int):
    booster = xgb.Booster()
    booster.load_model(MODEL_CACHE / f"{model_name}.ubj")
    trees = parse_trees(booster)
    x_np = load_rows(model_name, rows, seed)
    n_features = booster.num_features()

    numpy_explainer = NumpyOrderQuadrature(trees, n_features, order, 8)
    numba_explainer = NumbaOrderQuadrature(trees, n_features, order, 8)
    numpy_explainer.explain(x_np[0])
    numba_explainer.explain(x_np[0])

    numpy_times = []
    numba_times = []
    max_diffs = []
    mean_diffs = []
    for row in x_np:
        t0 = time.perf_counter()
        numpy_values = numpy_explainer.explain(row)
        numpy_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        numba_values = numba_explainer.explain(row)
        numba_times.append(time.perf_counter() - t0)

        max_diff, mean_diff = compare_dicts(numpy_values, numba_values)
        max_diffs.append(max_diff)
        mean_diffs.append(mean_diff)

    return {
        "model": model_name,
        "rows": rows,
        "order": order,
        "points": 8,
        "numpy_total_s": float(sum(numpy_times)),
        "numba_total_s": float(sum(numba_times)),
        "numpy_mean_row_s": float(np.mean(numpy_times)),
        "numba_mean_row_s": float(np.mean(numba_times)),
        "speedup_numba_over_numpy": float(sum(numpy_times) / sum(numba_times)),
        "max_abs_diff": float(max(max_diffs, default=0.0)),
        "mean_abs_diff": float(np.mean(mean_diffs)) if mean_diffs else 0.0,
    }


def markdown_table(rows):
    lines = [
        "| model | order | rows | NumPy s | Numba s | speedup | max abs diff | mean abs diff |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['order']} | {row['rows']} | {row['numpy_total_s']:.3f} | "
            f"{row['numba_total_s']:.3f} | {row['speedup_numba_over_numpy']:.3f} | "
            f"{row['max_abs_diff']:.3e} | {row['mean_abs_diff']:.3e} |"
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="+", default=["cal_housing-small", "covtype-small"]
    )
    parser.add_argument("--rows", type=int, default=100)
    parser.add_argument("--order", type=int, default=3)
    parser.add_argument("--seed", type=int, default=432)
    parser.add_argument(
        "--out", type=Path, default=HERE / "results-numba-constant8.json"
    )
    parser.add_argument("--out-md", type=Path, default=HERE / "numba-constant8.md")
    args = parser.parse_args()

    results = []
    for model_name in args.models:
        print(f"Running {model_name}", flush=True)
        result = run(model_name, args.rows, args.order, args.seed)
        results.append(result)
        print(json.dumps(result, indent=2), flush=True)

    payload = {"rows": args.rows, "order": args.order, "points": 8, "results": results}
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.out_md.write_text(markdown_table(results), encoding="utf-8")


if __name__ == "__main__":
    main()
