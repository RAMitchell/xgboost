"""Benchmark Shapley interaction indices against TreeSHAP-IQ."""
# pylint: disable=missing-function-docstring,missing-class-docstring,too-many-instance-attributes,too-many-arguments,too-many-positional-arguments,too-many-locals,broad-exception-caught,cell-var-from-loop

import argparse
import importlib.util
import itertools
import json
import math
import multiprocessing as mp
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xgboost as xgb
from shapiq.explainer.tree import TreeExplainer
from shapiq.explainer.tree.conversion.xgboost import _convert_xgboost_tree_as_df

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "demo/guide-python/quadratureshap_rapids_benchmark.py"
MODEL_CACHE = ROOT / "experiments/2026-04-10-rapids-style-shap-benchmark/model-cache"
DEFAULT_MODELS = ["cal_housing-small", "cal_housing-sparse"]
BERNOULLI = {
    0: 1.0,
    1: -0.5,
    2: 1.0 / 6.0,
    3: 0.0,
}


def load_benchmark_module():
    spec = importlib.util.spec_from_file_location("qbench", BENCH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


def convert_feature_column(booster: xgb.Booster, frame):
    names = booster.feature_names or [f"f{i}" for i in range(booster.num_features())]
    feature_map = {name: i for i, name in enumerate(names)}
    feature_map["Leaf"] = -2
    # pandas 3 can use Arrow string columns here; replacing with ints in-place fails.
    frame["Feature"] = frame["Feature"].astype(object).replace(feature_map)


def learner_model_params(booster: xgb.Booster):
    return json.loads(booster.save_config())["learner"]["learner_model_param"]


def is_scalar_output(booster: xgb.Booster) -> bool:
    params = learner_model_params(booster)
    return int(params["num_class"]) == 0 and int(params["num_target"]) <= 1


def base_score(booster: xgb.Booster) -> float:
    params = learner_model_params(booster)
    value = params["base_score"]
    if isinstance(value, str):
        value = value.strip("[]")
        return float(sum(float(part) for part in value.split(",")))
    return float(np.sum(value))


def convert_for_shapiq(booster: xgb.Booster):
    frame = booster.trees_to_dataframe()
    if frame["Category"].notna().any():
        raise ValueError("This harness currently supports numeric-only XGBoost trees.")
    convert_feature_column(booster, frame)
    trees = []
    for _, tree_df in frame.groupby("Tree"):
        # Leaf-only trees affect the baseline but not order-s interactions for s > 0.
        if (tree_df["Feature"] < 0).all():
            continue
        trees.append(
            _convert_xgboost_tree_as_df(
                tree_df, intercept=0.0, output_type="raw", scaling=1.0
            )
        )
    return trees


def parse_trees(booster: xgb.Booster) -> list[PyTree]:
    frame = booster.trees_to_dataframe()
    if frame["Category"].notna().any():
        raise ValueError("This harness currently supports numeric-only XGBoost trees.")
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


class OrderQuadrature:
    def __init__(self, trees: list[PyTree], n_features: int, order: int, points: int):
        self.trees = trees
        self.n_features = n_features
        self.order = order
        self.nodes, self.weights = np.polynomial.legendre.leggauss(points)
        self.nodes = 0.5 * (self.nodes + 1.0)
        self.weights = 0.5 * self.weights
        self._subset_cache: dict[
            tuple[int, tuple[int, ...]], list[tuple[tuple[int, ...], tuple[int, ...]]]
        ] = {}

    def explain(self, row: np.ndarray) -> dict[tuple[int, ...], float]:
        out: dict[tuple[int, ...], float] = {}
        for tree in self.trees:
            path_prob = np.full(self.n_features, np.nan, dtype=np.float64)
            active_mask = np.zeros(self.n_features, dtype=np.bool_)
            active: list[int] = []
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
        self,
        tree: PyTree,
        row: np.ndarray,
        node: int,
        basis: np.ndarray,
        weight_prod: float,
        path_prob: np.ndarray,
        active_mask: np.ndarray,
        active: list[int],
        out: dict[tuple[int, ...], float],
    ) -> np.ndarray:
        feature = int(tree.feature[node])
        if feature < 0:
            return basis * tree.value[node] * weight_prod

        left = int(tree.left[node])
        right = int(tree.right[node])
        children = (left, right)
        fvalue = row[feature]
        if np.isnan(fvalue):
            hot_child = int(tree.missing[node])
        else:
            hot_child = left if fvalue < tree.threshold[node] else right

        total = np.zeros_like(self.nodes)
        for child in children:
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

    def _extract(
        self,
        feature: int,
        p_edge: float,
        p_up: float,
        h_child: np.ndarray,
        path_prob: np.ndarray,
        active: list[int],
        out: dict[tuple[int, ...], float],
    ) -> None:
        subset_data = self._subsets_for_active(feature, active)
        if not subset_data:
            return

        alpha_edge = p_edge - 1.0
        alpha_up = p_up - 1.0
        edge_delta = alpha_edge / (1.0 + alpha_edge * self.nodes)
        edge_delta -= alpha_up / (1.0 + alpha_up * self.nodes)
        base = self.weights * h_child * edge_delta
        partner_terms: dict[int, np.ndarray | None] = {}

        for subset, key in subset_data:
            gamma = np.ones_like(self.nodes)
            skip = False
            for partner in subset:
                term = partner_terms.get(partner)
                if term is None and partner not in partner_terms:
                    p_partner = path_prob[partner]
                    if p_partner == 1.0:
                        partner_terms[partner] = None
                        term = None
                    else:
                        alpha_partner = p_partner - 1.0
                        term = alpha_partner / (1.0 + alpha_partner * self.nodes)
                        partner_terms[partner] = term
                if term is None:
                    skip = True
                    break
                gamma *= term
            if skip:
                continue
            out[key] = out.get(key, 0.0) + float(np.sum(base * gamma))

    def _subsets_for_active(
        self, feature: int, active: list[int]
    ) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
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


def dict_from_interaction_values(iv, order: int) -> dict[tuple[int, ...], float]:
    result = {}
    for key, value in iv.dict_values.items():
        key = tuple(int(k) for k in key)
        if len(key) == order:
            result[key] = float(value)
    return result


def nonempty_dict_from_interaction_values(
    iv, max_order: int
) -> dict[tuple[int, ...], float]:
    result = {}
    for key, value in iv.dict_values.items():
        key = tuple(int(k) for k in key)
        if 0 < len(key) <= max_order:
            result[key] = float(value)
    return result


def aggregate_sii_to_k_sii(
    sii_values: dict[tuple[int, ...], float], max_order: int
) -> dict[tuple[int, ...], float]:
    result: dict[tuple[int, ...], float] = {}
    for base_interaction, base_value in sii_values.items():
        base_size = len(base_interaction)
        for size in range(1, min(base_size, max_order) + 1):
            scaling = BERNOULLI[base_size - size]
            if scaling == 0.0:
                continue
            for interaction in itertools.combinations(base_interaction, size):
                result[interaction] = (
                    result.get(interaction, 0.0) + scaling * base_value
                )
    return result


def count_nonzero(values: dict[tuple[int, ...], float], tol: float = 1e-12) -> int:
    return sum(abs(value) > tol for value in values.values())


def compare_dicts(lhs: dict[tuple[int, ...], float], rhs: dict[tuple[int, ...], float]):
    keys = set(lhs) | set(rhs)
    if not keys:
        return {"max_abs_diff": 0.0, "mean_abs_diff": 0.0, "nnz_lhs": 0, "nnz_rhs": 0}
    diffs = [abs(lhs.get(k, 0.0) - rhs.get(k, 0.0)) for k in keys]
    return {
        "max_abs_diff": float(max(diffs)),
        "mean_abs_diff": float(np.mean(diffs)),
        "nnz_lhs": count_nonzero(lhs),
        "nnz_rhs": count_nonzero(rhs),
    }


def run_model(
    model_name: str, rows: int, order: int, points: int, index: str, seed: int
):
    qbench = load_benchmark_module()
    dataset_name = model_name.rsplit("-", 1)[0]
    dataset = next(ds for ds in qbench.get_test_datasets() if ds.name == dataset_name)
    x_test = dataset.test_input(rows, seed)
    x_np = (
        x_test.to_numpy(dtype=np.float64)
        if hasattr(x_test, "to_numpy")
        else np.asarray(x_test)
    )
    x_np = np.atleast_2d(x_np)

    booster = xgb.Booster()
    booster.load_model(MODEL_CACHE / f"{model_name}.ubj")
    n_features = booster.num_features()

    t0 = time.perf_counter()
    q_trees = parse_trees(booster)
    q_explainers = {
        q_order: OrderQuadrature(q_trees, n_features, q_order, points)
        for q_order in (range(1, order + 1) if index == "k-SII" else [order])
    }
    q_construct = time.perf_counter() - t0

    t0 = time.perf_counter()
    iq_trees = convert_for_shapiq(booster)
    min_order = 0 if index == "k-SII" else order
    iq_explainer = TreeExplainer(
        iq_trees, max_order=order, min_order=min_order, index=index
    )
    iq_construct = time.perf_counter() - t0

    q_times = []
    iq_times = []
    diffs = []
    iq_efficiency_errors = []
    can_check_efficiency = index == "k-SII" and is_scalar_output(booster)
    xgb_base_score = base_score(booster)
    for row in x_np:
        t0 = time.perf_counter()
        if index == "k-SII":
            q_sii_values = {}
            for q_order, q_explainer in q_explainers.items():
                q_sii_values.update(
                    q_explainer.explain(np.asarray(row, dtype=np.float64))
                )
            q_values = aggregate_sii_to_k_sii(q_sii_values, order)
        else:
            q_values = q_explainers[order].explain(np.asarray(row, dtype=np.float64))
        q_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        iq_all_values = iq_explainer.explain(np.asarray(row))
        if index == "k-SII":
            iq_values = nonempty_dict_from_interaction_values(iq_all_values, order)
        else:
            iq_values = dict_from_interaction_values(iq_all_values, order)
        iq_times.append(time.perf_counter() - t0)
        diffs.append(compare_dicts(q_values, iq_values))

        if can_check_efficiency:
            raw_margin = float(
                np.sum(
                    booster.predict(
                        xgb.DMatrix(np.asarray(row, dtype=np.float64).reshape(1, -1)),
                        output_margin=True,
                    )
                )
            )
            tree_margin = raw_margin - xgb_base_score
            iq_efficiency_errors.append(
                abs(float(sum(iq_all_values.dict_values.values())) - tree_margin)
            )

    result = {
        "model": model_name,
        "rows": rows,
        "order": order,
        "index": index,
        "points": points,
        "n_features": n_features,
        "num_trees": len(q_trees),
        "quadrature_construct_s": q_construct,
        "treeshapiq_construct_s": iq_construct,
        "quadrature_total_s": float(sum(q_times)),
        "treeshapiq_total_s": float(sum(iq_times)),
        "quadrature_mean_row_s": float(np.mean(q_times)),
        "treeshapiq_mean_row_s": float(np.mean(iq_times)),
        "speedup": float(sum(iq_times) / sum(q_times)) if sum(q_times) else None,
        "max_abs_diff": float(max((d["max_abs_diff"] for d in diffs), default=0.0)),
        "mean_abs_diff": float(np.mean([d["mean_abs_diff"] for d in diffs]))
        if diffs
        else 0.0,
        "quadrature_nnz_max": int(max((d["nnz_lhs"] for d in diffs), default=0)),
        "treeshapiq_nnz_max": int(max((d["nnz_rhs"] for d in diffs), default=0)),
    }
    if index == "k-SII":
        result["treeshapiq_efficiency_error_max"] = float(
            max(iq_efficiency_errors, default=0.0)
        )
        result["treeshapiq_efficiency_error_mean"] = (
            float(np.mean(iq_efficiency_errors)) if iq_efficiency_errors else 0.0
        )
        result["treeshapiq_efficiency_checked"] = can_check_efficiency
    return result


def worker(queue, model_name, rows, order, points, index, seed):
    try:
        queue.put(
            {
                "ok": True,
                "result": run_model(model_name, rows, order, points, index, seed),
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
            args.order,
            args.points,
            args.index,
            args.seed,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--order", type=int, default=3)
    parser.add_argument("--index", default="k-SII")
    parser.add_argument("--points", type=int, default=8)
    parser.add_argument("--seed", type=int, default=432)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument(
        "--out", type=Path, default=Path(__file__).with_name("results-order3.json")
    )
    args = parser.parse_args()

    results = []
    for model_name in args.models:
        print(f"Running {model_name}", flush=True)
        result = run_with_timeout(model_name, args)
        results.append(result)
        print(json.dumps(result, indent=2), flush=True)

    payload = {
        "rows": args.rows,
        "order": args.order,
        "index": args.index,
        "points": args.points,
        "models": args.models,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
