"""Probe Fashion-MNIST sparse trees for feasibility of high-order TreeSHAP-IQ runs."""
# pylint: disable=missing-function-docstring,too-many-locals,line-too-long,protected-access

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import xgboost as xgb
from shapiq.explainer.tree import TreeExplainer
from shapiq.explainer.tree.conversion.xgboost import _convert_xgboost_tree_as_df

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "demo/guide-python/quadratureshap_rapids_benchmark.py"
MODEL_CACHE = ROOT / "experiments/2026-04-10-rapids-style-shap-benchmark/model-cache"


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


def tree_unique_depth(tree_df) -> int:
    by_node = {int(row.Node): row for row in tree_df.itertuples(index=False)}

    def child_node(value):
        if not isinstance(value, str) or not value:
            return None
        return int(value.rsplit("-", 1)[1])

    def walk(node_id: int, features: frozenset[int]) -> int:
        row = by_node[node_id]
        feature = int(row.Feature)
        if feature < 0:
            return len(features)
        next_features = features | {feature}
        return max(
            walk(child_node(row.Yes), next_features),
            walk(child_node(row.No), next_features),
        )

    return walk(0, frozenset())


def row_for_fashion_mnist(seed: int) -> np.ndarray:
    qbench = load_benchmark_module()
    dataset = next(
        ds for ds in qbench.get_test_datasets() if ds.name == "fashion_mnist"
    )
    x = dataset.test_input(1, seed)
    arr = (
        x.to_numpy(dtype=np.float64)
        if hasattr(x, "to_numpy")
        else np.asarray(x, dtype=np.float64)
    )
    return np.atleast_2d(arr)[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="fashion_mnist-sparse")
    parser.add_argument("--top", type=int, default=8)
    parser.add_argument("--order", type=int, default=3)
    parser.add_argument("--seed", type=int, default=432)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).with_name("fashion-sparse-tree-stability.json"),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path(__file__).with_name("fashion-sparse-tree-stability.md"),
    )
    args = parser.parse_args()

    booster = xgb.Booster()
    booster.load_model(MODEL_CACHE / f"{args.model}.ubj")
    frame = booster.trees_to_dataframe()
    if frame["Category"].notna().any():
        raise ValueError("Categorical trees are not supported by this probe.")
    convert_feature_column(booster, frame)
    row = row_for_fashion_mnist(args.seed)

    ranked = []
    for tree_id, tree_df in frame.groupby("Tree"):
        tree_df = tree_df.sort_values("Node")
        unique_depth = tree_unique_depth(tree_df)
        raw_depth = int(tree_df["Depth"].max()) if "Depth" in tree_df else None
        ranked.append((unique_depth, int(tree_id), raw_depth, tree_df))
    ranked.sort(reverse=True, key=lambda item: (item[0], item[2] or 0))

    rows = []
    for unique_depth, tree_id, raw_depth, tree_df in ranked[: args.top]:
        print(f"Running tree {tree_id} unique_depth={unique_depth}", flush=True)
        t0 = time.perf_counter()
        tree = _convert_xgboost_tree_as_df(
            tree_df, intercept=0.0, output_type="raw", scaling=1.0
        )
        convert_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        explainer = TreeExplainer(
            [tree], max_order=args.order, min_order=0, index="k-SII"
        )
        construct_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        values = explainer.explain(row)
        explain_s = time.perf_counter() - t0

        prediction = float(tree.predict_one(row))
        total = float(sum(values.dict_values.values()))
        n_interp = min(
            int(explainer._treeshapiq_explainers[0]._edge_tree.max_depth),
            int(tree.n_features_in_tree),
        )
        cond = float(
            np.linalg.cond(np.vander(np.polynomial.chebyshev.chebpts2(n_interp)).T)
        )
        result = {
            "tree": tree_id,
            "unique_feature_depth": unique_depth,
            "raw_depth": raw_depth,
            "n_nodes": int(tree.n_nodes),
            "n_features_in_tree": int(tree.n_features_in_tree),
            "interpolation_size": n_interp,
            "vandermonde_condition": cond,
            "convert_s": convert_s,
            "construct_s": construct_s,
            "explain_s": explain_s,
            "prediction": prediction,
            "sum_interactions": total,
            "efficiency_abs_error": abs(total - prediction),
            "efficiency_rel_error": abs(total - prediction)
            / max(abs(prediction), np.finfo(float).tiny),
        }
        rows.append(result)
        print(json.dumps(result, indent=2), flush=True)

    args.out.write_text(
        json.dumps({"model": args.model, "rows": rows}, indent=2) + "\n"
    )
    lines = [
        "| tree | unique depth | interpolation size | cond(V) | nodes | explain s | abs efficiency error | rel efficiency error |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for result in rows:
        lines.append(
            f"| {result['tree']} | {result['unique_feature_depth']} | {result['interpolation_size']} | "
            f"{result['vandermonde_condition']:.3e} | {result['n_nodes']} | {result['explain_s']:.3f} | "
            f"{result['efficiency_abs_error']:.3e} | {result['efficiency_rel_error']:.3e} |"
        )
    args.out_md.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
