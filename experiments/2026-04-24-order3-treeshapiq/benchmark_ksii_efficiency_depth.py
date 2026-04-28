"""Benchmark TreeSHAP-IQ k-SII efficiency error as model depth changes."""
# pylint: disable=missing-function-docstring,too-many-locals,too-many-arguments,broad-exception-caught,line-too-long

import argparse
import importlib.util
import json
import multiprocessing as mp
import re
import sys
import time
from pathlib import Path

import numpy as np
import xgboost as xgb
from shapiq.explainer.tree import TreeExplainer

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "demo/guide-python/quadratureshap_rapids_benchmark.py"
ORDER3 = Path(__file__).with_name("benchmark_order3.py")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def tree_stats(model: xgb.Booster) -> dict[str, float]:
    dump = model.get_dump(dump_format="json", with_stats=True)

    def walk(node: dict, depth: int = 0, path_features: frozenset[str] = frozenset()):
        children = node.get("children", [])
        if not children:
            return depth, len(path_features), 1, 1
        child_features = path_features | {node["split"]}
        max_depth = depth
        max_unique = len(child_features)
        nodes = 1
        leaves = 0
        for child in children:
            child_depth, child_unique, child_nodes, child_leaves = walk(
                child, depth + 1, child_features
            )
            max_depth = max(max_depth, child_depth)
            max_unique = max(max_unique, child_unique)
            nodes += child_nodes
            leaves += child_leaves
        return max_depth, max_unique, nodes, leaves

    depths = []
    unique_depths = []
    node_counts = []
    leaf_counts = []
    for tree_json in dump:
        tree_json = re.sub(r"\bnan\b", "0", tree_json)
        tree_json = re.sub(r"\binf\b", "0", tree_json)
        depth, unique_depth, nodes, leaves = walk(json.loads(tree_json))
        depths.append(depth)
        unique_depths.append(unique_depth)
        node_counts.append(nodes)
        leaf_counts.append(leaves)

    return {
        "num_trees": len(dump),
        "mean_max_depth": float(np.mean(depths)),
        "max_max_depth": int(np.max(depths)),
        "mean_unique_feature_depth": float(np.mean(unique_depths)),
        "max_unique_feature_depth": int(np.max(unique_depths)),
        "mean_nodes": float(np.mean(node_counts)),
        "mean_leaves": float(np.mean(leaf_counts)),
    }


def train_cal_housing(
    depth: int, rounds: int, seed: int
) -> tuple[xgb.Booster, np.ndarray, np.ndarray]:
    qbench = load_module("qbench_depth", BENCH)
    dataset = next(ds for ds in qbench.get_test_datasets() if ds.name == "cal_housing")
    dtrain = xgb.QuantileDMatrix(dataset.X, dataset.y)
    params = {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "device": "cpu",
        "eta": 0.05,
        "max_depth": depth,
        "seed": seed,
        "nthread": 35,
    }
    booster = xgb.train(params, dtrain, num_boost_round=rounds, verbose_eval=False)
    x_test = dataset.test_input(1, seed)
    x_np = (
        x_test.to_numpy(dtype=np.float64)
        if hasattr(x_test, "to_numpy")
        else np.asarray(x_test, dtype=np.float64)
    )
    return booster, np.atleast_2d(x_np), np.asarray(dataset.y)


def efficiency_error(booster: xgb.Booster, row: np.ndarray, order: int):
    order3 = load_module("order3_depth", ORDER3)
    trees = order3.convert_for_shapiq(booster)
    t0 = time.perf_counter()
    explainer = TreeExplainer(trees, max_order=order, min_order=0, index="k-SII")
    construct_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    values = explainer.explain(row)
    explain_s = time.perf_counter() - t0

    raw_margin = float(
        np.sum(booster.predict(xgb.DMatrix(row.reshape(1, -1)), output_margin=True))
    )
    tree_margin = raw_margin - order3.base_score(booster)
    err = abs(float(sum(values.dict_values.values())) - tree_margin)
    return construct_s, explain_s, err


def worker(queue, depth: int, rounds: int, seed: int, order: int):
    try:
        booster, rows, _ = train_cal_housing(depth, rounds, seed)
        stats = tree_stats(booster)
        construct_s, explain_s, err = efficiency_error(booster, rows[0], order)
        queue.put(
            {
                "ok": True,
                "result": {
                    "requested_depth": depth,
                    "rounds": rounds,
                    "order": order,
                    **stats,
                    "treeshapiq_construct_s": construct_s,
                    "treeshapiq_explain_s": explain_s,
                    "treeshapiq_efficiency_error": err,
                },
            }
        )
    except Exception as exc:  # noqa: BLE001
        queue.put({"ok": False, "error": repr(exc), "requested_depth": depth})


def run_depth(depth: int, args):
    queue = mp.Queue()
    proc = mp.Process(
        target=worker, args=(queue, depth, args.rounds, args.seed, args.order)
    )
    proc.start()
    proc.join(args.timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        return {"requested_depth": depth, "error": f"timeout after {args.timeout}s"}
    if queue.empty():
        return {
            "requested_depth": depth,
            "error": f"worker exited with code {proc.exitcode}",
        }
    payload = queue.get()
    if not payload["ok"]:
        return {"requested_depth": depth, "error": payload["error"]}
    return payload["result"]


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "| requested depth | max depth | max unique depth | trees | construct s | explain s | efficiency error |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        if "error" in row:
            lines.append(
                f"| {row['requested_depth']} | DNF | DNF | DNF | DNF | DNF | {row['error']} |"
            )
        else:
            lines.append(
                f"| {row['requested_depth']} | {row['max_max_depth']} | "
                f"{row['max_unique_feature_depth']} | {row['num_trees']} | "
                f"{row['treeshapiq_construct_s']:.3f} | {row['treeshapiq_explain_s']:.3f} | "
                f"{row['treeshapiq_efficiency_error']:.6e} |"
            )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--depths", type=int, nargs="+", default=[2, 4, 6, 8, 10, 12])
    parser.add_argument("--rounds", type=int, default=25)
    parser.add_argument("--order", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260424)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).with_name("ksii-efficiency-depth.json"),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path(__file__).with_name("ksii-efficiency-depth.md"),
    )
    args = parser.parse_args()

    rows = []
    for depth in args.depths:
        print(f"Running depth {depth}", flush=True)
        row = run_depth(depth, args)
        rows.append(row)
        print(json.dumps(row, indent=2), flush=True)
        args.out.write_text(json.dumps({"rows": rows}, indent=2) + "\n")
        write_summary(args.out_md, rows)


if __name__ == "__main__":
    main()
