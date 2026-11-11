"""Standard-rule third-order explainers with identical XGBoost model semantics."""
# pylint: disable=duplicate-code
# pylint: disable=consider-using-enumerate,missing-function-docstring

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "experiments/2026-04-24-order3-treeshapiq/benchmark_order3.py"
CACHE = ROOT / "experiments/2026-04-10-rapids-style-shap-benchmark/model-cache"


class OrderedFeatureSet(set):
    """Keep shapiq's local feature indexing and sorted interaction labels aligned."""

    def __iter__(self):
        return iter(sorted(set.__iter__(self)))


def load_old():
    spec = importlib.util.spec_from_file_location("order3_original", OLD)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def canonical_trees(old, booster):
    trees = old.parse_trees(booster)
    if old.has_categorical_splits(trees):
        raise ValueError("Categorical models are unsupported by TreeSHAP-IQ")
    for tree in trees:
        # The JSON/dataframe dump prints rounded decimals of float32 model values.
        for key in ["threshold", "cover", "value"]:
            setattr(tree, key, getattr(tree, key).astype(np.float32).astype(np.float64))
        for parent in range(len(tree.feature)):
            if tree.feature[parent] >= 0:
                for child in [tree.left[parent], tree.right[parent]]:
                    assert tree.cover[parent] > 0 and tree.cover[child] > 0
                    tree.child_weight[child] = tree.cover[child] / tree.cover[parent]
    return [tree for tree in trees if tree.feature[0] >= 0]


def make_explainer(old, trees, n_features, method):
    if method == "quadrature":
        return old.OrderQuadrature(trees, n_features, 3, 8)
    from shapiq.explainer.tree.base import TreeModel

    converted = []
    for t in trees:
        # shapiq tests <=; XGBoost and the quadrature implementation test <.
        thresholds = np.nextafter(t.threshold, -np.inf)
        converted.append(
            TreeModel(
                children_left=t.left.copy(),
                children_right=t.right.copy(),
                features=t.feature.copy(),
                thresholds=thresholds,
                values=t.value.copy(),
                node_sample_weight=t.cover.copy(),
                original_output_type="raw",
                feature_ids=OrderedFeatureSet(int(f) for f in t.feature if f >= 0),
            )
        )
    return old.TreeExplainer(converted, max_order=3, min_order=3, index="SII")


def explain(old, explainer, row, method):
    values = explainer.explain(row)
    return (
        values
        if method == "quadrature"
        else old.dict_from_interaction_values(values, 3)
    )


def pack(values):
    keys = sorted(values)
    return np.asarray(keys, dtype=np.int32).reshape(-1, 3), np.asarray(
        [values[k] for k in keys], dtype=np.float64
    )
