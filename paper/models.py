"""Numeric XGBoost model conversion shared by the Python experiments."""

# Numerical state and orchestration are kept explicit for reproduction.
# pylint: disable=too-many-instance-attributes

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .native import get_test_datasets

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODELS = [
    f"{dataset}-{setting}"
    for setting in ["small", "sparse", "large"]
    for dataset in ["adult", "cal_housing", "covtype", "fashion_mnist"]
]


@dataclass
class PyTree:
    """Numeric tree arrays with original feature IDs and missing directions."""

    left: np.ndarray
    right: np.ndarray
    missing: np.ndarray
    feature: np.ndarray
    threshold: np.ndarray
    categories: list[set[int] | None]
    value: np.ndarray
    cover: np.ndarray
    child_weight: np.ndarray


def load_treegrad(treegrad_root: Path):
    """Import the authors’ reference from its pinned checkout."""
    sys.path.insert(0, str(treegrad_root))
    from TreeGrad import treestab  # pylint: disable=import-outside-toplevel

    return treestab


def max_depth(tree: PyTree) -> int:
    """Return the maximum raw split depth."""

    def walk(node: int, depth: int) -> int:
        if tree.feature[node] < 0:
            return depth
        return max(
            walk(int(tree.left[node]), depth + 1),
            walk(int(tree.right[node]), depth + 1),
        )

    return walk(0, 0)


def xgboost_to_treegrad_model(trees: list[PyTree]):
    """Adapt strict numeric splits to the authors’ reference model."""
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


def parse_trees(booster):
    """Read model arrays directly; do not round-trip through a dataframe dump."""
    model = json.loads(booster.save_raw(raw_format="json"))["learner"][
        "gradient_booster"
    ]["model"]
    trees = []
    for data in model["trees"]:
        if any(data["split_type"]):
            raise ValueError(
                "Expand categorical features before the Python experiments"
            )
        left = np.asarray(data["left_children"], dtype=np.int32)
        right = np.asarray(data["right_children"], dtype=np.int32)
        features = np.where(left < 0, -2, data["split_indices"]).astype(np.int32)
        values = np.asarray(data["split_conditions"], dtype=np.float32).astype(
            np.float64
        )
        cover = np.asarray(data["sum_hessian"], dtype=np.float32).astype(np.float64)
        weights = np.ones(len(left))
        for parent in np.flatnonzero(left >= 0):
            for child in (left[parent], right[parent]):
                if not (cover[parent] > 0 and cover[child] > 0):
                    raise ValueError("Tree children must have positive cover")
                weights[child] = cover[child] / cover[parent]
        missing = np.where(data["default_left"], left, right).astype(np.int32)
        trees.append(
            PyTree(
                left,
                right,
                missing,
                features,
                values,
                [None] * len(left),
                values,
                cover,
                weights,
            )
        )
    return trees, model["tree_info"]


def counts(trees, nfeatures):
    """Return the exactness-sufficient per-tree point counts used in A–D."""
    return [max(1, (min(max_depth(t), nfeatures) + 1) // 2) for t in trees]


def input_rows(model, rows, seed=432):
    """Select the canonical seeded explained rows for a benchmark."""
    name = model.rsplit("-", 1)[0]
    dataset = next(d for d in get_test_datasets() if d.name == name)
    return np.asarray(dataset.test_input(rows, seed), dtype=np.float32).astype(
        np.float64
    )
