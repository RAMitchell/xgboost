"""Check the submission implementation against an independent subset SHAP oracle."""

import itertools
import json
import math
from pathlib import Path

import numpy as np
import pytest
import xgboost as xgb

pytest.importorskip("numba")
from paper import numba_quadrature as _module

QuadratureSHAPNumba = _module.QuadratureSHAPNumba


def subset_oracle(booster, rows):
    """Enumerate feature coalitions; use no quadrature or path-basis recurrences."""
    trees = [
        json.loads(t) for t in booster.get_dump(dump_format="json", with_stats=True)
    ]
    learner = json.loads(booster.save_raw(raw_format="json"))["learner"]
    groups = learner["gradient_booster"]["model"]["tree_info"]
    active = set()

    def collect(node):
        if "leaf" in node:
            return
        active.add(int(node["split"][1:]))
        for child in node["children"]:
            collect(child)

    for tree in trees:
        collect(tree)
    active = sorted(active)
    assert len(active) <= 8, "Subset enumeration is only for small active feature sets"

    def expectation(node, coalition, row):
        if "leaf" in node:
            return float(np.float32(node["leaf"]))
        feature = int(node["split"][1:])
        if feature in coalition:
            target = (
                node["missing"]
                if np.isnan(row[feature])
                else (
                    node["yes"]
                    if row[feature] < float(np.float32(node["split_condition"]))
                    else node["no"]
                )
            )
            child = next(c for c in node["children"] if c["nodeid"] == target)
            return expectation(child, coalition, row)
        return sum(
            float(np.float32(child["cover"]))
            / float(np.float32(node["cover"]))
            * expectation(child, coalition, row)
            for child in node["children"]
        )

    output = np.zeros((len(rows), max(groups, default=0) + 1, rows.shape[1]))
    for r, row in enumerate(rows):
        for tree, group in zip(trees, groups):
            for feature in active:
                others = [f for f in active if f != feature]
                for size in range(len(active)):
                    weight = 1 / (len(active) * math.comb(len(active) - 1, size))
                    for subset in itertools.combinations(others, size):
                        coalition = set(subset)
                        before = expectation(tree, coalition, row)
                        after = expectation(tree, coalition | {feature}, row)
                        output[r, group, feature] += weight * (after - before)
    return output


@pytest.mark.parametrize("objective", ["reg:squarederror", "multi:softprob"])
@pytest.mark.parametrize("nfeatures", [4, 54])
def test_independent_shap(objective, nfeatures):
    rng = np.random.default_rng(847)
    small = rng.normal(size=(120, 4)).astype(np.float32)
    x = np.zeros((len(small), nfeatures), dtype=np.float32)
    x[:, [0, 1, 2, 3] if nfeatures == 4 else [0, 12, 15, 45]] = small
    y = small[:, 0] ** 2 + small[:, 0] * small[:, 1] + small[:, 3]
    params = {
        "objective": objective,
        "max_depth": 7,
        "min_child_weight": 0,
        "nthread": 1,
        "base_score": 0.5,
        "seed": 432,
    }
    if objective.startswith("multi:"):
        params["num_class"] = 3
        y = np.digitize(y, [-0.2, 0.5])
    booster = xgb.train(params, xgb.DMatrix(x, label=y), 2)
    rows = x[:9].astype(np.float64)
    split = json.loads(booster.get_dump(dump_format="json")[0])
    feature = int(split["split"][1:])
    rows[0, feature] = float(np.float32(split["split_condition"]))
    rows[1, feature] = np.nan
    expected = subset_oracle(booster, rows)
    for fixed_eight in [False, True]:
        model = QuadratureSHAPNumba(booster, fixed_eight=fixed_eight)
        backing = np.empty((len(rows), nfeatures * 2))
        backing[:, ::2] = rows
        for matrix in [
            np.array(rows, order="C"),
            np.array(rows, order="F"),
            backing[:, ::2],
        ]:
            for size in [1, 3, 7, 9]:
                value = np.concatenate(
                    [
                        model.explain(matrix[i : i + size])
                        for i in range(0, len(matrix), size)
                    ]
                )
                assert np.isfinite(value).all()
                np.testing.assert_allclose(value, expected, rtol=1e-10, atol=1e-12)
        with pytest.raises(ValueError, match="Expected a matrix"):
            model.explain(rows[:, :-1])
        assert model.explain(rows[:0]).shape == (0, expected.shape[1], nfeatures)


def test_constant_ensemble():
    x = np.zeros((8, 3), dtype=np.float32)
    booster = xgb.train(
        {"nthread": 1, "base_score": 0.5}, xgb.DMatrix(x, label=np.ones(8)), 2
    )
    for fixed in [False, True]:
        value = QuadratureSHAPNumba(booster, fixed_eight=fixed).explain(x)
        np.testing.assert_array_equal(value, np.zeros((8, 1, 3)))


def test_reject_categorical():
    x = np.tile(np.arange(4, dtype=np.float32), 16).reshape(-1, 1)
    booster = xgb.train(
        {"nthread": 1, "max_depth": 2, "max_cat_to_onehot": 1},
        xgb.DMatrix(
            x, label=(x[:, 0] < 2), feature_types=["c"], enable_categorical=True
        ),
        2,
    )
    with pytest.raises(ValueError, match="Categorical splits"):
        QuadratureSHAPNumba(booster)


def test_standard_rule_and_extraction_cases():
    # Check the fixed rule's polynomial exactness, independently of tree SHAP.
    nodes = np.asarray(_module._NODES_8)
    weights = np.asarray(_module._WEIGHTS_8)
    for degree in range(16):
        np.testing.assert_allclose(
            np.dot(weights, nodes**degree), 1 / (degree + 1), rtol=0, atol=5e-16
        )
    workspace = _module._Workspace(np.zeros((2, 1)), 1, 1, 8)
    workspace.n = 8
    workspace.nodes = nodes
    workspace.weights = np.concatenate((weights, -1 / (1 - nodes)))
    rng = np.random.default_rng(52)
    workspace.h[:] = rng.normal(size=len(workspace.h))
    h = workspace.h[3:11].copy()
    for p, up in itertools.product(
        [0.0, 1.0, np.nextafter(1.0, 0.0), 2.0, 19.0], repeat=2
    ):
        delta = (p - 1) / (1 + (p - 1) * nodes) - (up - 1) / (1 + (up - 1) * nodes)
        expected = np.dot(weights * h, delta)
        for extract in [_module._extract8, _module._extract_variable]:
            np.testing.assert_allclose(
                extract(workspace, 3, p, up), expected, rtol=2e-13, atol=2e-14
            )
