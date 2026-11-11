"""Check precision, point selection, group preservation, and independent attributions."""
# pylint: disable=duplicate-code
# pylint: disable=consider-using-enumerate,wrong-import-position

import importlib
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb
from benchmark_matched import NodesProxy, load_old, make_objects
from numba import set_num_threads
from quadrature import Quadrature

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "2026-11-10-high-dimensional-accuracy")
)
from reference import flatten, predict_reference

set_num_threads(1)
old = load_old()
tg = old.load_treegrad(Path(__file__).resolve().parent / "dependencies/TreeGrad")
tgmodule = importlib.import_module("TreeGrad.treegrad_shap")
rng = np.random.default_rng(921)
x = rng.normal(size=(256, 4)).astype(np.float32)
y = x[:, 0] + x[:, 1] ** 2 + 0.2 * rng.normal(size=256)
for objective in ["reg:squarederror", "multi:softprob"]:
    labels = (
        y
        if objective == "reg:squarederror"
        else np.digitize(y, [-0.1, 0.8]).astype(int)
    )
    params = {
        "objective": objective,
        "max_depth": 7,
        "min_child_weight": 0,
        "reg_lambda": 0.1,
        "nthread": 4,
        "base_score": 0.5,
    }
    if objective == "multi:softprob":
        params["num_class"] = 3
    b = xgb.train(params, xgb.DMatrix(x, label=labels, nthread=4), 3)
    trees = old.parse_trees(b)
    for t in trees:
        for key in ["threshold", "cover", "value"]:
            setattr(t, key, getattr(t, key).astype(np.float32).astype(np.float64))
        for node in range(len(t.feature)):
            if t.feature[node] >= 0:
                for child in [t.left[node], t.right[node]]:
                    t.child_weight[child] = t.cover[child] / t.cover[node]
    learner = __import__("json").loads(b.save_raw(raw_format="json"))["learner"]
    info = learner["gradient_booster"]["model"]["tree_info"]
    groups = max(1, int(learner["learner_model_param"]["num_class"]))
    q, nm, dm, native, distinct, ds, en, ed = make_objects(old, trees, info, 4, groups)
    assert any(d > u for d, u in ds), "test must contain repeated features"
    rows = x[:5].copy()
    rows[0, trees[0].feature[0]] = trees[0].threshold[0]
    arrays, base, _ = flatten(b)
    ref = predict_reference(rows, arrays, base, 16)[:, :, :-1]
    for label, objs, expected in [("native", nm, en), ("distinct", dm, ed)]:
        for cached in [False, True]:
            proxy = NodesProxy(
                {n: np.polynomial.legendre.leggauss(n) for n in set(expected)}
                if cached
                else None,
                audit=True,
            )
            tgmodule.np = proxy
            result = np.array(
                [
                    [
                        tg(obj, row, (1, 1), class_index=None)
                        if obj.estimators_.size
                        else np.zeros(len(row))
                        for obj in objs
                    ]
                    for row in rows
                ]
            )
            assert proxy.calls == expected * len(rows)
            np.testing.assert_allclose(result, ref, rtol=1e-10, atol=1e-12)
    tgmodule.np = np
    for name, objs in q.items():
        result = np.array([[obj.explain(row) for obj in objs] for row in rows])
        assert result.dtype == np.float64
        np.testing.assert_allclose(result, ref, rtol=1e-10, atol=1e-12)
    # Arbitrary quadrature counts retain the same code path.
    for n in [3, 7, 8, 16, 26]:
        result = np.array(
            [
                Quadrature(
                    [t for t, g in zip(trees, info) if g == group],
                    4,
                    [n] * sum(g == group for g in info),
                ).explain(rows[0])
                for group in range(groups)
            ]
        )
        np.testing.assert_allclose(result, ref[0], rtol=1e-10, atol=1e-12)
    print(objective, "all matched/native/cached/variable-count checks passed")
# Native and cached node providers must give independently mutable rule arrays.
cache = {3: np.polynomial.legendre.leggauss(3)}
proxy = NodesProxy(cache)
assert "zeros" in proxy.__dict__ and proxy.zeros is np.zeros
assert "dot" in proxy.__dict__ and proxy.dot is np.dot
a, w = proxy.leggauss(3)
a += 100
w *= 100
c, d = proxy.leggauss(3)
np.testing.assert_array_equal(c, cache[3][0])
np.testing.assert_array_equal(d, cache[3][1])
print("All algorithm, threshold-equality, float64 and node-cache checks passed")
