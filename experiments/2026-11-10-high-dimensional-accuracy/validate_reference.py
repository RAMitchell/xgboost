"""Validate integration against coalition enumeration, including repeated splits."""
# pylint: disable=duplicate-code

import numpy as np
import pandas as pd
import xgboost as xgb
from numba import set_num_threads
from reference import exhaustive_reference, flatten, predict_reference, rule

set_num_threads(4)
rng = np.random.default_rng(83)
x = pd.DataFrame(rng.normal(size=(256, 3)), columns=["a", "b", "c"])
x["category"] = pd.Categorical(rng.integers(0, 4, len(x)))
x.loc[::11, "a"] = np.nan
raw = x.copy()
raw["category"] = raw["category"].cat.codes
raw = raw.to_numpy(dtype=np.float32)
y = (
    x["a"].fillna(0).to_numpy()
    + x["b"].to_numpy() ** 2
    + (x["category"].cat.codes == 2) * 2
    + rng.normal(0, 0.1, len(x))
)
for objective in ["reg:squarederror", "binary:logistic", "multi:softprob"]:
    labels = y if objective == "reg:squarederror" else (y > np.median(y)).astype(int)
    params = {
        "objective": objective,
        "max_depth": 7,
        "min_child_weight": 0,
        "reg_lambda": 0,
        "nthread": 4,
        "base_score": 0.5,
        "eta": 0.2,
    }
    if objective == "multi:softprob":
        params["num_class"] = 2
    data = xgb.DMatrix(x, label=labels, enable_categorical=True, nthread=4)
    model = xgb.train(params, data, 3)
    arrays, base, stats = flatten(model)
    assert stats["max_depth"] > stats["max_unique_depth"], stats
    assert any(mask != np.uint64((1 << 63) - 1) for mask in arrays[5]), (
        "categorical test must split categories"
    )
    sample = raw[:6]
    exact = predict_reference(sample, arrays, base, 16)
    brute = exhaustive_reference(sample, arrays, base)
    assert np.max(np.abs(exact - brute)) < 1e-12, np.max(np.abs(exact - brute))
    higher = predict_reference(sample, arrays, base, 32)
    assert np.max(np.abs(exact - higher)) < 1e-12
    test = xgb.DMatrix(x.iloc[:6], enable_categorical=True, nthread=4)
    model.set_param({"device": "cpu", "shap_algorithm": "treeshap"})
    incumbent = model.predict(test, pred_contribs=True, strict_shape=True)
    assert np.max(np.abs(incumbent - exact)) < 2e-6, np.max(np.abs(incumbent - exact))
    margin = model.predict(test, output_margin=True, strict_shape=True)
    assert np.max(np.abs(exact.sum(axis=-1) - margin)) < 2e-6
    print(objective, stats, "oracle max error", float(np.max(np.abs(exact - brute))))
# Standard eight-node rule integrates degree 15; transformed rule loses that guarantee.
for degree in range(16):
    nodes, weights = rule(8)
    assert abs(np.sum(weights * nodes**degree) - 1 / (degree + 1)) < 1e-14
nodes, weights = rule(8, True)
assert abs(np.sum(weights * nodes**15) - 1 / 16) > 1e-8
print(
    "Reference, categorical/missing/repeated features, multiclass and quadrature checks passed"
)
# Known high-dimensional conjunction: all observed features are symmetric, so each
# attribution equals (1 - p**d)/d. Zero-valued sibling leaves add no contribution.
for d in [16, 32, 50]:
    arrays = (
        np.array([0, d], np.int64),
        np.arange(d, dtype=np.int64),
        np.full(d, 0.5),
        np.full(d, -np.inf),
        np.full(d, 0.5),
        np.full(d, (1 << 63) - 1, np.uint64),
        np.zeros(d, np.bool_),
        np.array([1.0]),
        np.array([0], np.int64),
    )
    value = predict_reference(
        np.zeros((1, d), np.float32), arrays, np.array([0.0]), max(16, (d + 1) // 2)
    )
    expected = (1 - 0.5**d) / d
    assert np.max(np.abs(value[0, 0, :-1] - expected)) < 1e-14
    assert abs(value[0, 0, -1] - 0.5**d) < 1e-20
    print("Analytic conjunction distinct-depth", d, "passed")
