"""Accuracy checks for intercepts obtained by a bracketed root solve."""

import numpy as np
import pytest
from scipy.optimize import brentq

import xgboost as xgb
from xgboost.testing.updater import get_basescore


@pytest.mark.parametrize("objective", ["reg:expectileerror", "reg:pseudohubererror"])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("scale", [1e-6, 1.0, 1e6])
@pytest.mark.parametrize("family", ["skewed", "outliers", "constant"])
def test_iterative_intercept(
    objective: str, weighted: bool, scale: float, family: str
) -> None:
    rng = np.random.default_rng(315)
    y = rng.lognormal(0, 2, size=1024)
    if family == "outliers":
        y = rng.normal(size=y.size)
        y[0] = 1000
    elif family == "constant":
        y[:] = 7
    y = (scale * y).astype(np.float32)
    w = rng.lognormal(0, 2, size=y.size).astype(np.float32) if weighted else None
    weights = np.ones(y.size) if w is None else w.astype(np.float64)
    labels = y.astype(np.float64)
    params = {
        "objective": objective,
        "tree_method": "hist",
        "max_depth": 1,
        "nthread": 2,
        "eta": 0,
    }
    if objective == "reg:expectileerror":
        parameters = np.array([0.001, 0.1, 0.5, 0.9, 0.999], dtype=np.float32)
        params["expectile_alpha"] = parameters.tolist()
    else:
        parameters = [np.float32(scale)]
        params["huber_slope"] = float(parameters[0])
    data = xgb.DMatrix(np.zeros((y.size, 1)), label=y, weight=w)
    model = xgb.train(params, data, num_boost_round=1)
    actual = np.asarray(get_basescore(model))
    for j, parameter in enumerate(parameters):

        def gradient(value: float) -> float:
            r = value - labels
            if objective == "reg:expectileerror":
                return float(
                    np.dot(
                        weights
                        * np.where(r >= 0, 1 - float(parameter), float(parameter)),
                        r,
                    )
                )
            return float(np.dot(weights, r / np.hypot(1, r / float(parameter))))

        expected = (
            labels[0]
            if family == "constant"
            else brentq(gradient, labels.min(), labels.max(), xtol=1e-13 * scale)
        )
        mean = np.average(labels, weights=weights)
        std = np.sqrt(np.average((labels - mean) ** 2, weights=weights))
        tol = max(
            1e-5 * std,
            np.finfo(np.float32).eps * abs(mean),
            np.finfo(np.float32).smallest_subnormal,
        )
        # Include rounding the solved intercept into the float32 model.
        assert abs(actual[j] - expected) <= tol + np.finfo(np.float32).eps * abs(
            expected
        )
