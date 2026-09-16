"""Distributed iterative intercepts must match a solve on the combined data."""

import dask.array as da
import numpy as np
import pytest
from distributed import Client

import xgboost as xgb
from xgboost import dask as dxgb
from xgboost.testing.updater import get_basescore


@pytest.mark.parametrize("client_kwargs", [{"threads_per_worker": 2}], indirect=True)
@pytest.mark.parametrize("objective", ["reg:expectileerror", "reg:pseudohubererror"])
@pytest.mark.parametrize("weighted", [False, True])
def test_iterative_intercept(client: Client, objective: str, weighted: bool) -> None:
    rng = np.random.default_rng(617)
    n = 257
    X = rng.normal(size=(n, 2))
    y = rng.lognormal(0, 2, n).astype(np.float32)
    weights = rng.lognormal(0, 1, n).astype(np.float32) if weighted else None
    params = {
        "objective": objective,
        "tree_method": "hist",
        "max_depth": 1,
        "nthread": 2,
    }
    if objective == "reg:expectileerror":
        params["expectile_alpha"] = [0.001, 0.5, 0.999]
    else:
        # Independent response columns must retain their own global root.
        y = np.column_stack([y, 10 - 2 * y])
        params["huber_slope"] = 0.1

    local = xgb.train(params, xgb.DMatrix(X, y, weight=weights), num_boost_round=1)
    chunks = (17, 64, 176)
    dX = da.from_array(X, chunks=(chunks, 2))
    dy = da.from_array(y, chunks=chunks if y.ndim == 1 else (chunks, 2))
    dw = None if weights is None else da.from_array(weights, chunks=chunks)
    data = dxgb.DaskDMatrix(client, dX, dy, weight=dw)
    distributed = dxgb.train(client, params, data, num_boost_round=1)["booster"]
    np.testing.assert_allclose(
        get_basescore(distributed), get_basescore(local), rtol=1e-5, atol=1e-5
    )
