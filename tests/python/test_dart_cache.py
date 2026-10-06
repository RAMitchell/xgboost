"""Compare cached DART margins with an independently loaded, uncached model."""

import numpy as np
import pytest

import xgboost as xgb


def run_dart_cached_margins(device, normalization, sample_type, task):
    rng = np.random.default_rng(91)
    x = rng.normal(size=(120, 5)).astype(np.float32)
    if task == "categorical":
        x[:, 0] = rng.integers(0, 5, size=120)
    x[rng.random(x.shape) < 0.15] = np.nan
    y = rng.normal(size=120).astype(np.float32)
    params = dict(tree_method="hist", device=device, nthread=2, max_depth=3,
                  rate_drop=0.15, one_drop=True, skip_drop=0.25,
                  normalize_type=normalization, sample_type=sample_type, seed=17,
                  num_parallel_tree=2, base_score=0.37)
    if task == "multiclass":
        y = rng.integers(0, 3, size=120)
        params.update(objective="multi:softprob", num_class=3)
    elif task == "ranking":
        y = rng.integers(0, 4, size=120)
        params.update(objective="rank:ndcg")
    elif task == "vector":
        y = rng.normal(size=(120, 2)).astype(np.float32)
        params.update(multi_strategy="multi_output_tree", base_score=[0.37, 0.29])
    types = ["c", "q", "q", "q", "q"] if task == "categorical" else None
    dtrain = xgb.DMatrix(x, label=y, feature_types=types)
    dvalid = xgb.DMatrix(x[::-1].copy(), label=y[::-1].copy(), feature_types=types)
    if task == "ranking":
        dtrain.set_group([10] * 12)
        dvalid.set_group([10] * 12)
    model = xgb.Booster(params, [dtrain, dvalid])
    for iteration in range(40):
        model.update(dtrain, iteration)
        reference = xgb.Booster(params)
        reference.load_model(model.save_raw())
        reference.set_param(dict(device=device, nthread=2))
        for data in [dtrain, dvalid]:
            actual = model.predict(data, output_margin=True)
            expected = reference.predict(data, output_margin=True)
            np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
        if iteration == 20:
            # Full -> partial -> full must not reuse a partial-model prediction.
            for limits in [(2, 9), (0, 5), (0, 0)]:
                np.testing.assert_allclose(
                    model.predict(dvalid, output_margin=True, iteration_range=limits),
                    reference.predict(dvalid, output_margin=True, iteration_range=limits),
                    rtol=2e-5, atol=2e-5)
            # Base margins are not tree contributions and must never be rescaled.
            margin = rng.normal(size=expected.shape).astype(np.float32)
            dvalid.set_base_margin(margin)
            np.testing.assert_allclose(model.predict(dvalid, output_margin=True),
                                       reference.predict(dvalid, output_margin=True),
                                       rtol=2e-5, atol=2e-5)


def run_dart_cache_endpoints(device, rate, skip):
    rng = np.random.default_rng(5)
    x = rng.normal(size=(80, 3))
    d = xgb.DMatrix(x, label=x[:, 0] + x[:, 1])
    p = dict(device=device, rate_drop=rate, skip_drop=skip, tree_method="hist", nthread=2,
             max_depth=2, seed=2)
    b = xgb.train(p, d, 50, verbose_eval=False)
    for _ in range(5):
        b.predict(d, training=True)
        fresh = xgb.Booster(model_file=b.save_raw())
        np.testing.assert_allclose(b.predict(d), fresh.predict(d), rtol=2e-5, atol=2e-5)
    resumed = xgb.train(p, d, 10, xgb_model=b, verbose_eval=False)
    fresh = xgb.Booster(model_file=resumed.save_raw())
    np.testing.assert_allclose(resumed.predict(d), fresh.predict(d), rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("normalization", ["tree", "forest"])
@pytest.mark.parametrize("sample_type", ["uniform", "weighted"])
@pytest.mark.parametrize("task", ["regression", "multiclass", "ranking", "vector", "categorical"])
def test_dart_cached_margins(normalization, sample_type, task):
    run_dart_cached_margins("cpu", normalization, sample_type, task)


@pytest.mark.parametrize("rate,skip", [(0., 0.), (0., 1.), (1., 0.), (.8, .5)])
def test_dart_cache_endpoints(rate, skip):
    run_dart_cache_endpoints("cpu", rate, skip)
