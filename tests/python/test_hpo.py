"""Behavioral and numerical checks for the experimental ask/tell optimizer."""

import hashlib
import json

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm, rankdata

import xgboost as xgb
from xgboost import hpo


def example_prior(n=12):
    x = np.linspace(0, 1, n)
    k = np.exp(-(((x[:, None] - x[None, :]) / 0.3) ** 2)) + 1e-8 * np.eye(n)
    return hpo.Prior(np.sin(x * 6) * 0.3, k, 0.05)


def optimizer(n=12, **kwargs):
    return hpo.Optimizer([{"x": i} for i in range(n)], prior=example_prior(n), **kwargs)


def test_sequential_ownership_and_exhaustion():
    opt = optimizer(3)
    first = opt.ask()
    assert opt.ask() is first
    first.params["x"] = 999  # Copy, not an editable internal configuration.
    assert opt.ask().params["x"] != 999
    with pytest.raises(ValueError, match="pending"):
        opt.tell(optimizer(3).ask(), 1.0)
    for bad in [None, np.inf, np.nan, "1", [1], True, np.bool_(True), 1j]:
        with pytest.raises(ValueError):
            opt.tell(first, bad)
        assert opt.ask() is first
    opt.tell(first, 2.0)
    with pytest.raises(ValueError):
        opt.tell(first, 2.0)
    opt.tell(opt.ask(), 1.0)
    opt.tell(opt.ask(), status="failed", message="timeout")
    assert len(opt.history) == 3
    assert opt.best_trial == opt.history[1].trial
    assert opt.history[-1].loss is None
    with pytest.raises(StopIteration):
        opt.ask()


def test_failure_does_not_become_a_loss():
    opt = optimizer(4)
    for _ in range(4):
        trial = opt.ask()
        with pytest.raises(ValueError):
            opt.tell(trial, 100.0, status="failed")
        opt.tell(trial, status="failed")
    assert opt.best_trial is None
    assert all(o.loss is None for o in opt.history)
    with pytest.raises(StopIteration):
        opt.ask()


@pytest.mark.parametrize("pending", [True, False])
@pytest.mark.parametrize("direction", ["minimize", "maximize"])
def test_resume_with_failures_and_pending(tmp_path, pending, direction):
    opt = optimizer(random_state=9, direction=direction)
    for step in range(5):
        trial = opt.ask()
        if step in [0, 3]:
            opt.tell(trial, status="failed", message="explicit failure")
        else:
            opt.tell(trial, float(trial.params["x"] ** 2))
    if pending:
        opt.ask()
    path = tmp_path / "optimizer.json"
    opt.save(path)
    resumed = hpo.Optimizer.load(path)
    assert resumed.history == opt.history
    for _ in range(7):
        a, b = opt.ask(), resumed.ask()
        assert a == b
        opt.tell(a, float(a.candidate_id))
        resumed.tell(b, float(b.candidate_id))
    assert opt.best_trial == resumed.best_trial
    state = json.loads(path.read_text())
    state["observations"][0]["candidate_id"] = -1
    path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="replay mismatch"):
        hpo.Optimizer.load(path)


def test_monotone_metric_invariance_and_maximize():
    opts = [
        optimizer(random_state=5),
        optimizer(random_state=5),
        optimizer(random_state=5, direction="maximize"),
    ]
    for _ in range(12):
        trials = [o.ask() for o in opts]
        assert len({t.candidate_id for t in trials}) == 1
        loss = trials[0].candidate_id % 4  # Include ties.
        for o, t, y in zip(opts, trials, [loss, np.exp(loss), -loss]):
            o.tell(t, y)
    assert len({o.best_trial.candidate_id for o in opts}) == 1


def test_normal_scores_and_all_ties():
    values = np.array([1.0, 1.0, 3.0, 2.0])
    expected = norm.ppf((rankdata(values) - 0.5) / len(values))
    np.testing.assert_allclose(hpo._normal_rank_scores(values), expected)
    opt = optimizer(8)
    for _ in range(8):
        opt.tell(opt.ask(), 1.0)
    assert len({o.trial.candidate_id for o in opt.history}) == 8


@pytest.mark.parametrize("z", [-100.0, -50.0, -20.0, -5.0, -1.0, 0.0, 2.0, 10.0])
def test_log_ei_against_independent_quadrature(z):
    actual = hpo._log_ei(np.array([-z]), np.array([1.0]), 0.0)[0]
    if z < 0:
        a = -z
        integral = quad(lambda u: u * np.exp(-u - 0.5 * (u / a) ** 2), 0, np.inf)[0]
        expected = (
            -0.5 * a * a - 0.5 * np.log(2 * np.pi) - 2 * np.log(a) + np.log(integral)
        )
    else:
        expected = np.log(norm.pdf(z) + z * norm.cdf(z))
    np.testing.assert_allclose(actual, expected, atol=2e-10, rtol=1e-12)
    assert hpo._log_ei(np.array([1.0]), np.array([0.0]), 0.0)[0] == -np.inf
    assert hpo._log_ei(np.array([-2.0]), np.array([0.0]), 0.0)[0] == np.log(2.0)


def test_gp_difference_variance_against_conditioning():
    prior = example_prior()
    opt = optimizer()
    for y in [2.0, 1.0, 3.0]:
        opt.tell(opt.ask(), y)
    observations = list(opt.history)
    ids = np.array([o.trial.candidate_id for o in observations])
    available = np.setdiff1d(np.arange(12), ids)
    scores = norm.ppf((rankdata([2.0, 1.0, 3.0]) - 0.5) / 3)
    k = prior.covariance
    covariance = k[np.ix_(ids, ids)] + np.diag(prior.noise_variance[ids])
    posterior_mean = prior.mean + k[:, ids] @ np.linalg.solve(
        covariance, scores - prior.mean[ids]
    )
    posterior_cov = k - k[:, ids] @ np.linalg.solve(covariance, k[ids, :])
    best = ids[1]
    difference_var = (
        posterior_cov[best, best] + np.diag(posterior_cov) - 2 * posterior_cov[:, best]
    )
    expected = hpo._log_ei(
        posterior_mean[available] - posterior_mean[best],
        np.sqrt(difference_var[available]),
        0.0,
    )
    np.testing.assert_allclose(
        opt._scores(available, observations), expected, rtol=1e-9, atol=1e-9
    )


def params():
    return dict(
        learning_rate=0.1,
        max_depth=4,
        max_leaves=0,
        grow_policy="depthwise",
        min_child_weight=1.0,
        subsample=0.8,
        colsample_bytree=0.8,
        colsample_bylevel=1.0,
        colsample_bynode=1.0,
        reg_lambda=1.0,
        reg_alpha=0.0,
        gamma=0.0,
        max_bin=256,
    )


@pytest.mark.parametrize(
    "change",
    [
        dict(learning_rate=0.001),
        dict(max_depth=0),
        dict(max_depth=3.5),
        dict(max_bin=300),
        dict(colsample_bynode=0.7),
        dict(reg_lambda=1e-7),
        dict(scale_pos_weight=2.0),
        dict(subsample=True),
    ],
)
def test_unsupported_historical_space(change):
    p = params()
    p.update(change)
    with pytest.raises(ValueError):
        hpo.Prior.uninformative([p])


def test_input_validation_and_defensive_copies():
    with pytest.raises(ValueError):
        hpo.Prior([0, 0], [[1, 2], [2, 1]], 0)
    with pytest.raises(ValueError):
        hpo.Prior([0], [[1]], -1)
    with pytest.raises(ValueError):
        hpo.Prior([np.nan], [[1]], 1)
    with pytest.raises(ValueError):
        hpo.Optimizer([{"x": 1}, {"x": 1}], prior=example_prior(2))
    with pytest.raises(ValueError):
        hpo.Optimizer([{"x": np.nan}], prior=example_prior(1))
    candidates = [{"x": i} for i in range(4)]
    prior = example_prior(4)
    opt = hpo.Optimizer(candidates, prior=prior)
    reference = optimizer(4)
    candidates[0]["x"] = 100
    prior.mean[:] = 1000
    assert opt.ask().candidate_id == reference.ask().candidate_id
    fixed = [params(), dict(params(), max_depth=5)]
    projected = hpo.Prior.uninformative(fixed)
    with pytest.raises(ValueError, match="different candidates"):
        hpo.Optimizer(fixed[::-1], prior=projected)


def test_native_prior_loader(tmp_path):
    candidates = [dict(params(), max_depth=d) for d in range(2, 8)]
    x = hpo._encode(candidates)
    models = {}
    for name, y in [("mean", np.linspace(-1, 1, 6)), ("scale", np.zeros(6))]:
        model = xgb.train(
            {"nthread": 1, "max_depth": 2, "objective": "reg:squarederror"},
            xgb.DMatrix(x, label=y, nthread=1),
            num_boost_round=2,
        )
        model.save_model(tmp_path / f"{name}.ubj")
        models[name] = model
    theta = np.r_[np.zeros(19), np.log(0.1)]
    (tmp_path / "model.json").write_text(
        json.dumps({"features": hpo._FEATURES, "logtheta": theta.tolist()})
    )
    manifest = dict(
        schema_version=1,
        encoding="xgboost-hpo-v1",
        direction="minimize",
        files={
            name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
            for name in ["model.json", "mean.ubj", "scale.ubj"]
        },
    )
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    prior = hpo.Prior.load(tmp_path, candidates=candidates)
    np.testing.assert_array_equal(
        prior.mean, models["mean"].predict(xgb.DMatrix(x, nthread=1))
    )
    opt = hpo.Optimizer(candidates, prior=prior)
    assert opt.ask().params in candidates
    (tmp_path / "mean.ubj").write_bytes(b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        hpo.Prior.load(tmp_path, candidates=candidates)


def test_real_training_ask_tell():
    rng = np.random.default_rng(10)
    x = rng.normal(size=(100, 3))
    y = (x[:, 0] + x[:, 1] > 0).astype(float)
    dt = xgb.DMatrix(x[:70], label=y[:70], nthread=1)
    dv = xgb.DMatrix(x[70:], label=y[70:], nthread=1)
    candidates = [dict(params(), max_depth=d) for d in [2, 3, 4, 5]]
    opt = hpo.Optimizer(candidates, random_state=10)
    for _ in candidates:
        trial = opt.ask()
        results = {}
        xgb.train(
            dict(
                trial.params,
                objective="binary:logistic",
                eval_metric="logloss",
                nthread=1,
            ),
            dt,
            num_boost_round=5,
            evals=[(dv, "validation")],
            evals_result=results,
            verbose_eval=False,
        )
        opt.tell(trial, results["validation"]["logloss"][-1])
    assert opt.best_trial is not None
    assert len(opt.history) == 4


def test_single_candidate_and_numeric_duplicates():
    opt = optimizer(1)
    opt.tell(opt.ask(), 0.0)
    assert opt.best_trial.candidate_id == 0
    with pytest.raises(StopIteration):
        opt.ask()
    with pytest.raises(ValueError, match="distinct"):
        hpo.Optimizer([{"x": 1}, {"x": 1.0}], prior=example_prior(2))
