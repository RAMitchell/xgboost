"""Smoke and numerical checks for the self-contained paper reproduction package."""

import json
import os
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xgboost as xgb

pytest.importorskip("numba")
from paper.models import parse_trees
from paper.python_experiments import first_order, higher_order
from paper.reference import exhaustive_reference, flatten, predict_reference, rule
from paper.timing import run_case


def model():
    rng = np.random.default_rng(91)
    z = rng.normal(size=(80, 4)).astype(np.float32)
    x = np.zeros((80, 54), np.float32)
    x[:, [0, 12, 15, 45]] = z
    booster = xgb.train(
        {
            "objective": "reg:squarederror",
            "max_depth": 4,
            "min_child_weight": 0,
            "base_score": 0.5,
            "nthread": 1,
        },
        xgb.DMatrix(x, label=z[:, 0] * z[:, 1] + z[:, 2] ** 2),
        2,
    )
    return booster, x


def test_current_variants_and_reference():
    booster, x = model()
    rows = x[:3].astype(np.float64)
    arrays, base, _ = flatten(booster)
    expected = predict_reference(rows, arrays, base, 16)[:, :, :-1]
    root = Path(os.environ.get("TREEGRAD_ROOT", "paper/dependencies/TreeGrad"))
    variants = list("ABCDE") if root.exists() else list("BCDE")
    for variant in variants:
        actual = first_order(booster, variant, root)(rows)
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


def test_higher_order_feature_ids():
    booster, x = model()
    a = higher_order(booster, "quadrature", 3)(x[0].astype(np.float64))
    b = higher_order(booster, "treeshapiq", 3)(x[0].astype(np.float64))
    for key in a.keys() | b.keys():
        np.testing.assert_allclose(
            a.get(key, 0.0), b.get(key, 0.0), rtol=1e-9, atol=1e-11
        )
    assert all(set(k) <= set([0, 12, 15, 45]) for k in a)


def test_only_standard_rule():
    for n in [1, 3, 8, 16]:
        nodes, weights = rule(n)
        for degree in range(2 * n):
            np.testing.assert_allclose(
                np.dot(weights, nodes**degree), 1 / (degree + 1), rtol=1e-12, atol=1e-14
            )


def delayed_worker(pipe, case, args):
    samples = []
    for index, delay in enumerate(args.delays):
        pipe.send(
            {
                "event": "start",
                "phase": "timed",
                "index": index,
                "rows": 128,
                "started_monotonic": time.monotonic(),
            }
        )
        start = time.perf_counter()
        time.sleep(delay)
        elapsed = time.perf_counter() - start
        pipe.send(
            {
                "event": "end",
                "phase": "timed",
                "index": index,
                "rows": 128,
                "elapsed_s": elapsed,
            }
        )
        samples.append(elapsed)
    pipe.send(
        {
            "event": "result",
            "result": {
                "status": "complete",
                "error": None,
                "samples_s": samples,
                "mean_s": statistics.mean(samples),
                "sample_sd_s": statistics.stdev(samples),
            },
        }
    )
    pipe.close()


def test_timeout_is_per_complete_run(tmp_path):
    args = SimpleNamespace(limit=0.15, setup_limit=0.15, delays=[0.07] * 3)
    result = run_case(tmp_path / "complete", args, target=delayed_worker)
    assert result["status"] == "complete" and sum(result["samples_s"]) > args.limit
    args.delays = [0.01, 0.01, 0.4]
    result = run_case(tmp_path / "timeout", args, target=delayed_worker)
    assert (
        result["status"] == "timeout"
        and not result["samples_s"]
        and result["mean_s"] is None
    )


def test_float64_margin_reference():
    from paper.reference import predict_margin_float64

    booster, x = model()
    arrays, base, _ = flatten(booster)
    margin = predict_margin_float64(x[:3], arrays, base)
    phi = predict_reference(x[:3], arrays, base, 16)
    np.testing.assert_allclose(phi.sum(axis=-1), margin, rtol=1e-12, atol=1e-13)


class DelayedNativePredictor:
    def __init__(self, delays):
        self.delays = delays
        self.index = 0

    def __call__(self, booster, data, device, algorithm, interactions):
        time.sleep(self.delays[min(self.index, len(self.delays) - 1)])
        self.index += 1
        shape = (data.num_row(), data.num_col() + 1)
        return np.zeros(
            (*shape, data.num_col() + 1) if interactions else shape, np.float32
        )


@pytest.mark.parametrize("interactions", [False, True])
def test_native_timeout_phases(interactions):
    from paper.native_timeout import run_case as native_case

    x = np.asarray([[0.0], [1.0]], np.float32)
    b = xgb.train(
        {"nthread": 1, "base_score": 0.0, "objective": "reg:squarederror"},
        xgb.DMatrix(x, label=[0.0, 1.0]),
        1,
    )
    accuracy = 2 if interactions else 1
    ok = native_case(
        b,
        x,
        "cpu",
        "treeshap",
        3,
        np.zeros(2),
        warmup=1,
        nthread=1,
        limit=0.2,
        predictor_override=DelayedNativePredictor([0.08]),
        interactions=interactions,
    )
    assert ok["error"] is None, ok
    assert sum(ok["samples_s"]) > 0.2 and max(ok["samples_s"]) < 0.2
    assert [c["phase"] for c in ok["prediction_calls"]] == ["accuracy"] * accuracy + [
        "warmup"
    ] + ["timed"] * 3
    bad = native_case(
        b,
        x,
        "cpu",
        "treeshap",
        3,
        np.zeros(2),
        warmup=1,
        nthread=1,
        limit=0.2,
        predictor_override=DelayedNativePredictor([0.01] * (accuracy + 2) + [0.5]),
        interactions=interactions,
    )
    assert bad["error"] and not bad["samples_s"] and bad["mean_time_s"] is None
