"""Regression checks: a long total is allowed, but one slow call voids all samples."""
# pylint: disable=duplicate-code
# pylint: disable=missing-class-docstring,missing-function-docstring,too-many-arguments,too-many-positional-arguments

import time

import numpy as np
import xgboost as xgb
from per_prediction_timeout import run_case


class Predictor:
    def __init__(self, delays):
        self.delays = delays
        self.index = 0

    def __call__(self, booster, data, device, algorithm, interactions):
        delay = self.delays[min(self.index, len(self.delays) - 1)]
        self.index += 1
        time.sleep(delay)
        return np.zeros((data.num_row(), data.num_col() + 1), np.float32)


def main():
    x = np.array([[0.0], [1.0]], np.float32)
    b = xgb.train(
        {"nthread": 1, "base_score": 0.0, "objective": "reg:squarederror"},
        xgb.DMatrix(x, label=[0.0, 1.0], nthread=1),
        1,
    )
    margin = np.zeros(len(x), np.float32)
    ok = run_case(
        b,
        x,
        "cpu",
        "treeshap",
        4,
        margin,
        warmup=1,
        nthread=1,
        limit=0.2,
        predictor_override=Predictor([0.08]),
    )
    assert ok["error"] is None, ok
    assert sum(ok["samples_s"]) > 0.2 > max(ok["samples_s"])
    slow = run_case(
        b,
        x,
        "cpu",
        "treeshap",
        4,
        margin,
        warmup=1,
        nthread=1,
        limit=0.2,
        predictor_override=Predictor([0.01, 0.01, 0.01, 0.4]),
    )
    assert "individual prediction exceeded" in slow["error"], slow
    assert (
        slow["mean_time_s"] is None and slow["n_samples"] == 0 and not slow["samples_s"]
    )
    assert any(c["phase"] == "timed" for c in slow["prediction_calls"])
    print(
        "PASS: total repeat time may exceed limit; one over-limit call voids the entire entry"
    )


if __name__ == "__main__":
    main()
