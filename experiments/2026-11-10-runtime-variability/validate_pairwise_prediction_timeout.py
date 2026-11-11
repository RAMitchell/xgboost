"""Exercise pairwise accuracy/additive/warm-up/timed phase accounting and timeout."""
# pylint: disable=duplicate-code
# pylint: disable=missing-class-docstring,missing-function-docstring,too-many-arguments,too-many-positional-arguments

import time

import numpy as np
import xgboost as xgb
from pairwise_prediction_timeout import run_case


class Predictor:
    def __init__(self, delays):
        self.delays = delays
        self.index = 0

    def __call__(self, booster, data, device, algorithm, interactions):
        delay = self.delays[min(self.index, len(self.delays) - 1)]
        self.index += 1
        time.sleep(delay)
        shape = (data.num_row(), data.num_col() + 1)
        if interactions:
            shape = (*shape, data.num_col() + 1)
        return np.zeros(shape, np.float32)


def main():
    x = np.array([[0.0], [1.0]], np.float32)
    b = xgb.train(
        {"nthread": 1, "base_score": 0.0, "objective": "reg:squarederror"},
        xgb.DMatrix(x, label=[0.0, 1.0], nthread=1),
        1,
    )
    ok = run_case(
        b,
        x,
        "cpu",
        "treeshap",
        4,
        None,
        warmup=1,
        nthread=1,
        limit=0.2,
        predictor_override=Predictor([0.07]),
    )
    assert ok["error"] is None, ok
    assert [c["phase"] for c in ok["prediction_calls"]] == ["accuracy"] * 2 + [
        "warmup"
    ] + ["timed"] * 4
    assert len(ok["samples_s"]) == 4 and sum(ok["samples_s"]) > 0.2
    slow = run_case(
        b,
        x,
        "cpu",
        "treeshap",
        4,
        None,
        warmup=1,
        nthread=1,
        limit=0.2,
        predictor_override=Predictor([0.01, 0.01, 0.01, 0.01, 0.4]),
    )
    assert "individual prediction exceeded" in slow["error"], slow
    assert (
        slow["mean_time_s"] is None and not slow["samples_s"] and slow["n_samples"] == 0
    )
    assert slow["timed_out_call"]["observed_elapsed_s"] > 0.2
    assert any(c["phase"] == "timed" for c in slow["prediction_calls"])
    print(
        "PASS: pairwise phases correct; long total permitted; one slow call voids all samples"
    )


if __name__ == "__main__":
    main()
