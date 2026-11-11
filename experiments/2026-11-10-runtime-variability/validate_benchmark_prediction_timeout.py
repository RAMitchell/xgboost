"""Regression checks for the benchmark's default per-call timeout semantics."""
# pylint: disable=too-many-arguments,too-many-positional-arguments,wrong-import-position

import sys
import time
from pathlib import Path

import numpy as np
import xgboost as xgb

BENCH = (
    Path(__file__).resolve().parents[2]
    / "experiments/2026-04-10-rapids-style-shap-benchmark"
)
sys.path.insert(0, str(BENCH))
from prediction_timeout import run_case


class Predictor:
    """Provide deterministic prediction delays and correctly shaped zero outputs."""

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
    """Check per-call deadlines independently for both interaction modes."""
    x = np.array([[0.0], [1.0]], np.float32)
    b = xgb.train(
        {"nthread": 1, "base_score": 0.0, "objective": "reg:squarederror"},
        xgb.DMatrix(x, label=[0.0, 1.0], nthread=1),
        1,
    )
    for interactions in [False, True]:
        accuracy = 2 if interactions else 1

        def run(delays, interactions=interactions, **kwargs):
            """Run the delayed predictor for the current interaction mode."""
            return run_case(
                b,
                x,
                "cpu",
                "treeshap",
                4,
                np.zeros(2),
                warmup=1,
                nthread=1,
                limit=0.2,
                interactions=interactions,
                predictor_override=Predictor(delays),
                **kwargs,
            )

        ok = run([0.06])
        assert ok["error"] is None, ok
        assert sum(ok["samples_s"]) > 0.2 > max(ok["samples_s"])
        assert [c["phase"] for c in ok["prediction_calls"]] == [
            "accuracy"
        ] * accuracy + ["warmup"] + ["timed"] * 4
        for index in [accuracy - 1, accuracy, accuracy + 2]:
            slow = run([0.01] * index + [0.4])
            assert "individual prediction exceeded" in slow["error"], slow
            assert not slow["samples_s"] and slow["n_samples"] == 0
            assert slow["mean_time_s"] is None and slow["std_time_s"] is None
            assert slow["timed_out_call"]["index"] == index
        aggregate = run([0.01], aggregate_limit=0.001)
        assert "aggregate case exceeded" in aggregate["error"], aggregate
        print(
            "PASS:",
            "pairwise" if interactions else "first-order",
            "phases, long total, late failure and separate aggregate budget",
        )


if __name__ == "__main__":
    main()
