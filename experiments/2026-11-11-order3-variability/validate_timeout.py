"""Verify later timeout invalidates an entire third-order entry."""
# pylint: disable=duplicate-code
# pylint: disable=missing-function-docstring

import tempfile
from pathlib import Path

import numpy as np
from run_repeated import run_case


def main():
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        ok = run_case(
            "fake",
            "quadrature",
            np.zeros(3),
            root / "ok",
            niter=4,
            warmup=1,
            limit=0.2,
            setup_limit=30,
            delays=[0.06],
        )
        assert ok["error"] is None and len(ok["samples_s"]) == 4, ok
        assert sum(ok["samples_s"]) > 0.2
        slow = run_case(
            "fake",
            "quadrature",
            np.zeros(3),
            root / "slow",
            niter=4,
            warmup=1,
            limit=0.2,
            setup_limit=30,
            delays=[0.01, 0.01, 0.4],
        )
        assert (
            slow["status"] == "timeout" and slow["timed_out_call"]["phase"] == "timed"
        ), slow
        assert (
            slow["mean_s"] is None
            and slow["sample_sd_s"] is None
            and not slow["samples_s"]
        )
        assert any(c["phase"] == "timed" for c in slow["prediction_calls"])
    print("PASS: long repeated total allowed; later over-limit call voids the entry")


if __name__ == "__main__":
    main()
