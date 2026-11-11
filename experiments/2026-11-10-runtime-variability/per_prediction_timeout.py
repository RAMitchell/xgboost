"""A process watchdog for each prediction, with no shared budget across repeats."""
# pylint: disable=duplicate-code
# pylint: disable=broad-exception-caught,line-too-long,missing-class-docstring,missing-function-docstring,protected-access,too-many-arguments,too-many-branches,too-many-locals,too-many-positional-arguments

import importlib
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

BENCH = (
    Path(__file__).resolve().parents[2]
    / "experiments/2026-04-10-rapids-style-shap-benchmark"
)


def failed(message, calls):
    return {
        "mean_time_s": None,
        "std_time_s": None,
        "samples_s": [],
        "n_samples": 0,
        "error": message,
        "prediction_calls": calls,
    }


class TimedPredictor:
    def __init__(self, conn, predictor, warmup, limit):
        self.conn = conn
        self.predictor = predictor
        self.warmup = warmup
        self.limit = limit
        self.calls = []

    def __call__(self, *args, **kwargs):
        index = len(self.calls)
        phase = (
            "accuracy" if index == 0 else "warmup" if index <= self.warmup else "timed"
        )
        event = {
            "event": "start",
            "phase": phase,
            "index": index,
            "started_monotonic": time.monotonic(),
        }
        self.conn.send(event)
        # IPC is outside the prediction timer; preserve the original predict scope.
        start = time.perf_counter()
        value = self.predictor(*args, **kwargs)
        elapsed = time.perf_counter() - start
        call = {"phase": phase, "index": index, "elapsed_s": elapsed}
        self.calls.append(call)
        self.conn.send({"event": "end", **call})
        if elapsed > self.limit:
            raise TimeoutError("Prediction exceeded individual-run limit")
        return value


class ResultSink:
    def __init__(self, conn, predictor, niter):
        self.conn = conn
        self.predictor = predictor
        self.niter = niter

    def put(self, result):
        result["prediction_calls"] = self.predictor.calls
        if result["error"] is None:
            samples = [
                c["elapsed_s"] for c in self.predictor.calls if c["phase"] == "timed"
            ]
            assert len(samples) == self.niter
            result.update(
                samples_s=samples,
                n_samples=len(samples),
                mean_time_s=float(np.mean(samples)),
                std_time_s=float(np.std(samples, ddof=1)) if len(samples) > 1 else None,
                median_time_s=float(np.median(samples)),
                q25_time_s=float(np.quantile(samples, 0.25)),
                q75_time_s=float(np.quantile(samples, 0.75)),
                min_time_s=min(samples),
                max_time_s=max(samples),
            )
        self.conn.send({"event": "result", "result": result})


def worker(
    conn,
    raw,
    x,
    device,
    algorithm,
    niter,
    margin,
    warmup,
    nthread,
    limit,
    predictor_override,
):
    try:
        sys.path.insert(0, str(BENCH))
        bench = importlib.import_module("quadratureshap_rapids_benchmark")
        predictor = TimedPredictor(
            conn, predictor_override or bench.predict_with_algorithm, warmup, limit
        )
        bench.predict_with_algorithm = predictor
        bench._benchmark_case_worker(
            ResultSink(conn, predictor, niter),
            raw,
            x,
            device,
            algorithm,
            False,
            niter,
            margin,
            warmup,
            nthread,
        )
    except Exception as error:
        conn.send({"event": "result", "result": failed(str(error), [])})
    finally:
        conn.close()


def run_case(
    booster,
    x,
    device,
    algorithm,
    niter,
    margin,
    warmup=3,
    nthread=16,
    limit=600,
    progress=None,
    predictor_override=None,
):
    ctx = mp.get_context("spawn")
    recv, send = ctx.Pipe(duplex=False)
    proc = ctx.Process(
        target=worker,
        args=(
            send,
            bytes(booster.save_raw()),
            x,
            device,
            algorithm,
            niter,
            margin,
            warmup,
            nthread,
            limit,
            predictor_override,
        ),
    )
    proc.start()
    send.close()
    calls = []
    active = None
    result = None
    try:
        while result is None:
            # Small delivery grace avoids rejecting an in-limit call whose end
            # message is delayed; completed calls are checked against limit exactly.
            if (
                active is not None
                and time.monotonic() > active["started_monotonic"] + limit + 0.1
                and not recv.poll()
            ):
                result = failed(
                    f"DNF: individual prediction exceeded {limit:g}s ({active['phase']}, index {active['index']})",
                    calls,
                )
                break
            if recv.poll(0.05):
                try:
                    event = recv.recv()
                except EOFError:
                    result = failed(
                        f"Worker exited without result (exit code {proc.exitcode})",
                        calls,
                    )
                    break
                if event["event"] == "start":
                    active = event
                elif event["event"] == "end":
                    calls.append({k: event[k] for k in ["phase", "index", "elapsed_s"]})
                    if event["elapsed_s"] > limit:
                        result = failed(
                            f"DNF: individual prediction exceeded {limit:g}s ({event['phase']}, index {event['index']})",
                            calls,
                        )
                    active = None
                elif event["event"] == "result":
                    result = event["result"]
                else:
                    raise ValueError(event)
                if progress is not None:
                    progress(event, calls)
            elif not proc.is_alive():
                if recv.poll():
                    continue
                result = failed(
                    f"Worker exited without result (exit code {proc.exitcode})", calls
                )
    finally:
        if proc.is_alive():
            proc.join(timeout=1 if result and result["error"] is None else 0)
            if proc.is_alive():
                proc.terminate()
        proc.join()
        recv.close()
    result["prediction_timeout_seconds"] = limit
    result["timeout_scope"] = (
        "each prediction call; any over-limit call invalidates the entire algorithm/device entry"
    )
    return result
