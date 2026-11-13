"""Per-prediction watchdog shared by first- and second-order runtime benchmarks."""

# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals,too-many-branches,protected-access,broad-exception-caught

import multiprocessing as mp
import time

import numpy as np


def failed(message, calls):
    """Return a missing entry after a failed or over-limit prediction."""
    return {
        "mean_time_s": None,
        "std_time_s": None,
        "samples_s": [],
        "n_samples": 0,
        "error": message,
        "prediction_calls": calls,
    }


class TimedPredictor:
    """Measure predictions and report phase boundaries to the parent."""

    def __init__(self, conn, predictor, warmup, limit, accuracy_calls=1):
        self.conn = conn
        self.predictor = predictor
        self.warmup = warmup
        self.limit = limit
        self.calls = []
        self.accuracy_calls = accuracy_calls

    def __call__(self, *args, **kwargs):
        index = len(self.calls)
        phase = (
            "accuracy"
            if index < self.accuracy_calls
            else "warmup"
            if index < self.accuracy_calls + self.warmup
            else "timed"
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
        if self.limit is not None and elapsed > self.limit:
            raise TimeoutError("Prediction exceeded individual-run limit")
        return value


class ResultSink:
    """Publish worker results using prediction-only timing samples."""

    def __init__(self, conn, predictor, niter):
        self.conn = conn
        self.predictor = predictor
        self.niter = niter

    def put(self, result):
        """Replace wrapper timings with durations that exclude IPC."""
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
    interactions,
):
    """Run one algorithm/device case in a spawned process."""
    try:
        from . import native as bench

        predictor = TimedPredictor(
            conn,
            predictor_override or bench.predict_with_algorithm,
            warmup,
            limit,
            2 if interactions else 1,
        )
        bench.predict_with_algorithm = predictor
        bench._benchmark_case_worker(
            ResultSink(conn, predictor, niter),
            raw,
            x,
            device,
            algorithm,
            interactions,
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
    interactions=False,
):
    """Enforce independent call deadlines for each prediction call."""
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
            interactions,
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
                limit is not None
                and active is not None
                and time.monotonic() > active["started_monotonic"] + limit + 0.1
                and not recv.poll()
            ):
                result = failed(
                    f"DNF: individual prediction exceeded {limit:g}s "
                    f"({active['phase']}, index {active['index']})",
                    calls,
                )
                result["timed_out_call"] = {
                    **active,
                    "observed_elapsed_s": time.monotonic()
                    - active["started_monotonic"],
                }
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
                    if limit is not None and event["elapsed_s"] > limit:
                        result = failed(
                            f"DNF: individual prediction exceeded {limit:g}s "
                            f"({event['phase']}, index {event['index']})",
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
