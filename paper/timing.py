"""Independent setup/prediction deadlines; a failed call voids the whole entry."""

# Numerical state and orchestration are kept explicit for reproduction.
# pylint: disable=too-many-locals,too-many-branches,too-many-statements

import json
import multiprocessing as mp
import time


def save(path, value):
    """Atomically save a finite-valued JSON artifact."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def run_case(case, args, callback=None, *, target):
    """Watch each setup or prediction call and invalidate failed entries."""
    case.mkdir(parents=True, exist_ok=True)
    context = mp.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=target, args=(child, case, args))
    entry_started = time.monotonic()
    process.start()
    child.close()
    active = None
    calls = []
    result = None
    try:
        while result is None:
            if parent.poll(0.05):
                try:
                    event = parent.recv()
                except EOFError:
                    break
                if callback:
                    callback(event, calls)
                if event["event"] == "start":
                    active = event
                elif event["event"] == "end":
                    calls.append(event)
                    active = None
                    limit = (
                        args.setup_limit if event["phase"] == "setup" else args.limit
                    )
                    if event["elapsed_s"] > limit:
                        result = {
                            "status": "timeout",
                            "error": f"{event['phase']} call exceeded {limit}s",
                            "samples_s": [],
                            "mean_s": None,
                            "sample_sd_s": None,
                            "timed_out_call": event
                            | {"observed_elapsed_s": event["elapsed_s"]},
                        }
                        process.terminate()
                        break
                elif event["event"] == "result":
                    result = event["result"]
            now = time.monotonic()
            if active:
                limit = args.setup_limit if active["phase"] == "setup" else args.limit
                if (
                    now - active["started_monotonic"] > limit + 0.1
                    and not parent.poll()
                ):
                    result = {
                        "status": "timeout",
                        "error": f"{active['phase']} call exceeded {limit}s",
                        "samples_s": [],
                        "mean_s": None,
                        "sample_sd_s": None,
                        "timed_out_call": active
                        | {"observed_elapsed_s": now - active["started_monotonic"]},
                    }
                    process.terminate()
                    break
            elif not process.is_alive() and not parent.poll():
                break
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join()
        if result is None:
            result = {
                "status": "error",
                "error": f"Worker exited {process.exitcode} without result",
                "samples_s": [],
                "mean_s": None,
                "sample_sd_s": None,
            }
        entry_elapsed = time.monotonic() - entry_started
        result["entry_elapsed_s"] = entry_elapsed
        result["calls"] = calls
        result["prediction_limit_s"] = args.limit
        result["setup_limit_s"] = args.setup_limit
        save(case / "results.json", result)
        return result
    finally:
        if process.is_alive():
            process.terminate()
            process.join()
        parent.close()
