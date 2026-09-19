import os
import signal
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import pandas as pd
import pm4py

from src.naive_values.contribution_scores_log import Responsibility
from src.util.analysis_functions import (
    compute_handovers_manual,
    jaccard_similarity,
    compute_efg_metric,
    compute_conformance_checking,
)
from src.util.loader import load_ocel, flatten_log, load_event_log

OCEL_DIR_PATH = "data/ocels/"
RESULTS_DIR = "data/results/"
LOGS = {
        "BPIC17.jsonocel": {
                    "event_id": "EventID",
                    "resource_id": "resource",
                    "object_types": [
                    "Application",
                    "Workflow",
                    "Offer"
                    ],
                    "conformance_activities": {
                        "Application": ("A_Validating","A_Pending"),
                        "Workflow": ("W_Handle leads", "W_Complete application"),
                        "Offer": ("O_Sent (mail and online)", "O_Cancelled"),
                        "Case_R": ("W_Handle leads", "W_Complete application")
                    }
                }
}

naive_log_sizes = [2, 5, 10] #number of cases in log, for the naive/exact approach - kept small since it's combinatorial in the number of events
log_sizes = [2, 5, 10] #number of cases in log, for sampling based approaches
sample_sizes = [1000,5000,10000] #for sampling based approaches
values = ["shapley", "banzhaf", "owen", "banzhaf-owen"]
analysis = ["conformance","handover","efg"]
granularities = ["context", "event", "relation"]

MAX_WORKERS = 6
RUN_TIMEOUT_SECONDS = 600  # per (analysis, value[, sample_size]) run; set to None to disable

SYNTHETIC_EVENT_ATTR = "event:id"

RUNS_COLUMNS = ['run_id', 'log', 'case', 'log_size', 'granularity', 'sample_size', 'value', 'analysis', 'runtime', 'status', 'error']
VALUES_COLUMNS = ['run_id', 'event', 'orig_event', 'value']


def load_log(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".jsonocel", ".ocel"):
        return pm4py.read_ocel(path), True
    elif ext == ".xes":
        return load_event_log(path), False
    else:
        raise ValueError(f"unsupported log file extension {ext!r} for {path!r} (expected .jsonocel, .ocel, or .xes)")


def add_synthetic_event_ids(flat_log: pd.DataFrame, true_event_attr: str):
    # OCEL flattening can duplicate a real event across multiple rows (once per
    # related object of the chosen case-notion type) - using the true event id as
    # the Shapley/Owen player would silently collapse those duplicates into a
    # single player. Assign a fresh, row-unique id instead (as in eval.py), and
    # keep a mapping back to the original event id so per-row results can be
    # grouped per real event afterward, without losing any of them.
    flat_log = flat_log.assign(**{SYNTHETIC_EVENT_ATTR: flat_log.index})
    synthetic_to_orig = pd.Series(
        flat_log[true_event_attr].values, index=flat_log[SYNTHETIC_EVENT_ATTR]
    ).to_dict()
    return flat_log, synthetic_to_orig


def compute_naive(responsibility_calc, value, granularity):
    if value in ("shapley", "banzhaf"):
        return responsibility_calc.compute_simple_values(granularity=granularity, value=value)
    else:
        return responsibility_calc.compute_nested_value(granularity=granularity, value=value)


def compute_sampled(responsibility_calc, value, sample_size, granularity):
    if value in ("shapley", "banzhaf"):
        return responsibility_calc.compute_simple_value_sampling(sample_size, granularity=granularity, value=value)
    else:
        return responsibility_calc.compute_nested_value_sampling(sample_size, granularity=granularity, value=value)


def jaccard_similarity_safe(set1: set, set2: set) -> float:
    # jaccard_similarity(set(), set()) is 0/0; by convention two empty sets are
    # identical, so this coalition reproduced the (lack of) baseline behavior exactly
    if not set1 and not set2:
        return 1.0
    return jaccard_similarity(set1, set2)


def subsample_cases(flat_log: pd.DataFrame, log_size: int, seed: int = 26):
    # mirrors flatten_log's own downsampling, but reused across log_sizes so we
    # only have to flatten each (log, object_type) once
    case_ids = flat_log["case:concept:name"].drop_duplicates()
    if log_size > len(case_ids):
        return None
    sampled_case_ids = case_ids.sample(n=log_size, random_state=seed)
    return flat_log[flat_log["case:concept:name"].isin(sampled_case_ids)]


def build_analysis_func(analysis_name, flat_log, log_config, object_type):
    if analysis_name == "handover":
        if log_config["resource_id"] is None:
            return None  # log has no resource attribute -> handover isn't computable
        resource_attr = log_config["resource_id"]
        baseline_edges = set(compute_handovers_manual(flat_log, resource_attr=resource_attr).keys())
        return lambda x: jaccard_similarity_safe(
            baseline_edges, set(compute_handovers_manual(x, resource_attr=resource_attr).keys())
        )
    elif analysis_name == "efg":
        num_activities = flat_log["concept:name"].nunique()
        return lambda x: compute_efg_metric(x, num_activities)
    else:  # conformance
        activity_a, activity_b = log_config["conformance_activities"][object_type]
        event_attr = log_config["event_id"]
        return lambda x: compute_conformance_checking(x, activity_a, activity_b, "Response", event_attr)


class _RunTimeout(Exception):
    pass


def _raise_run_timeout(signum, frame):
    raise _RunTimeout()


def _run_with_timeout(compute_fn, timeout_seconds):
    # SIGALRM-based per-run timeout
    can_enforce = timeout_seconds is not None and hasattr(signal, "SIGALRM")
    old_handler = None
    if can_enforce:
        old_handler = signal.signal(signal.SIGALRM, _raise_run_timeout)
        signal.alarm(timeout_seconds)

    t0 = time.time()
    try:
        result_values = compute_fn()
        return result_values, time.time() - t0, "ok", None
    except _RunTimeout:
        return {}, time.time() - t0, "timeout", f"exceeded {timeout_seconds}s"
    except Exception as e:
        return {}, time.time() - t0, "error", repr(e)
    finally:
        if can_enforce:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)


def run_group(mode, log_name, object_type, log_size, flat_log, log_config, timeout_seconds):
    flat_log, synthetic_to_orig = add_synthetic_event_ids(flat_log, log_config["event_id"])
    results = []

    for analysis_name in analysis:
        analysis_func = build_analysis_func(analysis_name, flat_log, log_config, object_type)
        if analysis_func is None:
            results.append({
                "log": log_name, "case": object_type, "log_size": log_size,
                "granularity": None, "sample_size": None, "value": None, "analysis": analysis_name,
                "runtime": 0.0, "status": "skipped", "error": "no resource attribute",
                "values": {},
            })
            continue

        responsibility_calc = Responsibility(flat_log, False,
                                              analysis_function=analysis_func,
                                              event_attr=SYNTHETIC_EVENT_ATTR)

        for granularity in granularities:
            for value in values:
                sample_size_options = [None] if mode == "naive" else sample_sizes

                for sample_size in sample_size_options:
                    if mode == "sampled" and granularity == "context" and value in ("owen", "banzhaf-owen"):
                        results.append({
                            "log": log_name, "case": object_type, "log_size": log_size,
                            "granularity": granularity, "sample_size": sample_size, "value": value,
                            "analysis": analysis_name, "runtime": 0.0, "status": "skipped",
                            "error": "sampled nested value only supports granularity='event'",
                            "values": {},
                        })
                        continue

                    if mode == "naive":
                        compute_fn = lambda rc=responsibility_calc, v=value, g=granularity: compute_naive(rc, v, g)
                    else:
                        compute_fn = lambda rc=responsibility_calc, v=value, s=sample_size, g=granularity: \
                            compute_sampled(rc, v, s, g)

                    result_values, runtime, status, error = _run_with_timeout(compute_fn, timeout_seconds)

                    results.append({
                        "log": log_name, "case": object_type, "log_size": log_size,
                        "granularity": granularity, "sample_size": sample_size, "value": value,
                        "analysis": analysis_name, "runtime": runtime, "status": status, "error": error,
                        "values": result_values,
                    })

    return results, synthetic_to_orig


def append_results(run_dir, results, synthetic_to_orig):
    run_rows = []
    value_rows = []

    for r in results:
        run_id = uuid.uuid4()
        run_rows.append({
            "run_id": run_id, "log": r["log"], "case": r["case"], "log_size": r["log_size"],
            "granularity": r["granularity"], "sample_size": r["sample_size"], "value": r["value"],
            "analysis": r["analysis"], "runtime": r["runtime"], "status": r["status"], "error": r["error"],
        })
        for event, v in r["values"].items():
            value_rows.append({
                "run_id": run_id, "event": event, "orig_event": synthetic_to_orig[event], "value": v,
            })

    pd.DataFrame(run_rows, columns=RUNS_COLUMNS).to_csv(
        os.path.join(run_dir, "runs.csv"), mode="a", header=False, index=False
    )
    pd.DataFrame(value_rows, columns=VALUES_COLUMNS).to_csv(
        os.path.join(run_dir, "values.csv"), mode="a", header=False, index=False
    )

    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_timeout = sum(1 for r in results if r["status"] == "timeout")
    n_error = sum(1 for r in results if r["status"] == "error")
    n_skipped = sum(1 for r in results if r["status"] == "skipped")
    print(f"[done] {results[0]['log']}/{results[0]['case']}/{results[0]['log_size']}: "
          f"{len(run_rows)} runs appended (ok={n_ok}, timeout={n_timeout}, error={n_error}, skipped={n_skipped})")


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(RESULTS_DIR, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    pd.DataFrame(columns=RUNS_COLUMNS).to_csv(os.path.join(run_dir, "runs.csv"), index=False)
    pd.DataFrame(columns=VALUES_COLUMNS).to_csv(os.path.join(run_dir, "values.csv"), index=False)
    print(f"writing results to {run_dir}")

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []

        for log_name, log_config in LOGS.items():
            log, is_ocel = load_log(OCEL_DIR_PATH + log_name)

            if is_ocel:
                case_flat_logs = [(object_type, flatten_log(log, object_type)) for object_type in log_config["object_types"]]
            else:
                case_flat_logs = [(log_name, log)]

            for object_type, full_flat_log in case_flat_logs:

                sampled_log = full_flat_log
                for log_size in reversed(naive_log_sizes):
                    sampled_log = subsample_cases(sampled_log, log_size)
                    if sampled_log is None:
                        print(f"[skip] {log_name}/{object_type}: fewer than {log_size} cases available (naive)")
                        continue
                    futures.append(executor.submit(
                        run_group, "naive", log_name, object_type, log_size, sampled_log, log_config, RUN_TIMEOUT_SECONDS
                    ))

                sampled_log = full_flat_log
                for log_size in reversed(log_sizes):
                    sampled_log = subsample_cases(sampled_log, log_size)
                    if sampled_log is None:
                        print(f"[skip] {log_name}/{object_type}: fewer than {log_size} cases available (sampled)")
                        continue
                    futures.append(executor.submit(
                        run_group, "sampled", log_name, object_type, log_size, sampled_log, log_config, RUN_TIMEOUT_SECONDS
                    ))

        for future in as_completed(futures):
            results, synthetic_to_orig = future.result()
            append_results(run_dir, results, synthetic_to_orig)

    print(f"all results written to {run_dir}")


if __name__ == "__main__":
    main()
