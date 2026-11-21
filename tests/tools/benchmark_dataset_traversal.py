"""Opt-in fixed CPU traversal comparison; synthetic tests never run this matrix.

Run ``python -m tests.tools.benchmark_dataset_traversal --describe`` for the closed
profile, or ``--baseline-source /tmp/dryml/BASELINE --output /tmp/dryml/result.json``
to execute it. ``--workload`` and ``--mode`` are repeatable explicit selections.
The baseline must be an archive of BASELINE_REVISION, with its revision supplied
by ``--baseline-revision``. No checkout reset, installation, network or Store is
used. Each child has a startup-inclusive deadline; the parent charges failures,
instrumentation and startup against one overall deadline. There is no automatic
retry. Missing required, noisy or incomplete evidence cannot establish a pass;
only the closed applicability schema exempts known unsupported comparisons.
Cost-domain protocol v2 gates comparable old/new steady work. Full-invocation
and end-to-end comparisons remain separately labeled informational cost signals;
grouped native work never fabricates a per-update steady interval.
Confirmation reuses instrumentation only with matching cell-owned runtime and
driver content identity. Dirty current source and historical cells without an
identity cannot reuse instrumentation; they must be measured within the remaining
budget or remain unverified. Timing cells are always measured anew.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
from contextlib import contextmanager
from typing import Literal, TypedDict
import json
import math
import os
from pathlib import Path
import platform
import signal
from statistics import median
import subprocess
import sys
from tempfile import TemporaryDirectory, mkstemp
from time import perf_counter

BASELINE_REVISION = "9a681eabfe53b66f9d6d1459960ba10b95c1ff1d"
WORKLOADS = ("indexed-implicit", "indexed-explicit", "ordinary", "direct", "mixed",
             "buffered", "torch", "tf", "keras", "jax")
MODES = ("old", "new", "matched-native", "optimized-native")
MEASUREMENT_PROTOCOL = "dryml-dataset-traversal-cost-domains-v2"
TRAINING = ("torch", "tf", "keras", "jax")


class TimingDomain(TypedDict):
    """JSON metric identity; count is timed work, not total performed work.

    ``name`` fixes clock boundaries, ``unit`` fixes the throughput numerator,
    and ``count``/``sections`` fix the timed population. Unknown domains must not
    be compared, even when durations and total performed work happen to match.
    """

    name: str
    unit: Literal["updates", "batches", "outputs"]
    count: int
    sections: int


def _training_domain(metric="steady") -> TimingDomain:
    return dict(name={"steady": "accepted-report-within-section-v1",
                      "invocation": "full-training-invocation-v1",
                      "end_to_end": "complete-training-window-v1"}[metric],
                unit="updates", count=124 if metric == "steady" else 128, sections=4)


class _TrainingClock:
    """Observe only existing first/last accepted reports, never source hooks.

    The four disjoint report-1-to-report-32 intervals each contain 31 completed
    updates. Their interiors include all ordinary between-report overhead. Gaps
    and the final tail remain visible as mixed setup/first-update/postlude costs;
    no claim of an isolated compile or capture duration is made.
    """

    def __init__(self, *, grouped=False):
        self.grouped = grouped
        self.reports = []
        self.intervals = []

    def report(self, step):
        if self.grouped:
            if step % 32 == 0:
                self.reports.append(dict(step=step, at=perf_counter()))
        elif step % 32 in (1, 0):
            self.reports.append(dict(step=step, at=perf_counter()))

    def finish(self, started, finished):
        expected = list(range(32, 129, 32)) if self.grouped else [s for n in range(0, 128, 32) for s in (n+1, n+32)]
        assert [r["step"] for r in self.reports] == expected
        assert all(started <= r["at"] <= finished for r in self.reports)
        assert all(a["at"] <= b["at"] for a, b in zip(self.reports, self.reports[1:]))
        if not self.grouped:
            self.intervals = [dict(first_report_step=a["step"], last_report_step=b["step"],
                                   timed_updates=b["step"]-a["step"], seconds=b["at"]-a["at"])
                              for a, b in zip(self.reports[::2], self.reports[1::2])]
        steady = sum(i["seconds"] for i in self.intervals) if self.intervals else None
        first = self.reports[0]["at"]-started
        gaps = [self.reports[i+1]["at"]-self.reports[i]["at"]
                for i in range(1, len(self.reports)-1, 2)] if not self.grouped else None
        return dict(steady_seconds=steady,
                    timing_domain=None if self.grouped else _training_domain(),
                    invocation_seconds=finished-started, invocation_domain=_training_domain("invocation"),
                    first_result_seconds=None if self.grouped else first,
                    first_group_result_seconds=first if self.grouped else None,
                    report_intervals=self.intervals,
                    report_offsets=[dict(step=r["step"], seconds=r["at"]-started) for r in self.reports],
                    boundary_costs=dict(initial_setup_first_report_seconds=first,
                        between_section_postlude_setup_first_report_seconds=gaps,
                        final_report_to_return_seconds=finished-self.reports[-1]["at"],
                        outside_steady_seconds=None if steady is None else finished-started-steady),
                    steady_blocker=("32-update grouped execution exposes no within-section first-update report; adding Python batch callbacks would change the optimized path"
                                    if self.grouped else None))


_PROFILE = dict(
    seed=1729, warmups=2, repetitions=5, cell_seconds=120, total_seconds=1200,
    device="cpu", producers=1, threads=1,
    indexed=dict(rows=4096, features=8, batch=32, passes=8, sections=64,
                 section_length=16, batches=1024),
    sources=dict(outputs=512, features=8, section_length=16),
    training=dict(rows=4096, features=16, targets=1, batch=32, updates=128,
                  sections=4, checkpoint_cadences=[0, 32]),
    buffered=dict(capacity=32, capture_after=129, outputs=512),
    keras=dict(capacity=8, prefetch=2),
)


def fixed_profile() -> dict:
    """Return independent JSON controls; edits cannot alter the fixed protocol."""
    return copy.deepcopy(_PROFILE)


def _summary(samples):
    center = median(samples)
    return dict(median_seconds=center, mad_seconds=median(abs(x-center) for x in samples))


def _valid_instrumentation(cell):
    return (cell.get("status") == "ok" and cell.get("instrumentation") is True and cell.get("validated") is True
            and cell.get("measurement_protocol") == MEASUREMENT_PROTOCOL
            and type(cell.get("work")) is int and cell["work"] > 0
            and type(cell.get("peak_python_bytes")) is int and cell["peak_python_bytes"] >= 0)


def _positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _nonnegative(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _dataset_domain(workload, metric="steady") -> TimingDomain:
    indexed = workload.startswith("indexed")
    return dict(name={"steady": "post-first-section-delivery-v1",
                      "end_to_end": "complete-dataset-window-v1"}[metric],
                unit="batches" if indexed else "outputs",
                count=(1024 if indexed else 512) - (
                    64 if workload == "indexed-explicit" else 8 if indexed else 1 if workload == "buffered" else 32
                ) if metric == "steady" else 1024 if indexed else 512,
                sections=64 if workload == "indexed-explicit" else 8 if indexed else 1 if workload == "buffered" else 32)


def _expected_domain(workload, mode, metric):
    if workload in TRAINING:
        if metric == "steady" and mode == "optimized-native" and workload in ("keras", "jax"):
            return None
        return _training_domain(metric)
    if metric == "invocation":
        return None
    if metric == "steady" and mode == "optimized-native":
        return None
    return _dataset_domain(workload, metric)


def _valid_window(window, workload, mode):
    if not isinstance(window, dict) or window.get("status") != "ok":
        return False
    domain = _expected_domain(workload, mode, "steady")
    if window.get("timing_domain") != domain:
        return False
    if (window.get("end_to_end_domain") != _expected_domain(workload, mode, "end_to_end")
            or not _positive(window.get("end_to_end_seconds"))):
        return False
    if domain is None:
        if window.get("steady_seconds") is not None or not window.get("steady_blocker"):
            return False
    elif not _positive(window.get("steady_seconds")):
        return False
    if workload not in TRAINING:
        if domain is None:
            return (_positive(window.get("delivery_seconds"))
                    and window["end_to_end_seconds"] >= window["delivery_seconds"]
                    and window.get("delivery_domain") == {
                        **_dataset_domain(workload, "end_to_end"), "name": "full-vectorized-dataset-delivery-v1"})
        intervals = window.get("delivery_intervals")
        return (isinstance(intervals, list) and bool(intervals)
                and all(isinstance(i, dict) and type(i.get("count")) is int and i["count"] >= 0
                        and _nonnegative(i.get("seconds")) for i in intervals)
                and sum(i["count"] for i in intervals) == domain["count"]
                and sum(i["seconds"] for i in intervals) == window["steady_seconds"])
    if (window.get("invocation_domain") != _training_domain("invocation")
            or not _positive(window.get("invocation_seconds"))):
        return False
    costs = window.get("costs")
    if (not isinstance(costs, dict) or not _nonnegative(costs.get("final_save_seconds"))
            or not isinstance(costs.get("checkpoint_seconds"), list)
            or not all(_nonnegative(s) for s in costs["checkpoint_seconds"])
            or not _nonnegative(window.get("prepare_seconds"))
            or not _nonnegative(window.get("end_to_end_other_seconds"))
            or sum(costs["checkpoint_seconds"]) > window["invocation_seconds"]
            or window["end_to_end_seconds"] < window["invocation_seconds"]+window["prepare_seconds"]+costs["final_save_seconds"]):
        return False
    if not math.isclose(window["end_to_end_seconds"], window["invocation_seconds"]+window["prepare_seconds"]
                        +costs["final_save_seconds"]+window["end_to_end_other_seconds"], abs_tol=1e-12):
        return False
    boundary = window.get("boundary_costs")
    reports = window.get("report_offsets")
    if not isinstance(boundary, dict) or not isinstance(reports, list):
        return False
    expected = list(range(32, 129, 32)) if domain is None else [s for n in range(0, 128, 32) for s in (n+1, n+32)]
    if (len(reports) != len(expected) or any(not isinstance(r, dict) or type(r.get("step")) is not int or r["step"] != step
            or not _nonnegative(r.get("seconds")) or r["seconds"] > window["invocation_seconds"]
            for r, step in zip(reports, expected))
            or any(a["seconds"] > b["seconds"] for a, b in zip(reports, reports[1:]))):
        return False
    initial, tail = boundary.get("initial_setup_first_report_seconds"), boundary.get("final_report_to_return_seconds")
    if (not _nonnegative(initial) or not _nonnegative(tail)
            or not math.isclose(initial, reports[0]["seconds"], abs_tol=1e-12)
            or not math.isclose(tail+reports[-1]["seconds"], window["invocation_seconds"], abs_tol=1e-12)):
        return False
    if domain is None:
        return (window.get("first_result_seconds") is None and window.get("report_intervals") == []
                and window.get("first_group_result_seconds") == initial)
    if window.get("first_result_seconds") != initial:
        return False
    intervals = window.get("report_intervals")
    if not isinstance(intervals, list) or len(intervals) != 4:
        return False
    for index, interval in enumerate(intervals):
        if (not isinstance(interval, dict) or interval.get("first_report_step") != 32*index+1
                or interval.get("last_report_step") != 32*index+32
                or type(interval.get("timed_updates")) is not int or interval["timed_updates"] != 31
                or not _positive(interval.get("seconds"))):
            return False
        if not math.isclose(interval["seconds"], reports[2*index+1]["seconds"]-reports[2*index]["seconds"], abs_tol=1e-12):
            return False
    gaps = boundary.get("between_section_postlude_setup_first_report_seconds")
    if (not isinstance(gaps, list) or len(gaps) != 3 or not all(_nonnegative(g) for g in gaps)
            or any(not math.isclose(g, reports[2*i+2]["seconds"]-reports[2*i+1]["seconds"], abs_tol=1e-12)
                   for i, g in enumerate(gaps))
            or not _nonnegative(boundary.get("outside_steady_seconds"))
            or not math.isclose(initial+sum(gaps)+tail+window["steady_seconds"], window["invocation_seconds"], abs_tol=1e-12)
            or not math.isclose(boundary["outside_steady_seconds"]+window["steady_seconds"], window["invocation_seconds"], abs_tol=1e-12)):
        return False
    return (sum(i["seconds"] for i in intervals) == window["steady_seconds"]
            and window["invocation_seconds"] >= window["steady_seconds"])


def _valid_windows(cell):
    warmups, windows, totals = (cell.get(key, []) for key in ("warmups", "windows", "end_to_end_seconds"))
    workload, mode = cell.get("workload"), cell.get("mode")
    if workload not in WORKLOADS or mode not in MODES:
        return False
    samples = cell.get("samples")
    domain = _expected_domain(workload, mode, "steady")
    cadence = cell.get("checkpoint_cadence")
    return (type(cadence) is int and cadence in ((0, 32) if workload in TRAINING else (0,))
            and cell.get("measurement_protocol") == MEASUREMENT_PROTOCOL
            and cell.get("status") == "ok" and cell.get("validated") is True
            and type(cell.get("work")) is int and cell["work"] == _expected_domain(workload, mode, "end_to_end")["count"]
            and cell.get("timing_domain") == domain
            and cell.get("invocation_domain") == _expected_domain(workload, mode, "invocation")
            and cell.get("end_to_end_domain") == _expected_domain(workload, mode, "end_to_end")
            and isinstance(samples, list) and len(samples) == 5
            and all(isinstance(part, list) for part in (warmups, windows, totals))
            and len(warmups) == _PROFILE["warmups"]
            and all(isinstance(w, dict) and type(w.get("work")) is int
                    and w["work"] == cell.get("work") and _positive(w.get("elapsed_seconds"))
                    and w["elapsed_seconds"] == w.get("end_to_end_seconds")
                    and _valid_window(w, workload, mode) for w in warmups)
            and len(windows) == len(totals) == _PROFILE["repetitions"]
            and all(_valid_window(w, workload, mode) and type(w.get("work")) is int
                    and w["work"] == cell.get("work") and w.get("steady_seconds") == sample
                    and _positive(total) and total == w.get("end_to_end_seconds")
                    and total >= (w.get("invocation_seconds") or sample or 0)
                    for w, sample, total in zip(windows, samples, totals))
            and (workload not in TRAINING or all(
                w["costs"].get("checkpoint_steps") == ([32, 64, 96, 128] if cadence else [])
                and len(w["costs"]["checkpoint_seconds"]) == (4 if cadence else 0)
                for w in warmups+windows)))


def _aggregate_metrics(cell):
    domain, samples, totals = cell["timing_domain"], cell["samples"], cell["end_to_end_seconds"]
    metrics = dict(summary=_summary(samples) if domain else None,
                   throughput_per_second=domain["count"]/median(samples) if domain else None,
                   throughput_unit=domain["unit"]+"/second" if domain else None,
                   end_to_end_summary=_summary(totals),
                   end_to_end_throughput_per_second=cell["work"]/median(totals))
    if cell["workload"] in TRAINING:
        metrics["invocation_summary"] = _summary([w["invocation_seconds"] for w in cell["windows"]])
    return metrics


def _valid_protocol(cell):
    return _valid_windows(cell) and all(cell.get(key) == value for key, value in _aggregate_metrics(cell).items())


def _applicability():
    """Closed required-work schema, independent of installed backends or results."""
    result = []
    for workload in WORKLOADS:
        training = workload in ("torch", "tf", "keras", "jax")
        work = _PROFILE["training"]["updates"] if training else (
            _PROFILE["indexed"]["batches"] if workload.startswith("indexed") else _PROFILE["sources"]["outputs"])
        for cadence in _PROFILE["training"]["checkpoint_cadences"] if training else (0,):
            for mode in MODES:
                reason = None
                if mode == "old" and not training and workload != "indexed-implicit":
                    reason = "pinned baseline has no traversal sections/continuation API"
                elif mode.endswith("native") and workload in ("ordinary", "direct", "mixed", "buffered"):
                    reason = "native direct/mixed/replay/capture semantics unavailable"
                for instrumentation in (False, True):
                    result.append(dict(workload=workload, mode=mode, checkpoint_cadence=cadence,
                                       instrumentation=instrumentation, required=reason is None,
                                       expected_work=work, reason=reason))
    return result


def _cell_key(cell):
    return tuple(cell.get(k) for k in ("workload", "mode", "checkpoint_cadence", "instrumentation"))


def _valid_evidence(cell, spec):
    if cell.get("work") != spec["expected_work"]:
        return False
    if spec["instrumentation"]:
        return _valid_instrumentation(cell)
    return _valid_protocol(cell)


def _qualify_report(report):
    """Rebuild applicability, blockers and comparisons without executing any work."""
    schema = _applicability()
    report.update(applicability_schema="dryml-dataset-traversal-applicability-v1",
                  applicability=schema, comparisons=[], blockers=[], status="unverified",
                  decision_domain="old-new-steady", driver_limitations=[])
    specs = {_cell_key(s): s for s in schema}
    cells = {}
    for cell in report["cells"]:
        cells.setdefault(_cell_key(cell), []).append(cell)
    valid = {}
    for key, spec in specs.items():
        matches = cells.get(key, [])
        if len(matches) != 1:
            report["blockers"].append(dict(cell=list(key), reason="missing or duplicate cell"))
        elif spec["required"]:
            if _valid_evidence(matches[0], spec):
                valid[key] = matches[0]
                if not spec["instrumentation"] and matches[0]["timing_domain"] is None:
                    report["driver_limitations"].append(dict(cell=list(key),
                        reason=matches[0]["windows"][0]["steady_blocker"],
                        disposition="steady ratio unavailable; full-work costs required"))
            else:
                report["blockers"].append(dict(cell=list(key), reason="required evidence incomplete or invalid",
                                               status=matches[0].get("status")))
        elif matches[0].get("status") != "unsupported":
            report["blockers"].append(dict(cell=list(key), reason="known unsupported cell must remain labeled"))
    if set(cells) - set(specs):
        report["blockers"].append(dict(reason="unexpected cells outside fixed applicability schema"))
    for spec in schema:
        key = _cell_key(spec)
        if spec["mode"] != "new" or spec["instrumentation"] or key not in cells:
            continue
        for mode in ("old", "matched-native", "optimized-native"):
            base_key = (spec["workload"], mode, spec["checkpoint_cadence"], False)
            base_spec = specs[base_key]
            purpose = "old-new-regression" if mode == "old" else "native-gap"
            if not base_spec["required"]:
                comparison = dict(status="not-applicable", ratio=None, reason=base_spec["reason"])
            else:
                comparison = compare(valid.get(base_key, {}), valid.get(key, {}))
                if purpose == "old-new-regression" and comparison["status"] != "no-regression-observed":
                    report["blockers"].append(dict(cell=list(key), reason="unresolved old/new comparison",
                                                   status=comparison["status"]))
            report["comparisons"].append(dict(workload=spec["workload"], baseline=mode,
                checkpoint_cadence=spec["checkpoint_cadence"], purpose=purpose, metric="steady",
                followup=f"{spec['workload']}/{mode}/cadence-{spec['checkpoint_cadence']}-native-gap" if purpose == "native-gap" else None,
                 **comparison))
            if base_spec["required"]:
                for metric in (("invocation", "end_to_end") if spec["workload"] in TRAINING else ("end_to_end",)):
                    report["comparisons"].append(dict(workload=spec["workload"], baseline=mode,
                        checkpoint_cadence=spec["checkpoint_cadence"], purpose="informational-cost-domain",
                        metric=metric, **compare(valid.get(base_key, {}), valid.get(key, {}), metric=metric)))
    elapsed = report.get("elapsed_seconds")
    if (report.get("measurement_protocol") != MEASUREMENT_PROTOCOL
            or report.get("profile") != fixed_profile() or report.get("baseline_revision") != BASELINE_REVISION
            or type(elapsed) not in (int, float) or not math.isfinite(elapsed) or not 0 <= elapsed <= 1200):
        report["blockers"].append(dict(reason="invalid measurement protocol, fixed profile, revision or exhausted total budget"))
    if not report["blockers"]:
        report["status"] = "no-regression-observed"


def _note(error, message):
    add_note = getattr(error, "add_note", None)
    if callable(add_note):
        add_note(message)
    else:
        error.__notes__ = [*getattr(error, "__notes__", ()), message]


@contextmanager
def _report_writer(output):
    """Reserve a new name nonblockingly for the whole run; publish complete JSON.

    Requires local advisory locks and same-parent atomic replace. The stable lock
    file is never unlinked. Failed writes clean only their unique stage, preserve
    the prior complete snapshot, and propagate the primary exception. A failure
    around replace may have committed a complete new snapshot, never partial JSON.
    """
    from dryml.locking import FileLock
    output = output.absolute()
    lock = FileLock(output.with_name(f".{output.name}.benchmark.lock"))
    if not lock.acquire(blocking=False):
        raise SystemExit("evidence output already owned by a competing run")
    primary = None
    try:
        if os.path.lexists(output):
            raise SystemExit("refusing to overwrite existing evidence")

        def publish(report):
            raw = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode("utf-8")
            fd, name = mkstemp(prefix=f".{output.name}.staging-", dir=output.parent)
            stage = Path(name)
            replacing = False
            try:
                file = os.fdopen(fd, "wb")
                fd = None
                write_error = None
                try:
                    if file.write(raw) != len(raw):
                        raise OSError("short benchmark report write")
                    file.flush()
                    os.fsync(file.fileno())
                except BaseException as error:
                    write_error = error
                    raise
                finally:
                    try:
                        file.close()
                    except BaseException as cleanup:
                        if write_error is None:
                            raise
                        _note(write_error, f"Benchmark stage close failed: {cleanup}")
                replacing = True
                os.replace(stage, output)
                if os.name != "nt":
                    directory = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
                    try:
                        os.fsync(directory)
                    finally:
                        os.close(directory)
            except BaseException as error:
                if replacing:
                    _note(error, "Benchmark publication may have committed a complete snapshot around replace; inspect output.")
                try:
                    if fd is not None:
                        os.close(fd)
                    stage.unlink(missing_ok=True)
                except BaseException as cleanup:
                    _note(error, f"Benchmark stage cleanup failed: {cleanup}")
                raise
        yield publish
    except BaseException as error:
        primary = error
        raise
    finally:
        try:
            lock.release()
        except BaseException as cleanup:
            if primary is None:
                raise
            _note(primary, f"Benchmark output lock release failed: {cleanup}")


def compare(old: dict, new: dict, *, metric: str = "steady") -> dict:
    """Compare five protocol-v2 windows in one identical typed cost domain.

    ``metric`` selects steady, invocation, or end_to_end seconds. Returns no ratio
    for failed, unsupported, nonfinite, incomplete, unequal-work or incompatible
    domains. The numerator uses exactly the domain's timed work count and unit.
    Positive differences beyond the band are regressions. Differences within a
    nonzero band are unresolved noise, not equivalence or a universal allowance.
    Invocation/end-to-end comparisons describe total costs, not the R28 steady gate.
    """
    if metric not in ("steady", "invocation", "end_to_end"):
        raise ValueError("unknown timing metric")
    if (not (_valid_protocol(old) and _valid_protocol(new)) or old["work"] != new["work"]
            or old["workload"] != new["workload"] or old.get("checkpoint_cadence") != new.get("checkpoint_cadence")):
        return dict(status="unverified", ratio=None, reason="incomplete cost-domain protocol evidence")
    key = "timing_domain" if metric == "steady" else metric+"_domain"
    domain = old.get(key)
    if domain is None or domain != new.get(key):
        return dict(status="unverified", ratio=None, reason="unavailable or incompatible timing domains")
    seconds = "steady_seconds" if metric == "steady" else metric+"_seconds"
    samples = [[w[seconds] for w in c["windows"]] for c in (old, new)]
    a, b = (_summary(s) for s in samples)
    noise = 3 * max(a["mad_seconds"], b["mad_seconds"])
    delta = b["median_seconds"] - a["median_seconds"]
    status = "regression" if delta > noise else "noisy" if noise and abs(delta) <= noise else "no-regression-observed"
    return dict(status=status, ratio=b["median_seconds"] / a["median_seconds"],
                timing_domain=domain, noise_seconds=noise, delta_seconds=delta, old=a, new=b,
                old_throughput_per_second=domain["count"]/a["median_seconds"],
                new_throughput_per_second=domain["count"]/b["median_seconds"])


def run_worker(command: list[str], *, timeout: float, env: dict | None) -> dict:
    """Execute/reap one bounded JSON child, including startup in elapsed time.

    A timeout kills the POSIX process group (or child on Windows) and waits for
    exit. Nonzero exit, invalid JSON, or an invalid successful result is failed.
    Captured stderr remains local benchmark evidence; no credentials are requested.

    Args:
        command: Child executable and arguments as strings.
        timeout: Startup-inclusive deadline in seconds.
        env: Child environment mapping, or None to inherit the caller's.
    Returns:
        JSON result with status, elapsed seconds, reap confirmation and bounded
        stderr. Timing success requires two warmups and five work-matched windows;
        separate instrumentation success requires validated work and peak memory.
    Raises:
        OSError: Process creation or communication fails. Interruptions propagate
            after killing and reaping the child; this function writes no report.
    """
    started = perf_counter()
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, env=env, start_new_session=os.name != "nt") as child:
        def stop():
            try:
                if os.name != "nt":
                    os.killpg(child.pid, signal.SIGKILL)
                else:
                    child.kill()
            except ProcessLookupError:
                pass
            return child.communicate()
        try:
            stdout, stderr = child.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            stdout, stderr = stop()
            return dict(status="timeout", elapsed_seconds=perf_counter()-started,
                        worker_reaped=True, stderr=stderr[-8000:])
        except BaseException:
            stop()
            raise
        try:
            result = json.loads(stdout)
            if not isinstance(result, dict) or result.get("status") not in ("ok", "unsupported", "unavailable"):
                raise ValueError("invalid worker result")
            if result["status"] == "ok" and not (_valid_protocol(result) or _valid_instrumentation(result)):
                raise ValueError("invalid successful timing result")
            if child.returncode:
                raise ValueError("worker failed")
        except (ValueError, TypeError):
            result = dict(status="failed", stdout=stdout[-8000:])
        return {**result, "elapsed_seconds": perf_counter()-started,
                "worker_reaped": True, "stderr": stderr[-8000:]}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse fixed-profile selections; reject altered controls through argparse."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--describe", action="store_true")
    parser.add_argument("--workload", action="append", choices=WORKLOADS)
    parser.add_argument("--mode", action="append", choices=MODES)
    parser.add_argument("--baseline-source", type=Path)
    parser.add_argument("--baseline-revision", default=BASELINE_REVISION, choices=[BASELINE_REVISION])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summarize", type=Path, help="summarize existing JSON without running workloads")
    parser.add_argument("--confirm", type=Path, help="one confirmation, charged against prior report's overall budget")
    for name in ("seed", "warmups", "repetitions", "cell-seconds", "total-seconds"):
        value = _PROFILE[name.replace("-", "_")]
        parser.add_argument("--"+name, type=int, choices=[value], default=value)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--instrumentation", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint-cadence", type=int, choices=[0, 32], default=0,
                        help="worker cadence; parent runs both fixed training cadences")
    return parser.parse_args(argv)


def _ordinary(size=512, *, seed=1729):
    import numpy as np
    rng = np.random.default_rng(seed)
    for index in range(size):
        value = rng.random(8, dtype=np.float32)
        value[0] = index
        yield value


def _indexed_recipe(values):
    from dryml.data import ArrayDataset, Batch, Map, Scale
    return Batch(Map(ArrayDataset(values), Scale(std=2)), 32)


def _tf_layer_defs():
    from dryml import F
    return (F("InputLayer", shape=(16,)),
            F("Dense", 1, use_bias=False, kernel_initializer="zeros"))


def _tf_training_recipe(workload, mode, *, sections=4):
    """Build untimed TF owners for old/new explicit-loop or Keras training.

    ``sections`` defaults to the fixed four-section recipe; tiny functional
    qualification may supply two. Returns the zero-initialized 16-to-1 model
    and SGD(.001)/explicit mean-MSE trainer without acquiring data or executing
    updates. The loss capability also reaches pinned-old Keras compile kwargs.
    Installs DRYML's import guard before TensorFlow. Native construction errors
    propagate; unknown workloads/modes raise ValueError before framework imports.
    """
    if workload not in ("tf", "keras") or mode not in ("old", "new"):
        raise ValueError("TF recipe requires tf/keras and old/new")
    import dryml  # noqa: F401
    import tensorflow as tf
    from dryml.models.tf import BasicTraining, Loss, Optimizer, Sequential, Training

    tf.keras.utils.set_random_seed(1729)
    model = Sequential(layer_defs=_tf_layer_defs())
    trainer_type = BasicTraining if workload == "keras" else Training
    trainer = trainer_type(optimizer=Optimizer(tf.keras.optimizers.SGD, learning_rate=.001),
                           loss=Loss(tf.keras.losses.MeanSquaredError),
                           epochs=1 if mode == "old" else sections, verbose=0,
                           compile_kwargs={"run_eagerly": False, "jit_compile": False})
    return model, trainer


_canary_counts = dict(opens=0, yields=0)


def _canary_factory(size):
    import numpy as np
    _canary_counts["opens"] += 1
    for index in range(size):
        _canary_counts["yields"] += 1
        yield np.int64(index)


def _direct_class():
    from dryml.data import RecoverableGenerator
    globals()["RecoverableGenerator"] = RecoverableGenerator

    class Direct(RecoverableGenerator):
        """Benchmark-only direct producer with complete next-yield state."""

        def __init__(self, size=512, *, seed=1729):
            import numpy as np
            self.rng = np.random.default_rng(seed)
            self.index, self.size = 0, size

        def __next__(self):
            if self.index == self.size:
                raise StopIteration
            value = self.rng.random(8, dtype="float32")
            value[0] = self.index
            self.index += 1
            return value

        def capture_state(self):
            return dict(index=self.index, rng=self.rng.bit_generator.state)

        def restore_state(self, state):
            self.index = state["index"]
            self.rng.bit_generator.state = state["rng"]

    return Direct


def work_count_canary() -> dict:
    """Check 20x7 continuous sections and three five-row implicit source opens.

    Uses no optional backend, persistent data, timing assertion or capture hook.
    Returns exact source acquisition/yield counts; invariant violations raise.
    """
    from dryml.core import TensorSpec
    from dryml.data import Epoch, GeneratorDataset
    counts = _canary_counts
    counts.update(opens=0, yields=0)
    source = GeneratorDataset(_canary_factory, 140, spec=TensorSpec("int64", shape=(), backend="numpy"))
    with Epoch(source, len=7).open() as traversal:
        assert sum(len(list(view)) for _, view in traversal.epochs(stop=20)) == 140
    explicit = dict(explicit_opens=counts["opens"], explicit_yields=counts["yields"])
    counts.update(opens=0, yields=0)
    with GeneratorDataset(_canary_factory, 5, spec=source.spec).open() as traversal:
        assert sum(len(list(view)) for _, view in traversal.epochs(stop=3)) == 15
    return {**explicit, "implicit_opens": counts["opens"], "implicit_yields": counts["yields"]}


def _consume(traversal, stop=None):
    return [value for _, view in traversal.epochs(stop=stop) for value in view]


def _timed(call):
    started = perf_counter()
    result = call()
    return result, perf_counter()-started


def _capture(dataset, traversal, directory, *, stop):
    from dryml.data import DatasetState
    state, capture = _timed(traversal.capture_state)
    path, save = _timed(lambda: state.save(directory))
    loaded, load = _timed(lambda: DatasetState.load(path))
    resumed, restore = _timed(lambda: dataset.open(state=loaded))
    with resumed:
        started = perf_counter()
        view = resumed.next_epoch()
        first = next(view)
        first_seconds = perf_counter()-started
        suffix = [first, *view, *_consume(resumed, stop=stop)]
        suffix_seconds = perf_counter()-started
    return suffix, dict(capture_seconds=capture, save_seconds=save, load_seconds=load,
                        restore_seconds=restore, first_resumed_result_seconds=first_seconds,
                        resumed_suffix_seconds=suffix_seconds,
                        serialized_bytes=sum(p.stat().st_size for p in path.rglob("*") if p.is_file()))


def _dataset_window(workload, mode):
    """Validate one Dataset window with inclusive first delivery and tail clocks.

    New-mode first results include epoch enumeration/staging; initial enumerator
    construction belongs to the first latency, while preparation/open are separate.
    Enumeration attempts with no first delivery have their own cost list, never a
    fabricated first result or steady output. The caller times the complete call.
    """
    from itertools import islice
    import numpy as np
    native = mode.endswith("native")
    old = mode == "old"
    indexed = workload.startswith("indexed")
    values = np.arange((4096 if indexed else 512)*8, dtype=np.float32).reshape(-1, 8)
    costs = {}
    steady, first_results, intervals = 0.0, [], []
    delivery = None
    result, saved_suffix = [], None

    def first(view, *, started=None):
        if started is None:
            started = perf_counter()
        value = next(view)
        seconds = perf_counter()-started
        first_results.append(seconds)
        result.append(value)

    def tail(view):
        nonlocal steady
        values, seconds = _timed(lambda: list(view))
        intervals.append(dict(count=len(values), seconds=seconds))
        result.extend(values)
        steady += seconds

    started = perf_counter()
    if native:
        if not indexed:
            # These modes have no native portable continuation protocol.
            return dict(status="unsupported", reason="native direct/mixed/replay/capture semantics unavailable")
        prepare = perf_counter()-started
        if mode == "optimized-native":
            def consume():
                materialized = (values * np.float32(.5)).reshape(128, 32, 8)
                return [batch for _ in range(8) for batch in materialized]
            result, delivery = _timed(consume)
            steady = None
        else:
            length = 16 if workload == "indexed-explicit" else 128
            for _ in range(8):
                for low in range(0, 4096, length*32):
                    value, seconds = _timed(lambda: values[low:low+32]*np.float32(.5))
                    result.append(value)
                    first_results.append(seconds)
                    tail_values, seconds = _timed(lambda: [values[i:i+32]*np.float32(.5)
                        for i in range(low+32, low+length*32, 32)])
                    result.extend(tail_values)
                    intervals.append(dict(count=len(tail_values), seconds=seconds))
                    steady += seconds
        open_seconds = None
    else:
        from dryml.data import ArrayDataset, GeneratorDataset, Shuffle, Zip
        from dryml.core import TensorSpec
        if old and workload not in ("indexed-implicit",):
            return dict(status="unsupported", reason="pinned baseline has no traversal sections/continuation API")
        if indexed:
            dataset = _indexed_recipe(values)
            if workload == "indexed-explicit":
                from dryml.data import Epoch, Repeat
                dataset = Epoch(Repeat(dataset, 8), len=16)
        elif workload == "buffered":
            dataset = Shuffle(ArrayDataset(values), buffer_size=32, seed=1729)
        else:
            from dryml.data import Epoch
            spec = TensorSpec("float32", shape=(8,), backend="numpy")
            factory = _direct_class() if workload == "direct" else _ordinary
            generator = GeneratorDataset(factory, spec=spec, seed=1729, seed_aware=True)
            if workload == "mixed":
                direct = GeneratorDataset(_direct_class(), spec=spec, seed=1729, seed_aware=True)
                generator = Zip(ArrayDataset(values), generator, direct)
            dataset = Epoch(generator, len=16)
        prepared = dataset.prepare()
        prepare = perf_counter()-started
        if old:
            open_seconds = 0.0
            for epoch in range(8):
                cursor, seconds = _timed(lambda: prepared.iterator(epoch=epoch))
                open_seconds += seconds
                with cursor:
                    first(cursor)
                    tail(cursor)
        else:
            opened = perf_counter()
            traversal = prepared.open()
            open_seconds = perf_counter()-opened
            with traversal:
                traversal.enable_replay_recovery_if_needed()
                with TemporaryDirectory(prefix="traversal-cut-") as directory:
                    stop = 8 if workload == "indexed-implicit" else 1 if workload == "buffered" else None
                    epochs = None
                    no_result_seconds = []
                    while True:
                        section_started = perf_counter()
                        if epochs is None:
                            epochs = iter(traversal.epochs(stop=stop))
                        try:
                            _, view = next(epochs)
                        except StopIteration:
                            no_result_seconds.append(perf_counter()-section_started)
                            break
                        try:
                            first(view, started=section_started)
                        except StopIteration:
                            no_result_seconds.append(perf_counter()-section_started)
                            continue
                        if workload == "buffered":
                            tail(islice(view, 128))
                        if not indexed and len(result) == 129:
                            saved_suffix, costs = _capture(prepared, traversal, Path(directory)/"cut", stop=stop)
                        tail(view)
                    if not indexed:
                        _assert_tree_equal(saved_suffix, result[129:])
                    costs["section_first_result_seconds"] = first_results
                    costs["epoch_enumeration_without_result_seconds"] = no_result_seconds
                    _, costs["final_capture_seconds"] = _timed(traversal.capture_state)
    if indexed:
        assert len(result) == 1024
        np.testing.assert_array_equal(np.concatenate(result), np.tile(values*.5, (8, 1)))
        work = 1024
    else:
        assert len(result) == 512
        if workload == "buffered":
            np.testing.assert_array_equal(np.sort(np.array(result)[:, 0]), values[:, 0])
        else:
            ids = [int(v[1][0] if workload == "mixed" else v[0]) for v in result]
            assert ids == list(range(512))
            if workload == "mixed":
                for index, v in enumerate(result):
                    np.testing.assert_array_equal(v[0], values[index])
                    np.testing.assert_array_equal(v[1], v[2])
        work = 512
    domain = _expected_domain(workload, mode, "steady")
    if domain is not None:
        assert sum(i["count"] for i in intervals) == domain["count"]
    return dict(status="ok", work=work, steady_seconds=steady, timing_domain=domain,
                prepare_seconds=prepare, open_seconds=open_seconds,
                first_result_seconds=first_results[0] if first_results else None,
                section_first_result_seconds=first_results,
                delivery_seconds=delivery,
                delivery_domain={**_dataset_domain(workload, "end_to_end"), "name": "full-vectorized-dataset-delivery-v1"} if delivery is not None else None,
                steady_blocker="vectorized whole-source transform precedes batch materialization; isolating post-first work would change the optimized path" if delivery is not None else None,
                delivery_intervals=intervals, costs=costs,
                effective_source_seed=(_source_seed() if workload in ("ordinary", "direct", "mixed") else
                                       1729 if workload == "buffered" else None),
                semantic_difference="vectorized map before batching" if mode == "optimized-native" else None)


def _source_seed():
    from dryml.data.source import _epoch_seed
    return _epoch_seed(1729, 0, 1)


def _assert_tree_equal(left, right):
    import numpy as np
    if isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_tree_equal(a, b)
    else:
        np.testing.assert_array_equal(left, right)


def _jax_init(key):
    import jax.numpy as jnp
    del key
    return {"kernel": jnp.zeros((16, 1), dtype=jnp.float32)}, {}


def _jax_apply(parameters, mutable, rng, x, training):
    del training
    return x @ parameters["kernel"], mutable, rng


def _jax_loss(predictions, targets):
    import jax.numpy as jnp
    return jnp.mean((predictions-targets) ** 2)


def jax_init_factory():
    """Build an importable initializer ``(key) -> (float32 16x1 kernel, {})``.

    The key is unused: the bias-free dense kernel is identically zero. Building
    this callable has no native side effects; invoking it allocates JAX arrays.
    """
    return _jax_init


def jax_apply_factory():
    """Build the pure functional dense apply callable for the benchmark Model.

    Accepts parameters, mutable state, RNG, feature batch and training boolean;
    returns predictions and unchanged mutable/RNG state. Shape errors propagate
    from native matrix multiplication; no owner state is installed here.
    """
    return _jax_apply


def jax_loss_factory():
    """Build ``(predictions, targets) -> scalar mean squared error`` for JAX.

    The returned callable uses float32 input arithmetic, has no state effects,
    and propagates native incompatible-shape/dtype errors.
    """
    return _jax_loss


class _JaxDriver:
    """Untimed builder/runner seam shared by fixed windows and tiny unit probes."""

    def __init__(self, mode, x, y, *, batch=32, sections=4):
        if mode not in MODES or len(x) != len(y) or len(x) % (batch*sections):
            raise ValueError("JAX driver requires a known mode and equal complete sections")
        self.mode, self.x, self.y = mode, x, y
        self.batch, self.sections = batch, sections
        self.updates = len(x)//batch
        self.section_updates = self.updates//sections
        self.delivery_ids, self.exposures = [], []
        self.completed_sections = []
        self.exp = None
        if mode.endswith("native"):
            import jax
            import optax
            self.parameters = jax.numpy.zeros((16, 1), dtype=jax.numpy.float32)
            transformation = optax.sgd(.001)
            self.slots = transformation.init(self.parameters)
            # Constant-rate momentum-free Optax SGD has no mutable slot leaves.
            assert not jax.tree_util.tree_leaves(self.slots)
            self.count = jax.numpy.asarray(0, dtype=jax.numpy.int32)
            self.native_x, self.native_y = jax.numpy.asarray(x), jax.numpy.asarray(y)

            def update(carry, pair):
                parameters, slots, count = carry
                a, b = pair
                loss, grads = jax.value_and_grad(lambda p: _jax_loss(a @ p, b))(parameters)
                delta, slots = transformation.update(grads, slots, parameters)
                return (optax.apply_updates(parameters, delta), slots, count+1), (loss, count+1)

            self.update = jax.jit(update)
            self.scan = jax.jit(lambda carry, pairs: jax.lax.scan(update, carry, pairs))
        else:
            from dryml import F
            from dryml.core import TensorSpec
            from dryml.data import ArrayDataset, Batch
            from dryml.models import Experiment
            from dryml.models.jax import Model, Optimizer, Training
            # A script worker still authors factories by their stable import path,
            # never by __main__, closures, or baseline-owned test fixtures.
            root = str(Path(__file__).resolve().parents[2])
            if root not in sys.path:
                sys.path.insert(0, root)
            from tests.tools import benchmark_dataset_traversal as factories
            model = Model(F(factories.jax_init_factory), F(factories.jax_apply_factory),
                          output_spec=TensorSpec("float32", shape=(1,), backend="jax"), seed=1729)
            trainer = Training(optimizer=Optimizer(F("sgd", learning_rate=.001)),
                               loss=F(factories.jax_loss_factory), epochs=1 if mode == "old" else sections, verbose=0)
            data = Batch(ArrayDataset((x, y)), batch)
            if mode == "new":
                from dryml.data import Epoch
                data = Epoch(data, len=self.section_updates)
            else:
                high = self.section_updates*batch
                data = Batch(ArrayDataset((x[:high], y[:high])), batch)
            self.exp = Experiment(model, trainer, train_data=data)

    def weights(self):
        import numpy as np
        return np.asarray(self.exp.model.parameters["kernel"] if self.exp is not None else self.parameters).copy()

    def train(self, checkpoint, cadence):
        clock = getattr(self, "clock", None)
        if self.exp is not None:
            exp = self.exp

            def report():
                step = exp.state.step
                if clock is not None and step % 32 in (1, 0):
                    clock.report(step)
                sequence = exp.state.input_sequence if self.mode == "new" else step
                self.delivery_ids.append(sequence-1)
                self.exposures.append(exp.state.examples_seen)
                assert (step, exp.state.examples_seen) == (len(self.delivery_ids), step*self.batch)
                assert exp.state.epoch == step//self.section_updates
                if step % self.section_updates == 0:
                    self.completed_sections.append(exp.state.epoch)
                if cadence and step % cadence == 0:
                    checkpoint(step)

            losses = []
            if self.mode == "old":
                from dryml.data import ArrayDataset, Batch
                for section in range(self.sections):
                    low = section*self.section_updates*self.batch
                    high = low+self.section_updates*self.batch
                    exp.train_data = Batch(ArrayDataset((self.x[low:high], self.y[low:high])), self.batch)
                    losses.extend(exp.train_fn(exp, callbacks=(report,)))
            else:
                losses = exp.train_fn(exp, callbacks=(report,))
            return losses
        import numpy as np
        losses = []
        carry = self.parameters, self.slots, self.count
        if self.mode == "optimized-native":
            for section in range(self.sections):
                low = section*self.section_updates*self.batch
                high = low+self.section_updates*self.batch
                pairs = (self.native_x[low:high].reshape(self.section_updates, self.batch, 16),
                         self.native_y[low:high].reshape(self.section_updates, self.batch, 1))
                carry, (values, counts) = self.scan(carry, pairs)
                self.parameters, self.slots, self.count = carry
                # Materializing reports synchronizes the actual section's work.
                losses.extend(np.asarray(values).tolist())
                self.delivery_ids.extend((np.asarray(counts)-1).tolist())
                self.exposures.extend((np.asarray(counts)*self.batch).tolist())
                step = int(self.count)
                if clock is not None:
                    clock.report(step)
                self.completed_sections.append(step//self.section_updates)
                if cadence and step % cadence == 0:
                    checkpoint(step)
        else:
            # Ordinary native slice batching, one JIT update and scalar report
            # per batch. No Dataset, conversion bridge, or DRYML state owner.
            for offset in range(0, len(self.x), self.batch):
                pair = self.native_x[offset:offset+self.batch], self.native_y[offset:offset+self.batch]
                carry, (loss, count) = self.update(carry, pair)
                self.parameters, self.slots, self.count = carry
                losses.append(float(loss))
                step = int(count)
                if clock is not None and step % 32 in (1, 0):
                    clock.report(step)
                self.delivery_ids.append(offset//self.batch)
                self.exposures.append(step*self.batch)
                if step % self.section_updates == 0:
                    self.completed_sections.append(step//self.section_updates)
                if cadence and step % cadence == 0:
                    checkpoint(step)
        return losses

    def save_input(self, path, step):
        import numpy as np
        if self.mode == "new":
            state = self.exp.state
            record = state._input_provider.capture() if state._input_provider else state.dataset_state
            assert record is not None and record.position.sequence == step
            record.save(path)
        else:
            np.savez(path.with_suffix(".npz"), x=self.x, y=self.y,
                     next_row=np.int64(step*self.batch), next_batch=np.int64(step),
                     section_updates=np.int64(self.section_updates))

    def validate(self, losses, expected):
        import numpy as np
        assert len(losses) == self.updates and np.isfinite(losses).all()
        assert self.delivery_ids == list(range(self.updates))
        assert self.exposures == list(range(self.batch, len(self.x)+1, self.batch))
        assert self.completed_sections == list(range(1, self.sections+1))
        np.testing.assert_allclose(self.weights(), expected, rtol=5e-5, atol=1e-6)
        if self.exp is not None:
            state = self.exp.state
            assert (state.step, state.examples_seen, state.epoch) == (self.updates, len(self.x), self.sections)
        else:
            assert int(self.count) == self.updates
        return dict(delivery_ids=self.delivery_ids, updates=self.updates, examples_seen=len(self.x),
                    sections=self.sections, completed_sections=self.completed_sections,
                    weight_checksum=float(np.sum(self.weights(), dtype=np.float64)),
                    expected_weight_checksum=float(np.sum(expected, dtype=np.float64)))


def _jax_training_window(mode, cadence):
    import numpy as np
    rng = np.random.default_rng(1729)
    x = rng.normal(0, .25, (4096, 16)).astype("float32")
    y = (x.sum(axis=1, keepdims=True)*np.float32(.125)).astype("float32")
    expected = np.zeros((16, 1), np.float32)
    prefixes = []
    for offset in range(0, 4096, 32):
        a, b = x[offset:offset+32], y[offset:offset+32]
        expected -= np.float32(.001)*(np.float32(2/32)*a.T @ (a @ expected-b))
        prefixes.append(expected.copy())
    costs = dict(checkpoint_seconds=[], checkpoint_steps=[], checkpoint_bytes=[], final_capture_seconds=None)
    with TemporaryDirectory(prefix="traversal-jax-") as directory:
        started = perf_counter()
        driver = _JaxDriver(mode, x, y)
        prepare = perf_counter()-started
        driver.clock = _TrainingClock(grouped=mode == "optimized-native")
        snapshots = []

        def save(step, name):
            assert step == (driver.exp.state.step if driver.exp is not None else int(driver.count))
            assert driver.delivery_ids[-1] == step-1 and driver.exposures[-1] == step*32
            path = Path(directory)/name
            path.mkdir()
            driver.save_input(path/"input", step)
            weights = driver.weights()
            # Constant-rate SGD has no slots; this count is accepted updates,
            # not an invented Optax slot counter.
            np.savez(path/"model-sgd.npz", weights=weights, step=np.int64(step),
                     examples_seen=np.int64(step*32))
            snapshots.append((step, weights))
            return path

        def checkpoint(step):
            path, seconds = _timed(lambda: save(step, str(step)))
            costs["checkpoint_seconds"].append(seconds)
            costs["checkpoint_steps"].append(step)
            costs["checkpoint_bytes"].append(sum(p.stat().st_size for p in path.rglob("*") if p.is_file()))

        invocation_started = perf_counter()
        losses = driver.train(checkpoint, cadence)
        timing = driver.clock.finish(invocation_started, perf_counter())
        facts = driver.validate(losses, expected)
        path, costs["final_save_seconds"] = _timed(lambda: save(128, "final"))
        costs["final_save_bytes"] = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
        for step, weights in snapshots:
            np.testing.assert_allclose(weights, prefixes[step-1], rtol=5e-5, atol=1e-6)
        assert [step for step, _ in snapshots] == ([32, 64, 96, 128] if cadence else []) + [128]
    return dict(status="ok", work=128, **timing, prepare_seconds=prepare,
                costs=costs, work_facts=facts,
                cold_import_seconds=None, compile_seconds=None, open_seconds=None,
                model=dict(architecture="Dense(16,1,bias=False,zero kernel)", optimizer="SGD(lr=.001,momentum=0)",
                           loss="mean squared error", dtype="float32"),
                timing_limitation="preparation includes imports/setup; first report includes open/setup/first-update compilation; boundary gaps and final tail include postludes and DRYML final input capture; isolated import/open/compile/capture durations are unknown; all remain in invocation/end-to-end costs",
                semantic_difference=("equivalent manual sections; baseline lacks exact input continuation" if mode == "old" else
                    "JIT scan of 32 updates per section; host reports and synchronization grouped at section boundaries" if mode == "optimized-native" else None))


def _training_window(workload, mode, cadence):
    if workload == "jax":
        import importlib.util
        missing = [name for name in ("jax", "optax") if importlib.util.find_spec(name) is None]
        if missing:
            return dict(status="unavailable", reason="real JAX training requires missing " + ", ".join(missing))
        return _jax_training_window(mode, cadence)
    import numpy as np
    native = mode.endswith("native")
    optimized = mode == "optimized-native"
    old = mode == "old"
    if not native:
        # Install DRYML's import guard before optional frameworks, as in tests.sh.
        import dryml  # noqa: F401
    rng = np.random.default_rng(1729)
    x = rng.normal(0, .25, (4096, 16)).astype("float32")
    y = (x.sum(axis=1, keepdims=True) * np.float32(.125)).astype("float32")
    expected = np.zeros((16, 1), np.float32)
    for offset in range(0, 4096, 32):
        a, b = x[offset:offset+32], y[offset:offset+32]
        expected -= np.float32(.001) * (np.float32(2/32) * a.T @ (a @ expected-b))
    costs = dict(checkpoint_seconds=[], checkpoint_steps=[], checkpoint_bytes=[], final_capture_seconds=None)
    clock = _TrainingClock(grouped=optimized and workload == "keras")
    with TemporaryDirectory(prefix="traversal-training-") as directory:
        root = Path(directory)
        prepare_started = perf_counter()
        if workload == "torch":
            import torch
            torch.set_num_threads(1)
            torch.manual_seed(1729)
            if native:
                model = torch.nn.Linear(16, 1, bias=False)
                torch.nn.init.zeros_(model.weight)
                optimizer = torch.optim.SGD(model.parameters(), lr=.001)
                if optimized:
                    model = torch.compile(model)
                tensor_x, tensor_y = torch.from_numpy(x), torch.from_numpy(y)
                if optimized:
                    batches = [(tensor_x[i:i+32], tensor_y[i:i+32]) for i in range(0, 4096, 32)]
                else:
                    batches = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(tensor_x, tensor_y), batch_size=32, num_workers=0)
                prepare = perf_counter()-prepare_started
                def train():
                    losses = []
                    for step, (a, b) in enumerate(batches, 1):
                        optimizer.zero_grad()
                        loss = torch.nn.functional.mse_loss(model(a), b)
                        loss.backward()
                        optimizer.step()
                        losses.append(float(loss.detach()))
                        if step % 32 in (1, 0):
                            clock.report(step)
                        if cadence and step % cadence == 0:
                            checkpoint(step)
                    return losses
                def weights():
                    return next(model.parameters()).detach().numpy().T.copy()
            else:
                from dryml.models.torch import Model, Optimizer, Training
                model = Model(torch.nn.Linear, 16, 1, bias=False)
                torch.nn.init.zeros_(model.obj.weight)
                trainer = Training(optimizer=Optimizer(torch.optim.SGD, target=model, lr=.001),
                                   loss_cls=torch.nn.MSELoss, epochs=1 if old else 4, verbose=0)
                weights = lambda: model.obj.weight.detach().numpy().T.copy()
        else:
            import tensorflow as tf
            if native:
                tf.keras.utils.set_random_seed(1729)
                model = tf.keras.Sequential([tf.keras.Input((16,)), tf.keras.layers.Dense(1, use_bias=False, kernel_initializer="zeros")])
                optimizer = tf.keras.optimizers.SGD(.001)
                dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(32)
                options = tf.data.Options()
                options.threading.private_threadpool_size = 1
                dataset = dataset.with_options(options).prefetch(2)
                prepare = perf_counter()-prepare_started
                if workload == "keras":
                    model.compile(optimizer=optimizer, loss="mse", run_eagerly=False,
                                  jit_compile=False, steps_per_execution=32 if optimized else 1)
                    class Reports(tf.keras.callbacks.Callback):
                        def on_train_batch_end(self, batch, logs=None):
                            step = self.section*32+batch+1
                            if step % 32 in (1, 0):
                                clock.report(step)
                            if cadence:
                                if step % cadence == 0:
                                    assert step == int(optimizer.iterations.numpy())
                                    checkpoint(step)
                    def train():
                        # Four native fit segments preserve section/report boundaries.
                        for section in range(4):
                            reports = Reports()
                            reports.section = section
                            model.fit(dataset.skip(section*32).take(32), epochs=1,
                                      verbose=0, callbacks=[reports])
                        assert int(optimizer.iterations.numpy()) == 128
                else:
                    def update(a, b):
                        with tf.GradientTape() as tape:
                            loss = tf.reduce_mean(tf.square(model(a)-b))
                        optimizer.apply_gradients(zip(tape.gradient(loss, model.trainable_variables), model.trainable_variables))
                        return loss
                    update = tf.function(update, jit_compile=False) if optimized else update
                    def train():
                        for step, (a, b) in enumerate(dataset, 1):
                            float(update(a, b))  # matched existing per-update reporting
                            if step % 32 in (1, 0):
                                clock.report(step)
                            if cadence and step % cadence == 0:
                                checkpoint(step)
                weights = lambda: model.get_weights()[0].copy()
            else:
                model, trainer = _tf_training_recipe(workload, mode)
                weights = lambda: model.obj.get_weights()[0].copy()
        if not native:
            from dryml.data import ArrayDataset, Batch
            from dryml.models import Experiment
            if old:
                dataset = Batch(ArrayDataset((x[:1024], y[:1024])), 32)
            else:
                from dryml.data import Epoch
                dataset = Epoch(Batch(ArrayDataset((x, y)), 32), len=32)
            exp = Experiment(model, trainer, train_data=dataset)
            prepare = perf_counter()-prepare_started
            accepted = []
            def report():
                step = exp.state.step
                if step % 32 in (1, 0):
                    clock.report(step)
                accepted.append(step)
                if cadence and step % cadence == 0:
                    checkpoint(step)
            def train():
                if old:
                    # Explicit equivalent manual sections, not old session support.
                    for section in range(4):
                        low, high = section*1024, (section+1)*1024
                        exp.train_data = Batch(ArrayDataset((x[low:high], y[low:high])), 32)
                        trainer(exp, callbacks=(report,))
                else:
                    trainer(exp, callbacks=(report,))
        def checkpoint(step, *, final=False):
            started = perf_counter()
            path = root / str(len(costs["checkpoint_seconds"]))
            path.mkdir()
            if not native and not old:
                state = exp.state._input_provider.capture() if exp.state._input_provider else exp.state.dataset_state
                state.save(path/"input")
            # Plain SGD has no momentum slots; step plus weights are its full state.
            np.savez(path/"model-sgd.npz", weights=weights(), step=np.int64(step))
            seconds = perf_counter()-started
            size = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
            if final:
                costs.update(final_save_seconds=seconds, final_save_bytes=size)
            else:
                costs["checkpoint_seconds"].append(seconds)
                costs["checkpoint_steps"].append(step)
                costs["checkpoint_bytes"].append(size)
        invocation_started = perf_counter()
        train()
        timing = clock.finish(invocation_started, perf_counter())
        np.testing.assert_allclose(weights(), expected, rtol=5e-5, atol=1e-6)
        if not native:
            assert accepted == list(range(1, 129))
            assert (exp.state.step, exp.state.examples_seen, exp.state.epoch) == (128, 4096, 4)
            if not old:
                assert exp.state.dataset_state is not None
        checkpoint(128, final=True)
    return dict(status="ok", work=128, **timing, prepare_seconds=prepare,
                costs=costs, cold_import_seconds=None, compile_seconds=None, open_seconds=None,
                model=dict(architecture="Dense(16,1,bias=False,zero kernel)", optimizer="SGD(lr=.001,momentum=0)",
                           loss="mean squared error", dtype="float32"),
                timing_limitation="first report and boundary gaps include setup/compile/first updates; final tail includes postludes/final capture; isolated import/open/compile/capture costs are unknown and retained in full invocation/end-to-end",
                semantic_difference=("equivalent manual sections; baseline lacks exact input continuation" if old else
                    "grouped native reports every 32 updates" if optimized and workload == "keras" else None))


def _worker(args):
    import importlib.metadata
    import resource
    workload, mode = args.workload[0], args.mode[0]
    versions = {}
    for name in ("numpy", "torch", "tensorflow", "jax", "optax"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    window = (lambda: _training_window(workload, mode, args.checkpoint_cadence)) if workload in ("torch", "tf", "keras", "jax") else (lambda: _dataset_window(workload, mode))
    if args.instrumentation:
        import tracemalloc
        captures, selections, bridges = [], [], []
        if mode in ("new", "old"):
            from dryml.methods import Method
            original_selection = Method.find_implementation
            def selection(*a, **k):
                selections.append(1)
                return original_selection(*a, **k)
            Method.find_implementation = selection
            if mode == "new":
                from dryml.data import DatasetTraversal
                original_capture = DatasetTraversal.capture_state
                def capture(*a, **k):
                    result, elapsed = _timed(lambda: original_capture(*a, **k))
                    captures.append(elapsed)
                    return result
                DatasetTraversal.capture_state = capture
                if workload == "keras":
                    from dryml.tf.training_data import _KerasInputBridge
                    original_init = _KerasInputBridge.__init__
                    def init(owner, *a, **k):
                        original_init(owner, *a, **k)
                        bridges.append(owner)
                    _KerasInputBridge.__init__ = init
        tracemalloc.start()
        result = window()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        if result["status"] != "ok":
            return result
        return dict(status="ok", instrumentation=True, validated=True,
                    measurement_protocol=MEASUREMENT_PROTOCOL,
                    work=result["work"], peak_python_bytes=peak,
                    peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    source_buffer_copied_bytes=None,
                    pending_high_water_count=max((b.high_water_count for b in bridges), default=None),
                    pending_high_water_bytes=max((b.high_water_bytes for b in bridges), default=None),
                    compile_count=None, selection_count=len(selections) if mode in ("old", "new") else None,
                    full_capture_count=len(captures) if mode == "new" else None,
                    full_capture_seconds=captures, producer_hook_count=None, recovery_only_synchronizations=None,
                    unknown_metrics_reason="not instrumented; never inferred from throughput",
                    costs=result["costs"])
    warmups, windows = [], []
    for index in range(7):
        started = perf_counter()
        result = window()
        elapsed = perf_counter()-started
        if result["status"] != "ok":
            return result
        result["end_to_end_seconds"] = elapsed
        result["end_to_end_domain"] = _expected_domain(workload, mode, "end_to_end")
        if workload in TRAINING:
            result["end_to_end_other_seconds"] = elapsed-result["prepare_seconds"]-result["invocation_seconds"]-result["costs"]["final_save_seconds"]
        if index < 2:
            warmups.append({**result, "elapsed_seconds": elapsed})
        else:
            windows.append(result)
    return dict(**_aggregate_windows(workload, mode, warmups, windows, cadence=args.checkpoint_cadence),
                versions=versions, python=sys.version, cpu=platform.processor(),
                dryml_module=getattr(sys.modules.get("dryml"), "__file__", None),
                threads={key: os.environ.get(key) for key in _thread_env()},
                semantic_difference=result.get("semantic_difference"))


def _aggregate_windows(workload, mode, warmups, windows, *, cadence=0):
    """Aggregate only identical validated domains; no performed/timed count alias."""
    samples = [w["steady_seconds"] for w in windows]
    domain = _expected_domain(workload, mode, "steady")
    totals = [w["end_to_end_seconds"] for w in windows]
    cell = dict(status="ok", validated=True, workload=workload, mode=mode, checkpoint_cadence=cadence,
                measurement_protocol=MEASUREMENT_PROTOCOL,
                work=windows[0]["work"], timing_domain=domain, samples=samples,
                invocation_domain=_expected_domain(workload, mode, "invocation"),
                warmups=warmups, windows=windows, end_to_end_seconds=totals,
                end_to_end_domain=_expected_domain(workload, mode, "end_to_end"))
    if not _valid_windows(cell):
        raise ValueError("invalid or inconsistent per-window timing domain/work count")
    cell.update(_aggregate_metrics(cell))
    return cell


def _thread_env():
    return dict(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                NUMEXPR_NUM_THREADS="1", TF_NUM_INTRAOP_THREADS="1", TF_NUM_INTEROP_THREADS="1")


def _instrumentation_identities(baseline_source, revision, source_dirty):
    """Identify actual archived runtime/current clean revision plus current driver.

    A dirty current source has no reusable identity, including staged/untracked
    changes. Archive fingerprints cover source/resource contents, not generated
    bytecode. Missing/unreadable content yields unknown identity, never equality
    of two unknowns. This assumes cooperating processes keep inputs stable during
    the run; it is not an execution sandbox or a dependency-environment lock.
    """
    identities = dict.fromkeys(MODES)
    try:
        driver = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    except OSError:
        return identities

    def identity(runtime):
        return dict(schema="dryml-benchmark-instrumentation-identity-v1",
                    runtime=runtime, driver_sha256=driver)

    if not source_dirty:
        current = identity(dict(kind="clean-git-revision", revision=revision))
        identities.update({mode: current for mode in MODES if mode != "old"})
    if baseline_source is not None and baseline_source.resolve() != Path(__file__).resolve().parents[2]:
        source = baseline_source/"src/dryml"
        try:
            digest = hashlib.sha256()
            files = sorted(p for p in source.rglob("*") if p.is_file()
                           and "__pycache__" not in p.relative_to(source).parts
                           and p.suffix not in (".pyc", ".pyo"))
            if files:
                for path in files:
                    digest.update(path.relative_to(source).as_posix().encode()+b"\0")
                    digest.update(hashlib.sha256(path.read_bytes()).digest())
                identities["old"] = identity(dict(kind="archived-source-sha256",
                    revision=BASELINE_REVISION, sha256=digest.hexdigest()))
        except OSError:
            pass
    return identities


def _instrumentation_reuse(prior, key, spec, identity):
    """Return reusable cell and diagnostic; never retrofit historical identities."""
    matches = [cell for cell in prior["cells"] if _cell_key(cell) == _cell_key(key)]
    if identity is None:
        return None, "current runtime/driver identity unavailable (dirty source or missing/unreadable content)"
    if len(matches) != 1 or not _valid_evidence(matches[0], spec):
        return None, "prior instrumentation missing, duplicate or invalid"
    cell = matches[0]
    if not cell.get("instrumentation_identity"):
        return None, "prior instrumentation identity missing"
    if cell["instrumentation_identity"] != identity:
        return None, "runtime or driver content identity changed"
    return cell, "matching trusted runtime and driver content identity"


def main(argv: list[str] | None = None) -> int:
    """Emit bounded local JSON evidence; return nonzero unless all gates resolve.

    Fixed selections are opt-in and serial. Results are saved after every cell so
    interruption preserves complete evidence. The output name is reserved with a
    nonblocking sibling FileLock before any worker, through final atomic JSON
    publication. Existing evidence is never overwritten by another invocation;
    contention fails before work. I/O errors and interruptions propagate.
    Required applicability, fixed work, warmups, repeats and instrumentation gate
    closeout; old/new regressions block while native gaps are named follow-ups.
    Confirmation retains prior charges and allows one attempt. Only instrumentation
    with matching runtime/driver identity can be reused; dirty current source or
    missing historical identity forces measurement or an unverified budget result.
    """
    args = parse_args(argv)
    if args.summarize:
        report = json.loads(args.summarize.read_text())
        for cell in report["cells"]:
            if cell["instrumentation"]:
                continue
            print(json.dumps({key: cell.get(key) for key in (
                "workload", "mode", "checkpoint_cadence", "status", "elapsed_seconds", "summary", "reason")}
                | {"error": cell.get("stderr", "").splitlines()[-6:],
                   "costs": cell.get("windows", [{}])[-1].get("costs")}))
        print(json.dumps({"status": report["status"], "elapsed_seconds": report["elapsed_seconds"],
                          "comparisons": report["comparisons"]}))
        return 0
    if args.describe:
        print(json.dumps(dict(profile=fixed_profile(), baseline_revision=BASELINE_REVISION,
                              measurement_protocol=MEASUREMENT_PROTOCOL,
                              workloads=WORKLOADS, modes=MODES,
                              applicability_schema="dryml-dataset-traversal-applicability-v1",
                              applicability=_applicability())))
        return 0
    if args.worker:
        print(json.dumps(_worker(args), allow_nan=False))
        return 0
    if args.output is None:
        raise SystemExit("--output is required for execution")
    if args.baseline_source is not None and not (args.baseline_source/"src/dryml/data").is_dir():
        raise SystemExit("baseline source must contain archived src/dryml/data")
    with _report_writer(args.output) as publish:
        return _execute(args, argv, publish)


def _execute(args, argv, publish):
    started = perf_counter()
    prior_elapsed = 0
    if args.confirm:
        prior = json.loads(args.confirm.read_text())
        if (prior["profile"] != fixed_profile() or prior["baseline_revision"] != BASELINE_REVISION
                or prior.get("measurement_protocol") != MEASUREMENT_PROTOCOL
                or prior["confirmation_count"] != 0):
            raise SystemExit("confirmation requires matching measurement protocol, fixed profile and no prior confirmation")
        prior_elapsed = prior["elapsed_seconds"]
    root = Path(__file__).resolve().parents[2]
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    source_dirty = bool(subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=normal", "--", "src/dryml",
         ":(exclude)src/dryml/graph"], cwd=root, env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"}))
    identities = _instrumentation_identities(args.baseline_source, revision, source_dirty)
    report = dict(schema="dryml-dataset-traversal-benchmark-v2", measurement_protocol=MEASUREMENT_PROTOCOL,
                  profile=fixed_profile(),
                  baseline_revision=args.baseline_revision, revision=revision, source_dirty=source_dirty,
                   instrumentation_identities=identities,
                   instrumentation_identity_policy="clean current source revision or archived source fingerprint, plus current driver SHA-256; unknown identities never reuse",
                   command=[sys.executable, "-m", "tests.tools.benchmark_dataset_traversal", *(argv if argv is not None else sys.argv[1:])],
                   cells=[], comparisons=[], status="unverified", confirmation_count=int(args.confirm is not None),
                   decision_domain="old-new-steady",
                   prior_elapsed_seconds=prior_elapsed,
                   applicability_schema="dryml-dataset-traversal-applicability-v1",
                   applicability=_applicability(),
                   driver_limitations=[])
    specs = {_cell_key(s): s for s in report["applicability"]}
    report["elapsed_seconds"] = prior_elapsed+perf_counter()-started
    publish(report)
    script = str(Path(__file__).resolve())
    for workload in args.workload or WORKLOADS:
        cadences = (0, 32) if workload in ("torch", "tf", "keras", "jax") else (0,)
        for cadence in cadences:
            for mode in args.mode or MODES:
                cell_started = perf_counter()
                for instrumentation in (False, True):
                    remaining = 1200-prior_elapsed-(perf_counter()-started)
                    cell_remaining = 120-(perf_counter()-cell_started)
                    key = dict(workload=workload, mode=mode, checkpoint_cadence=cadence,
                               instrumentation=instrumentation)
                    spec = specs[_cell_key(key)]
                    reused, reuse_reason = None, "fresh instrumentation requested"
                    identity = identities.get(mode)
                    if args.confirm and instrumentation:
                        reused, reuse_reason = _instrumentation_reuse(prior, key, spec, identity)
                    if not spec["required"]:
                        result = dict(status="unsupported", reason=spec["reason"])
                    elif reused is not None:
                        result = {**reused, "reused_instrumentation": True, "confirmation_elapsed_seconds": 0}
                    elif remaining <= 0:
                        result = dict(status="budget-exhausted")
                    elif cell_remaining <= 0:
                        result = dict(status="timeout", reason="cell startup/timing exhausted instrumentation deadline")
                    elif mode == "old" and args.baseline_source is None:
                        result = dict(status="unavailable", reason="pinned baseline source not supplied")
                    else:
                        root = args.baseline_source if mode == "old" else Path(__file__).resolve().parents[2]
                        environment = {**os.environ, **_thread_env(), "CUDA_VISIBLE_DEVICES": "-1",
                                       "JAX_PLATFORMS": "cpu", "TMPDIR": "/tmp/dryml",
                                       "PYTHONPATH": str(root/"src")}
                        command = [sys.executable, script, "--worker", "--workload", workload,
                                   "--mode", mode, "--checkpoint-cadence", str(cadence)]
                        if instrumentation:
                            command.append("--instrumentation")
                        result = run_worker(command, timeout=min(cell_remaining, remaining), env=environment)
                    if instrumentation:
                        result = {**result, "instrumentation_identity": identity,
                                  "reused_instrumentation": reused is not None and spec["required"],
                                  "instrumentation_reuse_reason": reuse_reason}
                    report["cells"].append({**result, **key})
                    report["elapsed_seconds"] = prior_elapsed+perf_counter()-started
                    publish(report)
    _qualify_report(report)
    report["elapsed_seconds"] = prior_elapsed+perf_counter()-started
    if report["elapsed_seconds"] > 1200:
        report["status"] = "unverified"
        report["blockers"].append(dict(reason="total budget exhausted during aggregation"))
    publish(report)
    print(json.dumps(dict(status=report["status"], output=str(args.output), elapsed_seconds=report["elapsed_seconds"])))
    return 0 if report["status"] == "no-regression-observed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
