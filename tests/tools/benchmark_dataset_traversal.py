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
"""

from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
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


def _valid(cell):
    samples = cell.get("samples", [])
    return (cell.get("status") == "ok" and cell.get("validated") is True
            and type(cell.get("work")) is int and cell["work"] > 0
            and isinstance(samples, list) and len(samples) == 5
            and all(type(x) in (int, float) and math.isfinite(x) and x > 0 for x in samples))


def _valid_instrumentation(cell):
    return (cell.get("status") == "ok" and cell.get("instrumentation") is True and cell.get("validated") is True
            and type(cell.get("work")) is int and cell["work"] > 0
            and type(cell.get("peak_python_bytes")) is int and cell["peak_python_bytes"] >= 0)


def _positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _valid_protocol(cell):
    warmups, windows, totals = (cell.get(key, []) for key in ("warmups", "windows", "end_to_end_seconds"))
    return (_valid(cell) and all(isinstance(part, list) for part in (warmups, windows, totals))
            and len(warmups) == _PROFILE["warmups"]
            and all(isinstance(w, dict) and type(w.get("work")) is int
                    and w["work"] == cell["work"] and _positive(w.get("elapsed_seconds")) for w in warmups)
            and len(windows) == len(totals) == _PROFILE["repetitions"]
            and all(isinstance(w, dict) and w.get("status") == "ok" and type(w.get("work")) is int
                    and w["work"] == cell["work"]
                    and w.get("steady_seconds") == sample and _positive(total) and total >= sample
                    for w, sample, total in zip(windows, cell["samples"], totals)))


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
    return (cell.get("work") == spec["expected_work"]
            and (_valid_instrumentation(cell) if spec["instrumentation"] else _valid_protocol(cell)))


def _qualify_report(report):
    """Rebuild applicability, blockers and comparisons without executing any work."""
    schema = _applicability()
    report.update(applicability_schema="dryml-dataset-traversal-applicability-v1",
                  applicability=schema, comparisons=[], blockers=[], status="unverified")
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
                checkpoint_cadence=spec["checkpoint_cadence"], purpose=purpose,
                followup=f"{spec['workload']}/{mode}/cadence-{spec['checkpoint_cadence']}-native-gap" if purpose == "native-gap" else None,
                **comparison))
    elapsed = report.get("elapsed_seconds")
    if (report.get("profile") != fixed_profile() or report.get("baseline_revision") != BASELINE_REVISION
            or type(elapsed) not in (int, float) or not math.isfinite(elapsed) or not 0 <= elapsed <= 1200):
        report["blockers"].append(dict(reason="invalid fixed profile, revision or exhausted total budget"))
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


def compare(old: dict, new: dict) -> dict:
    """Compare validated five-window useful work with a three-max-MAD triage band.

    Returns no ratio for failed, unsupported, nonfinite, incomplete or unequal work.
    Positive differences beyond the band are regressions. Differences within a
    nonzero band are unresolved noise, not equivalence or a universal allowance.
    """
    if not (_valid(old) and _valid(new)) or old["work"] != new["work"]:
        return dict(status="unverified", ratio=None)
    a, b = _summary(old["samples"]), _summary(new["samples"])
    noise = 3 * max(a["mad_seconds"], b["mad_seconds"])
    delta = b["median_seconds"] - a["median_seconds"]
    status = "regression" if delta > noise else "noisy" if noise and abs(delta) <= noise else "no-regression-observed"
    return dict(status=status, ratio=b["median_seconds"] / a["median_seconds"],
                noise_seconds=noise, delta_seconds=delta, old=a, new=b)


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
    import numpy as np
    native = mode.endswith("native")
    old = mode == "old"
    indexed = workload.startswith("indexed")
    values = np.arange((4096 if indexed else 512)*8, dtype=np.float32).reshape(-1, 8)
    costs = {}
    started = perf_counter()
    if native:
        if not indexed:
            # These modes have no native portable continuation protocol.
            return dict(status="unsupported", reason="native direct/mixed/replay/capture semantics unavailable")
        prepare = perf_counter()-started
        def consume():
            if mode == "optimized-native":
                batches = (values * np.float32(.5)).reshape(128, 32, 8)
                return [batch for _ in range(8) for batch in batches]
            return [values[i:i+32] * np.float32(.5) for _ in range(8) for i in range(0, 4096, 32)]
        result, steady = _timed(consume)
        first_result = None
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
            def consume():
                return [v for epoch in range(8) for v in prepared.iterator(epoch=epoch)]
            result, steady = _timed(consume)
            first_result = None
        else:
            opened = perf_counter()
            with prepared.open() as traversal:
                traversal.enable_replay_recovery_if_needed()
                view = traversal.next_epoch()
                first = next(view)
                first_result = perf_counter()-opened
                prefix = [first]
                if not indexed:
                    for _ in range(128):
                        try:
                            prefix.append(next(view))
                        except StopIteration:
                            view = traversal.next_epoch()
                            prefix.append(next(view))
                prefix_seconds = perf_counter()-opened
                with TemporaryDirectory(prefix="traversal-cut-") as directory:
                    if not indexed:
                        saved_suffix, costs = _capture(prepared, traversal, Path(directory)/"cut",
                                                       stop=1 if workload == "buffered" else None)
                    def finish():
                        tail = list(view)
                        if workload != "buffered":
                            tail.extend(_consume(traversal, stop=8 if workload == "indexed-implicit" else None))
                        return tail
                    suffix, tail_seconds = _timed(finish)
                    if not indexed:
                        _assert_tree_equal(saved_suffix, suffix)
                    result = prefix + suffix
                    steady = prefix_seconds + tail_seconds
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
    return dict(status="ok", work=work, steady_seconds=steady, prepare_seconds=prepare,
                first_result_seconds=first_result, costs=costs,
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


def _training_window(workload, mode, cadence):
    # Real JAX training qualification requires Optax even if pure JAX is present.
    if workload == "jax":
        import importlib.util
        if importlib.util.find_spec("optax") is None:
            return dict(status="unavailable", driver_status="unimplemented",
                        reason="real Optax/JAX qualification requires missing optax; fixed-profile driver also unimplemented")
        return dict(status="unsupported", driver_status="unimplemented",
                    reason="required JAX fixed-profile training driver unimplemented even with Optax installed")
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
    costs = dict(checkpoint_seconds=[], checkpoint_bytes=[], final_capture_seconds=None)
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
            tf.keras.utils.set_random_seed(1729)
            if native:
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
                            if cadence:
                                step = int(optimizer.iterations.numpy())
                                if step % cadence == 0:
                                    checkpoint(step)
                    def train():
                        # Four native fit segments preserve section/report boundaries.
                        for section in range(4):
                            model.fit(dataset.skip(section*32).take(32), epochs=1,
                                      verbose=0, callbacks=[Reports()])
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
                            if cadence and step % cadence == 0:
                                checkpoint(step)
                weights = lambda: model.get_weights()[0].copy()
            else:
                from dryml.models.tf import BasicTraining, Optimizer, Sequential, Training
                model = Sequential(layer_defs=_tf_layer_defs())
                trainer_type = BasicTraining if workload == "keras" else Training
                trainer = trainer_type(optimizer=Optimizer(tf.keras.optimizers.SGD, learning_rate=.001),
                                       epochs=1 if old else 4, verbose=0,
                                       compile_kwargs={"run_eagerly": False, "jit_compile": False})
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
                accepted.append(exp.state.step)
                if cadence and exp.state.step % cadence == 0:
                    checkpoint(exp.state.step)
            def train():
                if old:
                    # Explicit equivalent manual sections, not old session support.
                    for section in range(4):
                        low, high = section*1024, (section+1)*1024
                        exp.train_data = Batch(ArrayDataset((x[low:high], y[low:high])), 32)
                        trainer(exp, callbacks=(report,))
                else:
                    trainer(exp, callbacks=(report,))
        def checkpoint(step):
            started = perf_counter()
            path = root / str(len(costs["checkpoint_seconds"]))
            path.mkdir()
            if not native and not old:
                state = exp.state._input_provider.capture() if exp.state._input_provider else exp.state.dataset_state
                state.save(path/"input")
            # Plain SGD has no momentum slots; step plus weights are its full state.
            np.savez(path/"model-sgd.npz", weights=weights(), step=np.int64(step))
            costs["checkpoint_seconds"].append(perf_counter()-started)
            costs["checkpoint_bytes"].append(sum(p.stat().st_size for p in path.rglob("*") if p.is_file()))
        _, steady = _timed(train)
        np.testing.assert_allclose(weights(), expected, rtol=5e-5, atol=1e-6)
        if not native:
            assert accepted == list(range(1, 129))
            assert (exp.state.step, exp.state.examples_seen, exp.state.epoch) == (128, 4096, 4)
            if not old:
                assert exp.state.dataset_state is not None
        _, costs["final_save_seconds"] = _timed(lambda: checkpoint(128))
    return dict(status="ok", work=128, steady_seconds=steady, prepare_seconds=prepare,
                first_result_seconds=None, costs=costs,
                model=dict(architecture="Dense(16,1,bias=False,zero kernel)", optimizer="SGD(lr=.001,momentum=0)",
                           loss="mean squared error", dtype="float32"),
                timing_limitation="training invocation includes fit/compile setup; warmed kernel-only time is not isolated",
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
    warmups, windows, e2e = [], [], []
    for index in range(7):
        started = perf_counter()
        result = window()
        elapsed = perf_counter()-started
        if result["status"] != "ok":
            return result
        if index < 2:
            warmups.append(dict(work=result["work"], elapsed_seconds=elapsed))
        else:
            windows.append(result)
            e2e.append(elapsed)
    samples = [window["steady_seconds"] for window in windows]
    return dict(status="ok", validated=True, work=result["work"], samples=samples,
                warmups=warmups, windows=windows, end_to_end_seconds=e2e,
                summary=_summary(samples), throughput_per_second=result["work"]/median(samples),
                versions=versions, python=sys.version, cpu=platform.processor(),
                dryml_module=getattr(sys.modules.get("dryml"), "__file__", None),
                threads={key: os.environ.get(key) for key in _thread_env()},
                semantic_difference=result.get("semantic_difference"))


def _thread_env():
    return dict(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                NUMEXPR_NUM_THREADS="1", TF_NUM_INTRAOP_THREADS="1", TF_NUM_INTEROP_THREADS="1")


def main(argv: list[str] | None = None) -> int:
    """Emit bounded local JSON evidence; return nonzero unless all gates resolve.

    Fixed selections are opt-in and serial. Results are saved after every cell so
    interruption preserves complete evidence. The output name is reserved with a
    nonblocking sibling FileLock before any worker, through final atomic JSON
    publication. Existing evidence is never overwritten by another invocation;
    contention fails before work. I/O errors and interruptions propagate.
    Required applicability, fixed work, warmups, repeats and instrumentation gate
    closeout; old/new regressions block while native gaps are named follow-ups.
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
                or prior["confirmation_count"] != 0):
            raise SystemExit("confirmation requires matching fixed profile and no prior confirmation")
        prior_elapsed = prior["elapsed_seconds"]
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    source_dirty = subprocess.run(["git", "diff", "--quiet", "--", "src/dryml"]).returncode != 0
    report = dict(schema="dryml-dataset-traversal-benchmark-v1", profile=fixed_profile(),
                  baseline_revision=args.baseline_revision, revision=revision, source_dirty=source_dirty,
                   command=[sys.executable, "-m", "tests.tools.benchmark_dataset_traversal", *(argv if argv is not None else sys.argv[1:])],
                   cells=[], comparisons=[], status="unverified", confirmation_count=int(args.confirm is not None),
                   prior_elapsed_seconds=prior_elapsed,
                   applicability_schema="dryml-dataset-traversal-applicability-v1",
                   applicability=_applicability(),
                   driver_limitations=["Required JAX driver is unimplemented even if Optax is installed; missing Optax is an additional blocker."])
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
                    reused = next((c for c in prior["cells"] if all(c.get(k) == v for k, v in key.items())
                                   and _valid_evidence(c, spec)), None) if args.confirm and instrumentation else None
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
