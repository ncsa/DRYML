"""Backend-free qualification of the fixed traversal measurement protocol."""

import json
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from tests.tools import benchmark_dataset_traversal as benchmark


def test_fixed_profile_cannot_silently_reduce_work_or_repeats():
    profile = benchmark.fixed_profile()
    assert profile["seed"] == 1729
    assert (profile["warmups"], profile["repetitions"]) == (2, 5)
    assert (profile["cell_seconds"], profile["total_seconds"]) == (120, 1200)
    assert profile["indexed"] == dict(rows=4096, features=8, batch=32,
                                      passes=8, sections=64, section_length=16, batches=1024)
    assert profile["sources"] == dict(outputs=512, features=8, section_length=16)
    assert profile["training"] == dict(rows=4096, features=16, targets=1, batch=32,
                                       updates=128, sections=4, checkpoint_cadences=[0, 32])
    assert profile["buffered"] == dict(capacity=32, capture_after=129, outputs=512)
    assert profile["keras"] == dict(capacity=8, prefetch=2)
    profile["indexed"]["rows"] = 1
    assert benchmark.fixed_profile()["indexed"]["rows"] == 4096


def test_result_comparison_requires_complete_validated_fixed_work():
    def cell(samples, **kwargs):
        return dict(status="ok", validated=True, work=1024, samples=samples, **kwargs)
    old = cell([2, 2.1, 1.9, 2, 2])
    new = cell([3, 3.1, 2.9, 3, 3])
    result = benchmark.compare(old, new)
    assert result["status"] == "regression"
    assert result["ratio"] == 1.5
    assert result["noise_seconds"] == 0
    for invalid in (
        dict(status="unsupported"), dict(status="timeout"),
        cell([1] * 4), {**new, "validated": False}, {**new, "work": 1000},
        cell([float("nan")] * 5), cell([0] * 5),
    ):
        result = benchmark.compare(old, invalid)
        assert result["status"] == "unverified"
        assert result["ratio"] is None
    assert benchmark.compare(old, cell([2, 2.2, 1.8, 2.1, 1.9]))["status"] == "noisy"


def test_deadline_includes_startup_and_failed_cells_and_kills_worker():
    result = benchmark.run_worker([sys.executable, "-c", "import time; time.sleep(30)"],
                                  timeout=.1, env=None)
    assert result["status"] == "timeout"
    assert result["elapsed_seconds"] >= .1
    assert result["worker_reaped"] is True


def test_worker_rejects_failure_and_malformed_json_without_timing_pass():
    for code in ("raise RuntimeError('synthetic')", "print('{}')", "print('not json')",
                 "print('{\"status\":\"ok\",\"instrumentation\":true}')"):
        result = benchmark.run_worker([sys.executable, "-c", code], timeout=5, env=None)
        assert result["status"] == "failed"
        assert "samples" not in result


def test_cli_profile_is_backend_free_and_rejects_profile_mutation():
    script = '''
import importlib.abc
import runpy
import sys
class RejectNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'tensorflow', 'torch', 'jax', 'jaxlib', 'optax'}:
            raise AssertionError('synthetic CLI imported ' + fullname)
sys.meta_path.insert(0, RejectNative())
sys.argv = ['benchmark', '--describe']
runpy.run_module('tests.tools.benchmark_dataset_traversal', run_name='__main__')
'''
    result = subprocess.run([sys.executable, "-c", script],
                            check=True, capture_output=True, text=True, timeout=10)
    assert json.loads(result.stdout)["profile"] == benchmark.fixed_profile()
    for args in (("--repetitions", "1"), ("--seed", "3"), ("--cell-seconds", "121")):
        with pytest.raises(SystemExit):
            benchmark.parse_args(list(args))


def test_work_count_canary_has_one_source_and_no_steady_capture(monkeypatch):
    from dryml.data import DatasetTraversal
    monkeypatch.setattr(DatasetTraversal, "capture_state",
                        lambda _: pytest.fail("steady work captured state"))
    result = benchmark.work_count_canary()
    assert result == {"explicit_opens": 1, "explicit_yields": 140,
                      "implicit_opens": 3, "implicit_yields": 15}


def test_indexed_recipe_transform_matches_native_operation():
    """Tiny numerical canary for Scale's mean/std rather than multiplier API."""
    import numpy as np
    values = np.arange(16, dtype=np.float32).reshape(2, 8)
    with benchmark._indexed_recipe(values).prepare().iterator() as cursor:
        (batch,) = list(cursor)
    np.testing.assert_array_equal(batch, values * np.float32(.5))


def test_tf_layer_declarations_are_factories_for_the_sequential_owner():
    """Qualify inert declarations only; no fake native training qualification."""
    from dryml.models.tf import Sequential
    definitions = benchmark._tf_layer_defs()
    recipe = Sequential.defn(layer_defs=definitions)
    assert recipe is not None
    # These pure declaration probes execute no backend or model constructor.
    assert definitions[0].args == ()
    assert dict(definitions[0].kwargs) == {"shape": (16,)}
    assert definitions[1].args == (1,)
    assert dict(definitions[1].kwargs) == {"use_bias": False, "kernel_initializer": "zeros"}


def test_interrupted_worker_is_killed_and_reaped(monkeypatch):
    original = subprocess.Popen.communicate
    children = []
    def interrupt_once(child, *args, **kwargs):
        if not children:
            children.append(child)
            raise KeyboardInterrupt
        return original(child, *args, **kwargs)
    monkeypatch.setattr(subprocess.Popen, "communicate", interrupt_once)
    with pytest.raises(KeyboardInterrupt):
        benchmark.run_worker([sys.executable, "-c", "import time; time.sleep(30)"],
                             timeout=5, env=None)
    assert children[0].poll() is not None


def test_confirmation_charges_prior_failures_and_never_launches_past_budget(tmp_path, monkeypatch):
    prior = tmp_path / "prior.json"
    output = tmp_path / "result.json"
    record = dict(profile=benchmark.fixed_profile(), baseline_revision=benchmark.BASELINE_REVISION,
                  elapsed_seconds=1200, confirmation_count=0, cells=[])
    prior.write_text(json.dumps(record))
    monkeypatch.setattr(benchmark, "run_worker", lambda *a, **k: pytest.fail("exhausted budget launched child"))
    assert benchmark.main(["--confirm", str(prior), "--output", str(output),
                           "--workload", "indexed-implicit", "--mode", "old"]) == 1
    result = json.loads(output.read_text())
    assert result["status"] == "unverified"
    assert result["confirmation_count"] == 1
    assert all(cell["status"] == "budget-exhausted" for cell in result["cells"])
    assert not result["comparisons"]


def test_second_confirmation_is_rejected_before_worker_launch(tmp_path, monkeypatch):
    prior = tmp_path / "prior.json"
    prior.write_text(json.dumps(dict(profile=benchmark.fixed_profile(),
        baseline_revision=benchmark.BASELINE_REVISION, confirmation_count=1)))
    monkeypatch.setattr(benchmark, "run_worker", lambda *a, **k: pytest.fail("second confirmation launched child"))
    with pytest.raises(SystemExit, match="no prior confirmation"):
        benchmark.main(["--confirm", str(prior), "--output", str(tmp_path / "result.json")])


def test_capture_probe_validates_full_multi_section_suffix(tmp_path):
    """Tiny real IO proof for the tool's first-result/suffix cost split."""
    import numpy as np
    from dryml.data import ArrayDataset, Epoch
    prepared = Epoch(ArrayDataset(np.arange(9)), len=4).prepare()
    with prepared.open() as traversal:
        view = traversal.next_epoch()
        assert [next(view), next(view)] == [0, 1]
        suffix, costs = benchmark._capture(prepared, traversal, tmp_path / "cut", stop=None)
        assert list(map(int, suffix)) == list(range(2, 9))
        assert [next(view), next(view)] == [2, 3]  # probe did not advance its owner
    assert costs["serialized_bytes"] > 0
    assert costs["resumed_suffix_seconds"] >= costs["first_resumed_result_seconds"] >= 0


def _synthetic_cell(workload, instrumentation=False, seconds=1):
    work = 1024 if workload.startswith("indexed") else 128 if workload in ("torch", "tf", "keras", "jax") else 512
    if instrumentation:
        return dict(status="ok", instrumentation=True, validated=True,
                    work=work, peak_python_bytes=64)
    return dict(status="ok", validated=True, work=work, samples=[seconds] * 5,
                warmups=[dict(work=work, elapsed_seconds=seconds)] * 2,
                windows=[dict(status="ok", work=work, steady_seconds=seconds)] * 5,
                end_to_end_seconds=[seconds] * 5)


def _synthetic_matrix_worker(command, **kwargs):
    workload = command[command.index("--workload") + 1]
    mode = command[command.index("--mode") + 1]
    unsupported = (mode == "old" and workload in ("indexed-explicit", "ordinary", "direct", "mixed", "buffered")
                   or mode.endswith("native") and workload in ("ordinary", "direct", "mixed", "buffered"))
    if unsupported:
        return dict(status="unsupported", reason="known unavailable semantics")
    # Large native gaps are intentional here; only new/pre-change gates parity.
    return _synthetic_cell(workload, "--instrumentation" in command,
                           seconds=.01 if mode.endswith("native") else 1)


def _synthetic_run(tmp_path, monkeypatch, worker=_synthetic_matrix_worker):
    baseline = tmp_path / "baseline"
    (baseline / "src/dryml/data").mkdir(parents=True)
    output = tmp_path / "report.json"
    monkeypatch.setattr(benchmark, "run_worker", worker)
    status = benchmark.main(["--output", str(output), "--baseline-source", str(baseline)])
    return status, json.loads(output.read_text())


def test_complete_applicable_evidence_passes_with_labeled_native_gaps(tmp_path, monkeypatch):
    status, report = _synthetic_run(tmp_path, monkeypatch)
    assert status == 0
    assert report["status"] == "no-regression-observed"
    assert report["applicability_schema"] == "dryml-dataset-traversal-applicability-v1"
    assert all(c["required"] for c in report["applicability"] if c["workload"] == "jax")
    gaps = [c for c in report["comparisons"] if c["purpose"] == "native-gap" and c["ratio"]]
    assert gaps and all(c["ratio"] == 100 and c["followup"] for c in gaps)
    unsupported = [c for c in report["comparisons"] if c["status"] == "not-applicable"]
    assert unsupported and all(c["ratio"] is None and c["reason"] for c in unsupported)


@pytest.mark.parametrize("defect", ["missing", "duplicate", "wrong-work", "warmup", "repeat", "window",
                                   "instrumentation", "unsupported", "unavailable", "failed", "timeout",
                                   "unvalidated", "regression", "noise"])
def test_required_evidence_never_waived_for_bad_results(tmp_path, monkeypatch, defect):
    _, report = _synthetic_run(tmp_path, monkeypatch)
    target = next(c for c in report["cells"] if c["workload"] == "jax" and c["mode"] == "new"
                  and c["checkpoint_cadence"] == 0 and not c["instrumentation"])
    if defect == "missing":
        report["cells"].remove(target)
    elif defect == "duplicate":
        report["cells"].append(dict(target))
    elif defect == "wrong-work":
        # Equal old/new work is still wrong for the fixed workload.
        for c in report["cells"]:
            if c["workload"] == "jax":
                c["work"] = 127
    elif defect == "warmup":
        target["warmups"] = target["warmups"][:1]
    elif defect == "repeat":
        target["samples"] = target["samples"][:4]
    elif defect == "window":
        target["windows"] = [{**w, "work": 127} for w in target["windows"]]
    elif defect == "instrumentation":
        report["cells"] = [c for c in report["cells"] if not (
            c["workload"] == "jax" and c["mode"] == "new" and c["instrumentation"])]
    elif defect == "unvalidated":
        target["validated"] = False
    elif defect in ("regression", "noise"):
        samples = [2] * 5 if defect == "regression" else [.8, .9, 1, 1.1, 1.2]
        target.update(samples=samples, windows=[dict(status="ok", work=128, steady_seconds=s) for s in samples],
                      end_to_end_seconds=samples)
    else:
        target["status"] = defect
    benchmark._qualify_report(report)
    assert report["status"] == "unverified"
    assert report["blockers"]


def test_output_ownership_rejects_thread_and_process_before_any_worker(tmp_path, monkeypatch):
    output = tmp_path / "owned.json"
    entered = [threading.Event(), threading.Event()]
    proceed = [threading.Event(), threading.Event()]
    calls = []

    def worker(command, **kwargs):
        calls.append(command)
        index = len(calls) - 1
        entered[index].set()
        assert proceed[index].wait(15)
        return _synthetic_matrix_worker(command, **kwargs)

    monkeypatch.setattr(benchmark, "run_worker", worker)
    arguments = ["--output", str(output), "--workload", "indexed-implicit", "--mode", "new"]
    code = """
import sys
from tests.tools import benchmark_dataset_traversal as b
b.run_worker = lambda *a, **k: (_ for _ in ()).throw(AssertionError('contender launched worker'))
b.main(sys.argv[1:])
"""
    with ThreadPoolExecutor(max_workers=1) as executor:
        owner = executor.submit(benchmark.main, arguments)
        try:
            # Probe before any worker result and again during instrumentation.
            for count in (1, 2):
                assert entered[count-1].wait(15)
                before = output.read_bytes() if output.exists() else None
                with pytest.raises(SystemExit, match="owned"):
                    benchmark.main(arguments)
                contender = subprocess.run([sys.executable, "-c", code, *arguments],
                                           text=True, capture_output=True, timeout=15)
                assert contender.returncode != 0 and "owned" in contender.stderr
                assert len(calls) == count
                assert (output.read_bytes() if output.exists() else None) == before
                proceed[count-1].set()
            assert owner.result(timeout=15) == 1  # deliberately partial evidence
        finally:
            for event in proceed:
                event.set()
    with pytest.raises(SystemExit, match="overwrite"):
        benchmark.main(arguments)
    assert len(calls) == 2


@pytest.mark.parametrize("fault", ["short-write", "disk-full", "interrupt", "fsync", "replace"])
def test_progress_publication_preserves_prior_complete_json(tmp_path, monkeypatch, fault):
    output = tmp_path / "report.json"
    foreign = tmp_path / ".report.json.staging-foreign"
    foreign.write_text("not ours")
    with benchmark._report_writer(output) as publish:
        publish(dict(cells=["complete"]))
        prior = output.read_bytes()
        with monkeypatch.context() as patch:
            if fault in ("short-write", "disk-full", "interrupt"):
                original = benchmark.os.fdopen

                class FaultyFile:
                    def __init__(self, *args, **kwargs):
                        self.file = original(*args, **kwargs)

                    def close(self):
                        self.file.close()

                    def write(self, raw):
                        self.file.write(raw[:3])
                        if fault == "short-write":
                            return 3
                        raise OSError("synthetic disk full") if fault == "disk-full" else KeyboardInterrupt("synthetic interruption")

                patch.setattr(benchmark.os, "fdopen", FaultyFile)
            else:
                def fail(*args, **kwargs):
                    raise OSError("synthetic " + fault)
                patch.setattr(benchmark.os, fault, fail)
            with pytest.raises(KeyboardInterrupt if fault == "interrupt" else OSError):
                publish(dict(cells=["complete", "next"]))
        assert output.read_bytes() == prior
        assert json.loads(output.read_text()) == dict(cells=["complete"])
        assert list(tmp_path.glob(".report.json.staging-*")) == [foreign]
        publish(dict(cells=["complete", "recovered"]))
    assert json.loads(output.read_text())["cells"][-1] == "recovered"


@pytest.mark.parametrize("mode", ["matched-native", "optimized-native"])
def test_native_keras_checkpoints_bind_cumulative_step_to_weight_snapshot(monkeypatch, mode):
    """Execute the real 128-update driver against a numerical, backend-free API."""
    import types
    import numpy as np
    snapshots = []

    class Scalar:
        value = 0

        def numpy(self):
            return self.value

    optimizer = types.SimpleNamespace(iterations=Scalar())

    class Dataset:
        def __init__(self, x, y):
            self.pairs = [(x[i:i+32], y[i:i+32]) for i in range(0, len(x), 32)]

        def batch(self, count):
            assert count == 32
            return self

        def with_options(self, options):
            return self

        def prefetch(self, count):
            assert count == 2
            return self

        def skip(self, count):
            result = Dataset.__new__(Dataset)
            result.pairs = self.pairs[count:]
            return result

        def take(self, count):
            result = Dataset.__new__(Dataset)
            result.pairs = self.pairs[:count]
            return result

    class Model:
        def __init__(self, layers):
            self.weights = np.zeros((16, 1), np.float32)

        def compile(self, **kwargs):
            self.group = kwargs["steps_per_execution"]

        def fit(self, dataset, *, epochs, verbose, callbacks):
            for batch, (x, y) in enumerate(dataset.pairs):
                self.weights -= np.float32(.001) * (np.float32(2/32) * x.T @ (x @ self.weights-y))
                optimizer.iterations.value += 1
                if (batch + 1) % self.group == 0:
                    for callback in callbacks:
                        callback.on_train_batch_end(batch)

        def get_weights(self):
            return [self.weights]

    tf = types.SimpleNamespace(
        keras=types.SimpleNamespace(utils=types.SimpleNamespace(set_random_seed=lambda _: None),
            Sequential=Model, Input=lambda _: None, layers=types.SimpleNamespace(Dense=lambda *a, **k: None),
            optimizers=types.SimpleNamespace(SGD=lambda _: optimizer),
            callbacks=types.SimpleNamespace(Callback=object)),
        data=types.SimpleNamespace(Dataset=types.SimpleNamespace(from_tensor_slices=lambda pair: Dataset(*pair)),
            Options=lambda: types.SimpleNamespace(threading=types.SimpleNamespace())))
    monkeypatch.setitem(sys.modules, "tensorflow", tf)
    original = np.savez

    def save(path, **values):
        snapshots.append((int(values["step"]), optimizer.iterations.value, values["weights"].copy()))
        original(path, **values)

    monkeypatch.setattr(np, "savez", save)
    result = benchmark._training_window("keras", mode, 32)
    assert result["work"] == 128
    assert [s[0] for s in snapshots] == [32, 64, 96, 128, 128]
    assert all(step == updates for step, updates, _ in snapshots)
    rng = np.random.default_rng(1729)
    x = rng.normal(0, .25, (4096, 16)).astype("float32")
    y = x.sum(axis=1, keepdims=True) * np.float32(.125)
    weights = np.zeros((16, 1), np.float32)
    for step in range(1, 129):
        a, b = x[(step-1)*32:step*32], y[(step-1)*32:step*32]
        weights -= np.float32(.001) * (np.float32(2/32) * a.T @ (a @ weights-b))
        if step % 32 == 0:
            np.testing.assert_array_equal(snapshots[step//32-1][2], weights)


@pytest.mark.parametrize("key,value", [("warmups", None), ("warmups", [1, 2]),
                                      ("windows", None), ("samples", None),
                                      ("end_to_end_seconds", None)])
def test_malformed_required_evidence_is_a_blocker_not_an_aggregation_crash(tmp_path, monkeypatch, key, value):
    _, report = _synthetic_run(tmp_path, monkeypatch)
    cell = next(c for c in report["cells"] if c["mode"] == "new" and not c["instrumentation"])
    cell[key] = value
    benchmark._qualify_report(report)
    assert report["status"] == "unverified"
    assert any(b["reason"] == "required evidence incomplete or invalid" for b in report["blockers"])


def test_unimplemented_jax_driver_stays_required_regardless_of_dependency_discovery(monkeypatch):
    """Discovery branches only: no fake Optax/native qualification or updates."""
    import importlib.util
    for discovered in (None, object()):
        monkeypatch.setattr(importlib.util, "find_spec", lambda _: discovered)
        result = benchmark._training_window("jax", "new", 0)
        assert result["status"] == ("unavailable" if discovered is None else "unsupported")
        assert result["driver_status"] == "unimplemented"
        assert "unimplemented" in result["reason"]
        assert all(s["required"] for s in benchmark._applicability() if s["workload"] == "jax")


def test_worker_interruption_retains_complete_progress_and_releases_stable_lock(tmp_path, monkeypatch):
    output = tmp_path / "interrupted.json"
    calls = []

    def worker(command, **kwargs):
        calls.append(command)
        if len(calls) == 2:
            raise KeyboardInterrupt("synthetic instrumentation interruption")
        return _synthetic_matrix_worker(command, **kwargs)

    monkeypatch.setattr(benchmark, "run_worker", worker)
    with pytest.raises(KeyboardInterrupt, match="synthetic instrumentation"):
        benchmark.main(["--output", str(output), "--workload", "indexed-implicit", "--mode", "new"])
    report = json.loads(output.read_text())
    assert report["status"] == "unverified"
    assert len(report["cells"]) == 1 and report["cells"][0]["validated"]
    from dryml.locking import FileLock
    path = output.with_name(f".{output.name}.benchmark.lock")
    inode = path.stat().st_ino
    lock = FileLock(path)
    assert lock.acquire(blocking=False)
    lock.release()
    assert path.stat().st_ino == inode
    assert not list(tmp_path.glob(".interrupted.json.staging-*"))


def test_publication_primary_error_survives_stage_cleanup_and_lock_release_failure(tmp_path, monkeypatch):
    from pathlib import Path
    from dryml.locking import FileLock
    output = tmp_path / "report.json"
    original_release = FileLock.release
    original_unlink = Path.unlink
    failure = OSError("primary fsync failure")

    def fsync(fd):
        raise failure

    def unlink(path, *args, **kwargs):
        if ".staging-" in path.name:
            raise OSError("secondary cleanup failure")
        return original_unlink(path, *args, **kwargs)

    def release(lock):
        original_release(lock)
        raise OSError("secondary lock release failure")

    with pytest.raises(OSError) as caught:
        with benchmark._report_writer(output) as publish:
            publish(dict(complete=True))
            prior = output.read_bytes()
            monkeypatch.setattr(benchmark.os, "fsync", fsync)
            monkeypatch.setattr(Path, "unlink", unlink)
            monkeypatch.setattr(FileLock, "release", release)
            publish(dict(complete=True, next=True))
    assert caught.value is failure
    assert "secondary cleanup failure" in " ".join(caught.value.__notes__)
    assert "secondary lock release failure" in " ".join(caught.value.__notes__)
    assert output.read_bytes() == prior


def test_publication_primary_error_survives_file_close_failure(tmp_path, monkeypatch):
    output = tmp_path / "report.json"
    original = benchmark.os.fdopen
    failure = OSError("primary flush failure")

    class FaultyFile:
        def __init__(self, *args, **kwargs):
            self.file = original(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self.file, name)

        def flush(self):
            raise failure

        def close(self):
            self.file.close()
            raise OSError("secondary close failure")

    with benchmark._report_writer(output) as publish:
        publish(dict(complete=True))
        prior = output.read_bytes()
        monkeypatch.setattr(benchmark.os, "fdopen", FaultyFile)
        with pytest.raises(OSError) as caught:
            publish(dict(complete=True, next=True))
        assert caught.value is failure
        assert "secondary close failure" in " ".join(failure.__notes__)
        assert output.read_bytes() == prior
        assert not list(tmp_path.glob(".report.json.staging-*"))
