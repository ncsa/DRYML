"""Synthetic protocol checks and tiny untimed JAX unit qualification, no matrix."""

import json
import io
import os
from pathlib import Path
import subprocess
import sys
import tarfile
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


def test_jax_driver_missing_dependencies_stay_required(monkeypatch):
    """Missing real prerequisites cannot exempt required JAX comparisons."""
    import importlib.util
    monkeypatch.setattr(importlib.util, "find_spec", lambda _: None)
    result = benchmark._training_window("jax", "new", 0)
    assert result["status"] == "unavailable"
    assert "missing" in result["reason"]
    assert all(s["required"] for s in benchmark._applicability() if s["workload"] == "jax")


@pytest.mark.parametrize("mode", benchmark.MODES)
@pytest.mark.parametrize("cadence", [0, 32])
def test_installed_jax_routes_every_required_cell_to_driver(monkeypatch, mode, cadence):
    """Routing proof only; the sentinel executes no comparison or native math."""
    import importlib.util
    monkeypatch.setattr(importlib.util, "find_spec", lambda _: object())
    sentinel = {"driver_called": (mode, cadence)}
    monkeypatch.setattr(benchmark, "_jax_training_window", lambda m, c: sentinel, raising=False)
    assert benchmark._training_window("jax", mode, cadence) is sentinel


@pytest.mark.parametrize("mode", benchmark.MODES)
def test_jax_driver_tiny_real_unit_qualification(tmp_path, monkeypatch, mode):
    """Four rows/four genuine SGD updates qualify builders, never fixed timings."""
    pytest.importorskip("jax")
    pytest.importorskip("optax")
    if mode == "old":
        # The old manual-section recipe belongs to the actual pinned API, not
        # a current TrainState with its stronger exact-input identity contract.
        archive = subprocess.check_output(["git", "archive", benchmark.BASELINE_REVISION, "src/dryml"],
                                          cwd=Path(benchmark.__file__).resolve().parents[2])
        baseline = tmp_path/"baseline"
        baseline.mkdir()
        with tarfile.open(fileobj=io.BytesIO(archive)) as source:
            source.extractall(baseline, filter="data")
        script = '''
import sys
from pathlib import Path
import dryml
import pytest
assert Path(dryml.__file__).resolve().is_relative_to(Path(sys.argv[1]).resolve())
from tests.tools.test_benchmark_dataset_traversal import _qualify_tiny_jax_driver
with pytest.MonkeyPatch.context() as patch:
    _qualify_tiny_jax_driver(Path(sys.argv[2]), patch, "old")
'''
        environment = {**os.environ, "PYTHONPATH": str(baseline/"src"),
                       "CUDA_VISIBLE_DEVICES": "-1", "JAX_PLATFORMS": "cpu"}
        result = subprocess.run([sys.executable, "-c", script, str(baseline), str(tmp_path)],
                                env=environment, capture_output=True, text=True, timeout=45)
        assert result.returncode == 0, result.stdout + result.stderr
    else:
        _qualify_tiny_jax_driver(tmp_path, monkeypatch, mode)


def _qualify_tiny_jax_driver(tmp_path, monkeypatch, mode):
    """Untimed math/continuation probe usable in a pinned-source subprocess."""
    import builtins
    import numpy as np
    jax = pytest.importorskip("jax")
    pytest.importorskip("optax")
    rng = np.random.default_rng(1729)
    x = rng.normal(0, .25, (4, 16)).astype("float32")
    y = x.sum(axis=1, keepdims=True)*np.float32(.125)
    expected = np.zeros((16, 1), np.float32)
    prefixes, expected_losses = [], []
    for a, b in zip(x, y):
        expected_losses.append(np.mean((a @ expected-b)**2))
        expected -= np.float32(.001)*np.float32(2)*a[:, None] @ (a @ expected-b)[None, :]
        prefixes.append(expected.copy())

    if mode.endswith("native"):
        original = builtins.__import__

        def native_only(name, *args, **kwargs):
            assert name != "dryml" and not name.startswith("dryml."), "native path used DRYML"
            return original(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", native_only)
    driver = benchmark._JaxDriver(mode, x, y, batch=1, sections=2)
    np.testing.assert_array_equal(driver.weights(), np.zeros((16, 1), np.float32))
    if driver.exp is not None:
        from dryml.core import FactorySpec
        assert isinstance(driver.exp.model.init_fn, FactorySpec)
        np.testing.assert_array_equal(jax.random.key_data(driver.exp.model.rng),
                                      jax.random.key_data(jax.random.split(jax.random.key(1729))[1]))
        assert dict(driver.exp.train_fn.optimizer.factory.kwargs) == {"learning_rate": .001}
        assert driver.exp.train_fn.epochs == (1 if mode == "old" else 2)
        assert driver.exp.train_fn.verbose == 0
        predictions, mutable, key = driver.exp.model._apply(
            driver.exp.model.parameters, {}, driver.exp.model.rng, jax.numpy.asarray(x), True)
        np.testing.assert_array_equal(predictions, np.zeros((4, 1), np.float32))
        assert mutable == {} and key is driver.exp.model.rng
    snapshots = []

    def checkpoint(step):
        snapshots.append((step, driver.weights()))
        driver.save_input(tmp_path/str(step), step)

    losses = driver.train(checkpoint, 2)
    facts = driver.validate(losses, expected)
    assert facts["delivery_ids"] == [0, 1, 2, 3]
    assert (facts["updates"], facts["examples_seen"], facts["sections"]) == (4, 4, 2)
    np.testing.assert_allclose(losses, expected_losses, rtol=5e-5, atol=1e-6)
    assert [step for step, _ in snapshots] == [2, 4]
    for step, weights in snapshots:
        np.testing.assert_allclose(weights, prefixes[step-1], rtol=5e-5, atol=1e-6)
        if mode != "new":
            with np.load(tmp_path/f"{step}.npz") as state:
                assert int(state["next_row"]) == int(state["next_batch"]) == step
                np.testing.assert_array_equal(state["x"], x)
                np.testing.assert_array_equal(state["y"], y)
    if mode == "new":
        from dryml.data import DatasetState
        record = DatasetState.load(tmp_path/"2")
        assert record.position.sequence == 2
        with driver.exp.train_data.prepare().open(state=record) as traversal:
            pairs = benchmark._consume(traversal)
        np.testing.assert_array_equal(np.concatenate([a for a, _ in pairs]), x[2:])
        np.testing.assert_array_equal(np.concatenate([b for _, b in pairs]), y[2:])


def test_jax_factory_defaults_are_real_importable_declarations():
    """Construction-only qualification of the default four-section Model recipe."""
    import numpy as np
    pytest.importorskip("jax")
    pytest.importorskip("optax")
    # Minimum four default batches; no training invocation or timing window.
    driver = benchmark._JaxDriver("new", np.zeros((128, 16), np.float32), np.zeros((128, 1), np.float32))
    assert (driver.batch, driver.sections, driver.exp.train_fn.epochs) == (32, 4, 4)
    assert driver.weights().dtype == np.float32 and driver.weights().shape == (16, 1)
    from dryml import F
    assert F(benchmark.jax_init_factory).build() is benchmark._jax_init
    assert F(benchmark.jax_apply_factory).build() is benchmark._jax_apply
    assert F(benchmark.jax_loss_factory).build() is benchmark._jax_loss


def test_jax_script_builder_has_stable_factories_outside_repository_cwd(tmp_path):
    """Construction-only child emulates script loading without a comparison cell."""
    pytest.importorskip("jax")
    pytest.importorskip("optax")
    script = '''
import runpy
import sys
import numpy as np
module = runpy.run_path(sys.argv[1], run_name="unit_qualification")
driver = module["_JaxDriver"]("new", np.zeros((4, 16), np.float32), np.zeros((4, 1), np.float32), batch=1, sections=2)
from tests.tools import benchmark_dataset_traversal as canonical
assert driver.exp.model._init is canonical._jax_init
assert driver.exp.model._apply is canonical._jax_apply
assert driver.exp.train_fn._loss_callable() is canonical._jax_loss
'''
    environment = {**os.environ, "PYTHONPATH": str(Path(benchmark.__file__).resolve().parents[2]/"src"),
                   "CUDA_VISIBLE_DEVICES": "-1", "JAX_PLATFORMS": "cpu"}
    result = subprocess.run([sys.executable, "-c", script, benchmark.__file__], cwd=tmp_path,
                            env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("mode", benchmark.MODES)
@pytest.mark.parametrize("cadence", [0, 32])
def test_jax_window_checkpoint_harness_with_numerical_protocol(monkeypatch, mode, cadence):
    """128-update scheduling/serialization proof, not real Optax qualification.

    The builder is replaced by a NumPy protocol fixture. Native loop/reporting
    code and the fixed window harness run with a synthetic clock; no real JAX
    comparison, compilation, warmup or throughput observation occurs.
    """
    import numpy as np
    snapshots, drivers = [], []
    original_driver = benchmark._JaxDriver

    class NumericalDriver(original_driver):
        def __init__(self, mode, x, y):
            self.mode, self.x, self.y = mode, x, y
            self.batch, self.sections, self.section_updates, self.updates = 32, 4, 32, 128
            self.exp = None
            self.parameters, self.slots, self.count = np.zeros((16, 1), np.float32), (), np.int32(0)
            self.native_x, self.native_y = x, y
            self.delivery_ids, self.exposures, self.completed_sections = [], [], []
            drivers.append(self)

            def update(carry, pair):
                weights, slots, step = carry
                a, b = pair
                loss = np.mean((a @ weights-b)**2)
                weights = weights-np.float32(.001)*(np.float32(2/32)*a.T @ (a @ weights-b))
                step = np.int32(step+1)
                return (weights, slots, step), (loss, step)

            def scan(carry, pairs):
                losses, counts = [], []
                for a, b in zip(*pairs):
                    carry, (loss, count) = update(carry, (a, b))
                    losses.append(loss)
                    counts.append(count)
                return carry, (np.array(losses), np.array(counts))

            self.update, self.scan = update, scan

        def save_input(self, path, step):
            # Only the new record's save API is simulated. Tiny real tests above
            # qualify its actual capture/save/load/suffix behavior independently.
            np.savez(path.with_suffix(".npz"), next_batch=np.int64(step), x=self.x, y=self.y)

    original_save = np.savez

    def save(path, **values):
        if "weights" in values:
            driver = drivers[-1]
            snapshots.append((int(values["step"]), int(driver.count), values["weights"].copy()))
            assert int(values["examples_seen"]) == int(values["step"])*32
        original_save(path, **values)

    monkeypatch.setattr(benchmark, "_JaxDriver", NumericalDriver)
    monkeypatch.setattr(np, "savez", save)
    monkeypatch.setattr(benchmark, "_timed", lambda call: (call(), 1.0))
    monkeypatch.setattr(benchmark, "perf_counter", lambda: 1.0)
    result = benchmark._jax_training_window(mode, cadence)
    assert result["work"] == 128 and result["status"] == "ok"
    assert result["work_facts"]["delivery_ids"] == list(range(128))
    assert result["work_facts"]["completed_sections"] == [1, 2, 3, 4]
    assert result["work_facts"]["examples_seen"] == 4096
    assert len(result["costs"]["checkpoint_seconds"]) == (4 if cadence else 0)
    assert result["costs"]["final_save_seconds"] == 1
    assert all(size > 0 for size in result["costs"]["checkpoint_bytes"])
    assert [step for step, _, _ in snapshots] == ([32, 64, 96, 128] if cadence else []) + [128]
    assert all(step == count for step, count, _ in snapshots)
    driver = drivers[-1]
    expected = np.zeros((16, 1), np.float32)
    for step in range(1, 129):
        a, b = driver.x[(step-1)*32:step*32], driver.y[(step-1)*32:step*32]
        expected -= np.float32(.001)*(np.float32(2/32)*a.T @ (a @ expected-b))
        for saved_step, _, weights in snapshots:
            if saved_step == step:
                np.testing.assert_allclose(weights, expected, rtol=5e-5, atol=1e-6)
    assert result["work_facts"]["expected_weight_checksum"] == float(np.sum(expected, dtype=np.float64))


@pytest.mark.parametrize("defect", ["losses", "delivery", "exposure", "sections", "weights", "count"])
def test_jax_work_validation_rejects_wrong_useful_work(defect):
    """Pure NumPy facts cannot validate duplicate work, wrong accounting or math."""
    import numpy as np
    driver = benchmark._JaxDriver.__new__(benchmark._JaxDriver)
    driver.x, driver.batch, driver.sections, driver.updates = np.zeros((4, 16)), 1, 2, 4
    driver.exp, driver.parameters, driver.count = None, np.zeros((16, 1), np.float32), 4
    driver.delivery_ids, driver.exposures, driver.completed_sections = [0, 1, 2, 3], [1, 2, 3, 4], [1, 2]
    losses = [0.] * 4
    if defect == "losses":
        losses.pop()
    elif defect == "delivery":
        driver.delivery_ids[-1] = 2
    elif defect == "exposure":
        driver.exposures[-1] = 3
    elif defect == "sections":
        driver.completed_sections.pop()
    elif defect == "weights":
        driver.parameters[0, 0] = .1
    else:
        driver.count = 3
    with pytest.raises(AssertionError):
        driver.validate(losses, np.zeros((16, 1), np.float32))


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
