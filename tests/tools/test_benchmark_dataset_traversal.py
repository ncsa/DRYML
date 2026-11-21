"""Synthetic protocol checks and tiny untimed TF/JAX qualification, no matrix."""

import json
import io
import copy
from itertools import count
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from tests.tools import benchmark_dataset_traversal as benchmark


def test_invocation_only_training_evidence_cannot_qualify_as_steady():
    """Legacy invocation time includes cold/setup/postlude costs, not steady work."""
    cell = dict(status="ok", validated=True, work=128, samples=[1]*5,
                warmups=[dict(work=128, elapsed_seconds=1)]*2,
                windows=[dict(status="ok", work=128, steady_seconds=1)]*5,
                end_to_end_seconds=[1]*5)
    assert not benchmark._valid_protocol(cell)


def test_phantom_128_update_throughput_and_incompatible_domains_refuse_ratios():
    """Equal performed work does not prove equal timed work or timing boundaries."""
    old = _synthetic_cell("jax")
    new = _synthetic_cell("jax")
    old["timing_domain"] = dict(name="accepted-report-within-section-v1", unit="updates", count=124)
    new["timing_domain"] = dict(name="full-training-invocation-v1", unit="updates", count=128)
    assert benchmark.compare(old, new)["ratio"] is None
    new["timing_domain"] = {**old["timing_domain"], "count": 128}
    assert benchmark.compare(old, new)["ratio"] is None


def test_report_clock_retains_between_update_overhead_and_exposes_all_gaps(monkeypatch):
    ticks = iter([10, 41, 110, 141, 210, 241, 310, 341])
    monkeypatch.setattr(benchmark, "perf_counter", lambda: next(ticks))
    clock = benchmark._TrainingClock()
    for step in range(1, 129):
        clock.report(step)
    result = clock.finish(0, 400)
    assert result["steady_seconds"] == 124
    assert result["timing_domain"] == benchmark._training_domain()
    assert [i["timed_updates"] for i in result["report_intervals"]] == [31]*4
    assert [(i["first_report_step"], i["last_report_step"]) for i in result["report_intervals"]] == [
        (1, 32), (33, 64), (65, 96), (97, 128)]
    assert result["invocation_seconds"] == 400
    assert result["boundary_costs"] == dict(initial_setup_first_report_seconds=10,
        between_section_postlude_setup_first_report_seconds=[69]*3,
        final_report_to_return_seconds=59, outside_steady_seconds=276)
    with pytest.raises(StopIteration):
        next(ticks)  # exactly eight timestamps; no per-yield clock/hook


def test_timed_numerator_is_124_while_full_cost_domains_keep_128():
    cell = _synthetic_cell("jax", seconds=2)
    assert cell["work"] == 128
    assert cell["timing_domain"]["count"] == 124
    assert cell["throughput_per_second"] == 62
    assert cell["throughput_unit"] == "updates/second"
    assert cell["end_to_end_domain"]["count"] == cell["invocation_domain"]["count"] == 128
    assert cell["end_to_end_throughput_per_second"] == 128/10
    cell["throughput_per_second"] = 128/2
    assert not benchmark._valid_protocol(cell)
    assert benchmark.compare(cell, cell)["ratio"] is None


@pytest.mark.parametrize("workload", ["jax", "keras", "indexed-implicit"])
def test_grouped_optimized_path_has_full_costs_but_no_fabricated_steady_ratio(workload):
    native = _synthetic_cell(workload, mode="optimized-native")
    new = _synthetic_cell(workload)
    assert benchmark._valid_protocol(native)
    assert native["summary"] is native["throughput_per_second"] is native["timing_domain"] is None
    assert benchmark.compare(native, new)["ratio"] is None
    comparison = benchmark.compare(native, new, metric="end_to_end")
    assert comparison["ratio"] == 1
    assert comparison["timing_domain"]["count"] == (1024 if workload.startswith("indexed") else 128)


@pytest.mark.parametrize("defect", ["124-as-128", "127", "interval-count", "boundary", "invocation", "cadence", "final-as-cadence", "historical"])
def test_per_window_timing_domain_and_work_corruption_is_ineligible(defect):
    cell = _synthetic_cell("jax")
    window = cell["windows"][0]
    if defect in ("124-as-128", "127"):
        window["timing_domain"] = {**window["timing_domain"], "count": 128 if defect == "124-as-128" else 127}
    elif defect == "interval-count":
        window["report_intervals"][0]["timed_updates"] = 32
    elif defect == "boundary":
        window["boundary_costs"]["final_report_to_return_seconds"] = 0
    elif defect == "invocation":
        window["invocation_domain"] = benchmark._training_domain()
    elif defect == "cadence":
        cell["checkpoint_cadence"] = 32
    elif defect == "final-as-cadence":
        window["costs"]["checkpoint_steps"] = [128]
        window["costs"]["checkpoint_seconds"] = [1]
    else:
        cell["measurement_protocol"] = "historical-v1"
    assert not benchmark._valid_protocol(cell)
    assert benchmark.compare(cell, cell)["ratio"] is None


def test_total_invocation_costs_are_informational_but_same_domain_steady_regresses(synthetic_report):
    _, report = synthetic_report
    target = next(c for c in report["cells"] if c["workload"] == "jax" and c["mode"] == "new"
                  and c["checkpoint_cadence"] == 0 and not c["instrumentation"])
    for window in target["windows"]:
        window["invocation_seconds"] += 100
        window["end_to_end_seconds"] += 100
        window["boundary_costs"]["final_report_to_return_seconds"] += 100
        window["boundary_costs"]["outside_steady_seconds"] += 100
    target["end_to_end_seconds"] = [w["end_to_end_seconds"] for w in target["windows"]]
    target.update(benchmark._aggregate_metrics(target))
    benchmark._qualify_report(report)
    assert report["status"] == "no-regression-observed"
    assert any(c["purpose"] == "informational-cost-domain" and c["status"] == "regression"
               for c in report["comparisons"])
    target.update(samples=[2]*5, windows=[_synthetic_window("jax", seconds=2) for _ in range(5)],
                  end_to_end_seconds=[10]*5)
    target.update(benchmark._aggregate_metrics(target))
    benchmark._qualify_report(report)
    assert report["status"] == "unverified"
    assert any(c["workload"] == "jax" and c["purpose"] == "old-new-regression"
               and c["status"] == "regression" for c in report["comparisons"])


def test_historical_protocol_cannot_confirm_or_reuse_instrumentation(tmp_path, monkeypatch):
    prior = tmp_path/"prior.json"
    prior.write_text(json.dumps(dict(profile=benchmark.fixed_profile(), baseline_revision=benchmark.BASELINE_REVISION,
                                    measurement_protocol="historical-v1", confirmation_count=0, elapsed_seconds=1, cells=[])))
    before = prior.read_bytes()
    monkeypatch.setattr(benchmark, "run_worker", lambda *a, **k: pytest.fail("historical confirmation launched work"))
    with pytest.raises(SystemExit, match="matching measurement protocol"):
        benchmark.main(["--confirm", str(prior), "--output", str(tmp_path/"new.json")])
    assert prior.read_bytes() == before


@pytest.mark.parametrize("mode", ["old", "new"])
@pytest.mark.parametrize("change", ["runtime", "driver", "historical", "cell-missing", "current-unknown"])
def test_confirmation_remeasures_stale_instrumentation_identity(tmp_path, monkeypatch, mode, change):
    """Synthetic confirmation charges prior work but cannot borrow stale evidence."""
    identity = dict(schema="dryml-benchmark-instrumentation-identity-v1",
                    runtime=dict(revision="current", sha256="runtime-before"),
                    driver_sha256="driver-before")
    current = copy.deepcopy(identity)
    if change == "runtime":
        current["runtime"]["sha256"] = "runtime-after"
    elif change == "driver":
        current["driver_sha256"] = "driver-after"
    elif change == "current-unknown":
        current = None
    timing = _synthetic_cell("indexed-implicit", mode=mode)
    instrument = dict(_synthetic_cell("indexed-implicit", instrumentation=True),
                      workload="indexed-implicit", mode=mode, checkpoint_cadence=0)
    if change not in ("historical", "cell-missing"):
        instrument["instrumentation_identity"] = None if change == "current-unknown" else identity
    record = dict(profile=benchmark.fixed_profile(), baseline_revision=benchmark.BASELINE_REVISION,
                  measurement_protocol=benchmark.MEASUREMENT_PROTOCOL, confirmation_count=0,
                  elapsed_seconds=894.4321190109476, cells=[timing, instrument])
    if change != "historical":
        record["instrumentation_identities"] = {mode: identity}
    prior = tmp_path/"prior.json"
    prior.write_text(json.dumps(record))
    before = prior.read_bytes()
    baseline = tmp_path/"baseline"
    (baseline/"src/dryml/data").mkdir(parents=True)
    calls = []

    def worker(command, **kwargs):
        calls.append(command)
        assert 0 < kwargs["timeout"] <= 305.5678809890524
        return _synthetic_matrix_worker(command, **kwargs)

    monkeypatch.setattr(benchmark, "_instrumentation_identities", lambda *a, **k: {mode: current}, raising=False)
    monkeypatch.setattr(benchmark, "perf_counter", lambda: 0.0)
    monkeypatch.setattr(benchmark, "run_worker", worker)
    output = tmp_path/"confirmation.json"
    assert benchmark.main(["--confirm", str(prior), "--output", str(output),
                           "--baseline-source", str(baseline),
                           "--workload", "indexed-implicit", "--mode", mode]) == 1
    assert len(calls) == 2, "stale instrumentation was reused instead of remeasured"
    assert "--instrumentation" not in calls[0] and "--instrumentation" in calls[1]
    report = json.loads(output.read_text())
    measured = report["cells"][1]
    assert measured["instrumentation_identity"] == current
    assert not measured["reused_instrumentation"]
    assert measured["instrumentation_reuse_reason"] == (
        "current runtime/driver identity unavailable (dirty source or missing/unreadable content)" if current is None
        else "prior instrumentation identity missing" if change in ("historical", "cell-missing")
        else "runtime or driver content identity changed")
    assert report["prior_elapsed_seconds"] == report["elapsed_seconds"] == 894.4321190109476
    assert report["confirmation_count"] == 1 and report["status"] == "unverified"
    assert prior.read_bytes() == before
    with pytest.raises(SystemExit, match="no prior confirmation"):
        benchmark.main(["--confirm", str(output), "--output", str(tmp_path/"second.json")])
    assert len(calls) == 2


@pytest.mark.parametrize("mode", ["old", "new"])
def test_confirmation_reuses_matching_identity_but_always_remeasures_timing(tmp_path, monkeypatch, mode):
    """Identity permits instrumentation reuse only, with unchanged budget charge."""
    identity = dict(schema="dryml-benchmark-instrumentation-identity-v1",
                    runtime=dict(kind="clean-git-revision", revision="same"), driver_sha256="same-driver")
    instrument = dict(_synthetic_cell("indexed-implicit", instrumentation=True),
                      workload="indexed-implicit", mode=mode, checkpoint_cadence=0,
                      instrumentation_identity=identity, selection_count=17)
    prior = tmp_path/"prior.json"
    prior.write_text(json.dumps(dict(profile=benchmark.fixed_profile(), baseline_revision=benchmark.BASELINE_REVISION,
        measurement_protocol=benchmark.MEASUREMENT_PROTOCOL, confirmation_count=0,
        elapsed_seconds=894.4321190109476,
        cells=[_synthetic_cell("indexed-implicit", mode=mode), instrument])))
    baseline = tmp_path/"baseline"
    (baseline/"src/dryml/data").mkdir(parents=True)
    monkeypatch.setattr(benchmark, "_instrumentation_identities", lambda *a, **k: {mode: identity})
    monkeypatch.setattr(benchmark, "perf_counter", lambda: 0.0)
    calls = []

    def worker(command, **kwargs):
        calls.append(command)
        assert "--instrumentation" not in command
        return _synthetic_matrix_worker(command, **kwargs)

    monkeypatch.setattr(benchmark, "run_worker", worker)
    output = tmp_path/"confirmation.json"
    assert benchmark.main(["--confirm", str(prior), "--output", str(output),
        "--baseline-source", str(baseline), "--workload", "indexed-implicit", "--mode", mode]) == 1
    report = json.loads(output.read_text())
    assert len(calls) == 1
    cell = report["cells"][1]
    assert cell["reused_instrumentation"] and cell["selection_count"] == 17
    assert cell["confirmation_elapsed_seconds"] == 0
    assert cell["instrumentation_identity"] == report["instrumentation_identities"][mode] == identity
    assert cell["instrumentation_reuse_reason"] == "matching trusted runtime and driver content identity"
    assert report["elapsed_seconds"] == report["prior_elapsed_seconds"] == 894.4321190109476
    assert report["confirmation_count"] == 1 and report["status"] == "unverified"


@pytest.mark.parametrize("identity", [None, {"runtime": "stale", "driver_sha256": "stale"}])
def test_stale_instrumentation_stays_unverified_when_timing_consumes_remaining_budget(tmp_path, monkeypatch, identity):
    """A synthetic clock consumes exactly the residual charge, with no real work."""
    instrument = dict(_synthetic_cell("indexed-implicit", instrumentation=True),
        workload="indexed-implicit", mode="new", checkpoint_cadence=0, instrumentation_identity=identity)
    prior = tmp_path/"prior.json"
    prior.write_text(json.dumps(dict(profile=benchmark.fixed_profile(), baseline_revision=benchmark.BASELINE_REVISION,
        measurement_protocol=benchmark.MEASUREMENT_PROTOCOL, confirmation_count=0,
        elapsed_seconds=894.4321190109476, cells=[instrument])))
    monkeypatch.setattr(benchmark, "_instrumentation_identities", lambda *a, **k: {"new": {"runtime": "current"}})
    now, calls = [0.0], []
    monkeypatch.setattr(benchmark, "perf_counter", lambda: now[0])

    def worker(command, **kwargs):
        calls.append(command)
        now[0] = 305.5678809890524
        return dict(status="timeout")

    monkeypatch.setattr(benchmark, "run_worker", worker)
    output = tmp_path/"confirmation.json"
    assert benchmark.main(["--confirm", str(prior), "--output", str(output),
        "--workload", "indexed-implicit", "--mode", "new"]) == 1
    report = json.loads(output.read_text())
    assert len(calls) == 1
    assert report["cells"][1]["status"] == "budget-exhausted"
    assert not report["cells"][1]["reused_instrumentation"]
    assert report["status"] == "unverified" and report["blockers"]
    assert report["prior_elapsed_seconds"] == 894.4321190109476
    assert report["elapsed_seconds"] == 1200 and report["confirmation_count"] == 1


def test_instrumentation_identity_covers_archive_contents_and_current_driver(tmp_path, monkeypatch):
    """Tiny content fixtures distinguish archived runtime from the current recipe."""
    script = tmp_path/"current/tests/tools/benchmark.py"
    script.parent.mkdir(parents=True)
    script.write_text("recipe-before")
    archive = tmp_path/"archive"
    runtime = archive/"src/dryml/__init__.py"
    runtime.parent.mkdir(parents=True)
    runtime.write_text("runtime-before")
    monkeypatch.setattr(benchmark, "__file__", str(script))

    def identities(*, dirty=False, revision="same-head"):
        return benchmark._instrumentation_identities(archive, revision, dirty)

    before = identities()
    assert before["old"]["runtime"]["kind"] == "archived-source-sha256"
    assert before["old"]["runtime"]["revision"] == benchmark.BASELINE_REVISION
    assert before["new"]["runtime"] == dict(kind="clean-git-revision", revision="same-head")
    assert before["new"] == before["matched-native"] == before["optimized-native"]
    assert identities() == before
    assert identities(dirty=True)["new"] is None
    assert identities(dirty=True)["old"] == before["old"]
    assert identities(revision="other-head")["new"] != before["new"]
    assert identities(revision="other-head")["old"] == before["old"]
    runtime.write_text("runtime-after")
    after_runtime = identities()
    assert after_runtime["old"] != before["old"] and after_runtime["new"] == before["new"]
    script.write_text("recipe-after")
    after_recipe = identities()
    assert all(after_recipe[mode] != after_runtime[mode] for mode in benchmark.MODES)
    extra = runtime.with_name("added.py")
    extra.write_text("new-runtime-module")
    assert identities()["old"] != after_recipe["old"]
    extra.unlink()
    cache = runtime.parent/"__pycache__"
    cache.mkdir()
    (cache/"generated.pyc").write_bytes(b"bytecode")
    assert identities() == after_recipe
    runtime.unlink()
    assert identities()["old"] is None
    assert benchmark._instrumentation_identities(None, "same-head", False)["old"] is None
    assert benchmark._instrumentation_identities(script.parents[2], "same-head", False)["old"] is None
    script.unlink()
    assert all(value is None for value in identities().values())


@pytest.mark.parametrize("source_status", [b"", b" M src/dryml/runtime.py\n",
                                         b"M  src/dryml/runtime.py\n", b"?? src/dryml/added.py\n"])
def test_source_identity_requires_clean_index_worktree_and_known_source_files(tmp_path, monkeypatch, source_status):
    """Synthetic Git responses cover staged, unstaged and untracked source changes."""
    def git_output(command, **kwargs):
        if command == ["git", "rev-parse", "HEAD"]:
            return "same-head\n"
        assert command[:4] == ["git", "status", "--porcelain", "--untracked-files=normal"]
        return source_status

    monkeypatch.setattr(benchmark.subprocess, "check_output", git_output)
    monkeypatch.setattr(benchmark, "run_worker", _synthetic_matrix_worker)
    output = tmp_path/"report.json"
    assert benchmark.main(["--output", str(output), "--workload", "indexed-implicit", "--mode", "new"]) == 1
    report = json.loads(output.read_text())
    assert report["source_dirty"] == bool(source_status)
    identity = report["instrumentation_identities"]["new"]
    assert (identity is None) == bool(source_status)
    assert report["cells"][1]["instrumentation_identity"] == identity
    if identity is not None:
        assert identity["runtime"] == dict(kind="clean-git-revision", revision="same-head")


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
        result = _synthetic_cell("indexed-implicit")
        result.update(samples=samples,
                      windows=[_synthetic_window("indexed-implicit", "new", s) for s in samples],
                      end_to_end_seconds=[s+8 for s in samples], **kwargs)
        if benchmark._valid_windows(result):
            result.update(benchmark._aggregate_metrics(result))
        return result
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


@pytest.mark.exhaustive_only
@pytest.mark.parametrize("mode", ["old", "new"])
def test_tf_recipe_tiny_real_unit_qualification(tmp_path, mode):
    """Qualify current builders in isolated real runtimes, never timing cells.

    The old case opts into an existing pinned archive through
    DRYML_BENCHMARK_BASELINE_SOURCE; no archive is edited or installed.
    """
    _run_tiny_tf_qualification(tmp_path, mode)


@pytest.mark.exhaustive_only
def test_old_keras_original_recipe_missing_compile_loss(tmp_path):
    """Reproduce the original recipe's native missing-loss error on four rows."""
    _run_tiny_tf_qualification(tmp_path, "old", original=True)


def _run_tiny_tf_qualification(tmp_path, mode, *, original=False):
    root = Path(benchmark.__file__).resolve().parents[2]
    source = root
    if mode == "old":
        configured = os.environ.get("DRYML_BENCHMARK_BASELINE_SOURCE")
        if not configured:
            pytest.skip("set DRYML_BENCHMARK_BASELINE_SOURCE to the existing pinned archive")
        source = Path(configured).resolve()
        assert (source/"src/dryml/models/tf/base.py").is_file()
    script = '''
import sys
from pathlib import Path
import dryml
assert Path(dryml.__file__).resolve().is_relative_to(Path(sys.argv[1]).resolve())
assert "tensorflow" not in sys.modules
sys.path.insert(0, sys.argv[2])
from tests.tools.test_benchmark_dataset_traversal import _qualify_tiny_tf_recipe
_qualify_tiny_tf_recipe(sys.argv[3], original=sys.argv[4] == "True")
'''
    environment = {**os.environ, **benchmark._thread_env(), "PYTHONPATH": str(source/"src"),
                   "CUDA_VISIBLE_DEVICES": "-1", "JAX_PLATFORMS": "cpu", "TMPDIR": "/tmp/dryml"}
    result = subprocess.run([sys.executable, "-c", script, str(source), str(root), mode, str(original)],
                            cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr


def _qualify_tiny_tf_recipe(mode, *, original=False):
    """Check explicit compile loss and four genuine updates over two sections."""
    import numpy as np
    import tensorflow as tf
    from dryml.data import ArrayDataset, Batch
    from dryml.models import Experiment

    rng = np.random.default_rng(1729)
    x = rng.normal(0, .25, (4, 16)).astype("float32")
    y = x.sum(axis=1, keepdims=True)*np.float32(.125)
    expected = np.zeros((16, 1), np.float32)
    expected_losses = []
    for a, b in zip(x, y):
        expected_losses.append(float(np.mean((a @ expected-b)**2)))
        expected -= np.float32(.001)*np.float32(2)*a[:, None] @ (a @ expected-b)[None, :]

    for workload in (("keras",) if original else ("tf", "keras")):
        _, default = benchmark._tf_training_recipe(workload, mode)
        assert default.epochs == (1 if mode == "old" else 4)
        model, trainer = benchmark._tf_training_recipe(workload, mode, sections=2)
        data = Batch(ArrayDataset((x[:2], y[:2])), 1)
        if mode == "new":
            from dryml.data import Epoch
            data = Epoch(Batch(ArrayDataset((x, y)), 1), len=2)
        exp = Experiment(model, trainer, train_data=data)
        np.testing.assert_array_equal(model.obj.get_weights()[0], np.zeros((16, 1), np.float32))
        if original:
            trainer.loss = None
            assert "loss" not in trainer._compile_kwargs(exp)
            with pytest.raises(ValueError, match="No loss to compute"):
                trainer(exp)
            assert int(trainer.optimizer.obj.iterations.numpy()) == 0
            return

        compile_kwargs = trainer._compile_kwargs(exp)
        assert isinstance(compile_kwargs["loss"], tf.keras.losses.MeanSquaredError)
        assert compile_kwargs["loss"].reduction == "sum_over_batch_size"
        assert compile_kwargs["loss"] is trainer.loss.obj
        assert compile_kwargs["optimizer"] is trainer.optimizer.obj
        assert isinstance(compile_kwargs["optimizer"], tf.keras.optimizers.SGD)
        assert float(compile_kwargs["optimizer"].learning_rate.numpy()) == pytest.approx(.001)
        assert float(compile_kwargs["optimizer"].momentum) == 0
        assert compile_kwargs["run_eagerly"] is compile_kwargs["jit_compile"] is False
        assert trainer.verbose == 0 and trainer.epochs == (1 if mode == "old" else 2)
        # Exercise the same mean-MSE callable that compilation and explicit TF use.
        assert float(compile_kwargs["loss"](y, np.zeros_like(y))) == pytest.approx(float(np.mean(y**2)))
        accepted = []
        loss_totals = [0.0]

        def report():
            accepted.append(exp.state.step)
            assert exp.state.loss_denominator == exp.state.step
            loss_totals.append(exp.state.loss_numerator)

        if mode == "old":
            for section in range(2):
                low = section*2
                exp.train_data = Batch(ArrayDataset((x[low:low+2], y[low:low+2])), 1)
                trainer(exp, callbacks=(report,))
        else:
            trainer(exp, callbacks=(report,))
        assert accepted == [1, 2, 3, 4]
        assert (exp.state.step, exp.state.examples_seen, exp.state.epoch) == (4, 4, 2)
        assert int(trainer.optimizer.obj.iterations.numpy()) == 4
        np.testing.assert_allclose(model.obj.get_weights()[0], expected, rtol=5e-5, atol=1e-6)
        np.testing.assert_allclose(np.diff(loss_totals), expected_losses, rtol=5e-5, atol=1e-6)


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
                  measurement_protocol=benchmark.MEASUREMENT_PROTOCOL,
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


def _synthetic_window(workload, mode="new", seconds=1, cadence=0):
    domain = benchmark._expected_domain(workload, mode, "steady")
    result = dict(status="ok", work=benchmark._expected_domain(workload, mode, "end_to_end")["count"],
                  timing_domain=domain, steady_seconds=seconds if domain else None,
                  prepare_seconds=1, costs=dict(final_save_seconds=1,
                      checkpoint_seconds=[.01]*4 if cadence else [],
                      checkpoint_steps=[32, 64, 96, 128] if cadence else []),
                  end_to_end_seconds=seconds+8,
                  end_to_end_domain=benchmark._expected_domain(workload, mode, "end_to_end"))
    if workload in benchmark.TRAINING:
        result.update(invocation_domain=benchmark._training_domain("invocation"), invocation_seconds=seconds+5,
                      end_to_end_other_seconds=1,
                      steady_blocker=None if domain else "grouped reports cannot isolate within-section work",
                      first_result_seconds=1 if domain else None,
                      first_group_result_seconds=None if domain else 1+seconds/4, report_offsets=[],
                      boundary_costs=dict(initial_setup_first_report_seconds=1,
                          between_section_postlude_setup_first_report_seconds=[1]*3 if domain else None,
                          final_report_to_return_seconds=1, outside_steady_seconds=5 if domain else None),
                      cold_import_seconds=None, compile_seconds=None, open_seconds=None,
                      report_intervals=[dict(first_report_step=i*32+1, last_report_step=i*32+32,
                                             timed_updates=31, seconds=seconds/4) for i in range(4)] if domain else [])
        for i in range(4):
            start = 1+i*(1+seconds/4)
            if domain:
                result["report_offsets"].append(dict(step=32*i+1, seconds=start))
            result["report_offsets"].append(dict(step=32*i+32, seconds=start+seconds/4))
        if domain is None:
            result["boundary_costs"]["initial_setup_first_report_seconds"] = 1+seconds/4
    elif domain:
        result["delivery_intervals"] = [dict(count=domain["count"], seconds=seconds)]
    else:
        result.update(delivery_seconds=seconds,
                      delivery_domain={**benchmark._dataset_domain(workload, "end_to_end"),
                                       "name": "full-vectorized-dataset-delivery-v1"},
                      steady_blocker="whole-source vectorization cannot be split without changing the path")
    return result


def _synthetic_cell(workload, instrumentation=False, seconds=1, mode="new", cadence=0):
    work = 1024 if workload.startswith("indexed") else 128 if workload in ("torch", "tf", "keras", "jax") else 512
    if instrumentation:
        return dict(status="ok", instrumentation=True, validated=True,
                    measurement_protocol=benchmark.MEASUREMENT_PROTOCOL,
                    work=work, peak_python_bytes=64)
    windows = [_synthetic_window(workload, mode, seconds, cadence) for _ in range(5)]
    warmups = [{**_synthetic_window(workload, mode, seconds, cadence), "elapsed_seconds": seconds+8} for _ in range(2)]
    return benchmark._aggregate_windows(workload, mode, warmups, windows, cadence=cadence)


def _synthetic_matrix_worker(command, **kwargs):
    workload = command[command.index("--workload") + 1]
    mode = command[command.index("--mode") + 1]
    cadence = int(command[command.index("--checkpoint-cadence") + 1])
    unsupported = (mode == "old" and workload in ("indexed-explicit", "ordinary", "direct", "mixed", "buffered")
                   or mode.endswith("native") and workload in ("ordinary", "direct", "mixed", "buffered"))
    if unsupported:
        return dict(status="unsupported", reason="known unavailable semantics")
    # Large native gaps are intentional here; only new/pre-change gates parity.
    return _synthetic_cell(workload, "--instrumentation" in command,
                           seconds=.01 if mode.endswith("native") else 1, mode=mode, cadence=cadence)


def _synthetic_run(tmp_path, monkeypatch, worker=_synthetic_matrix_worker):
    baseline = tmp_path / "baseline"
    (baseline / "src/dryml/data").mkdir(parents=True)
    output = tmp_path / "report.json"
    monkeypatch.setattr(benchmark, "run_worker", worker)
    status = benchmark.main(["--output", str(output), "--baseline-source", str(baseline)])
    return status, json.loads(output.read_text())


@pytest.fixture(scope="module")
def _synthetic_report_seed(tmp_path_factory):
    """Build canonical evidence through the real CLI/publication path, then freeze it."""
    directory = tmp_path_factory.mktemp("synthetic-report")
    original_worker = benchmark.run_worker
    with pytest.MonkeyPatch.context() as patch:
        status, report = _synthetic_run(directory, patch)
    assert benchmark.run_worker is original_worker
    return status, json.dumps(report, allow_nan=False)


@pytest.fixture
def synthetic_report(_synthetic_report_seed):
    """Return a deep, independently mutable JSON tree for each report consumer."""
    status, seed = _synthetic_report_seed
    return status, json.loads(seed)


@pytest.mark.parametrize("consumer", [0, 1])
def test_synthetic_report_seed_is_built_once_and_copies_are_isolated(
        tmp_path_factory, _synthetic_report_seed, synthetic_report, consumer):
    """Two fixture consumers share one real construction without sharing nested state."""
    status, seed = _synthetic_report_seed
    assert status == 0 and isinstance(seed, str)
    assert len(list(tmp_path_factory.getbasetemp().glob("synthetic-report[0-9]*"))) == 1
    assert synthetic_report == (status, json.loads(seed))
    _, report = synthetic_report
    report["cells"][0]["windows"][0]["costs"]["checkpoint_steps"].append(consumer)
    report["cells"].pop()
    report["comparisons"].clear()
    fresh = json.loads(seed)
    assert fresh["cells"][0]["windows"][0]["costs"]["checkpoint_steps"] == []
    assert len(fresh["cells"]) == len(report["cells"])+1
    assert fresh["comparisons"]


def test_complete_applicable_evidence_passes_with_labeled_native_gaps(synthetic_report):
    status, report = synthetic_report
    assert status == 0
    assert report["status"] == "no-regression-observed"
    assert report["decision_domain"] == "old-new-steady"
    assert report["driver_limitations"] and all(item["reason"] and item["disposition"] ==
        "steady ratio unavailable; full-work costs required" for item in report["driver_limitations"])
    assert report["applicability_schema"] == "dryml-dataset-traversal-applicability-v1"
    assert all(c["required"] for c in report["applicability"] if c["workload"] == "jax")
    gaps = [c for c in report["comparisons"] if c["purpose"] == "native-gap" and c["ratio"]]
    assert gaps and all(c["ratio"] == 100 and c["followup"] for c in gaps)
    unsupported = [c for c in report["comparisons"] if c["status"] == "not-applicable"]
    assert unsupported and all(c["ratio"] is None and c["reason"] for c in unsupported)


@pytest.mark.parametrize("defect", ["missing", "duplicate", "wrong-work", "warmup", "repeat", "window",
                                   "instrumentation", "unsupported", "unavailable", "failed", "timeout",
                                   "unvalidated", "regression", "noise"])
def test_required_evidence_never_waived_for_bad_results(synthetic_report, defect):
    _, report = synthetic_report
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
        target.update(samples=samples, windows=[_synthetic_window("jax", "new", s) for s in samples],
                      end_to_end_seconds=[s+8 for s in samples])
        target.update(benchmark._aggregate_metrics(target))
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
    ticks = count()
    monkeypatch.setattr(benchmark, "perf_counter", lambda: float(next(ticks)))

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
    assert result["costs"]["checkpoint_steps"] == [32, 64, 96, 128]
    assert len(result["costs"]["checkpoint_seconds"]) == 4  # final save is separate
    assert result["timing_domain"] == (None if mode == "optimized-native" else benchmark._training_domain())
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


@pytest.mark.parametrize("workload,mode", [(w, "new") for w in benchmark.WORKLOADS[:6]] + [("indexed-implicit", "old")])
def test_dataset_window_cost_domains_with_synthetic_prepared_source(monkeypatch, workload, mode):
    """Clock/source fixture checks scheduling/counts without a real benchmark cell."""
    import numpy as np
    from dryml.data import Dataset
    indexed = workload.startswith("indexed")
    rows = np.arange((4096 if indexed else 512)*8, dtype=np.float32).reshape(-1, 8)
    if indexed:
        delivered = [a for _ in range(8) for a in (rows*.5).reshape(128, 32, 8)]
        section_length = 16 if workload == "indexed-explicit" else 128
    else:
        delivered = []
        for index, row in enumerate(rows):
            value = row.copy()
            if workload != "buffered":
                value[0] = index
            delivered.append((row, value, value) if workload == "mixed" else value)
        section_length = 512 if workload == "buffered" else 16
    consumed, captures = [0], []

    class View:
        def __init__(self, values):
            self.values = iter(values)

        def __iter__(self):
            return self

        def __next__(self):
            value = next(self.values)
            consumed[0] += 1
            return value

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class Prepared:
        def open(self):
            return self

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def enable_replay_recovery_if_needed(self):
            pass

        def epochs(self, stop=None):
            for index, low in enumerate(range(0, len(delivered), section_length)):
                yield index, View(delivered[low:low+section_length])

        def iterator(self, epoch):
            return View(delivered[epoch*128:(epoch+1)*128])

        def capture_state(self):
            captures.append(consumed[0])

    def capture(*args, **kwargs):
        assert consumed[0] == 129
        captures.append(129)
        return delivered[129:], dict(capture_seconds=10, save_seconds=20, restore_seconds=30)

    ticks = count()
    monkeypatch.setattr(benchmark, "perf_counter", lambda: float(next(ticks)))
    monkeypatch.setattr(Dataset, "prepare", lambda _: Prepared())
    monkeypatch.setattr(benchmark, "_capture", capture)
    window = benchmark._dataset_window(workload, mode)
    expected_domain = benchmark._dataset_domain(workload)
    assert window["timing_domain"] == expected_domain
    assert consumed[0] == window["work"] == len(delivered)
    assert sum(i["count"] for i in window["delivery_intervals"]) == expected_domain["count"]
    assert len(window["section_first_result_seconds"]) == expected_domain["sections"]
    assert captures == ([] if mode == "old" else [len(delivered)] if indexed else [129, len(delivered)])


@pytest.mark.parametrize("empty_final_view", [False, True])
def test_dataset_first_result_includes_staging_and_accounts_for_no_result_time(monkeypatch, empty_final_view):
    """Charge enumerator setup/staging once; retain tail, EOF and capture costs."""
    import numpy as np
    from dryml.data import Dataset
    now, delivered, staged, eof = [0.0], [], [], []

    def advance(seconds):
        now[0] += seconds

    class View:
        def __init__(self, batches):
            self.batches = iter(batches)
            self.first = True

        def __iter__(self):
            return self

        def __next__(self):
            try:
                value = next(self.batches)
            except StopIteration:
                advance(5)
                eof.append(True)
                raise
            advance(2 if self.first else 3)
            self.first = False
            delivered.append(value)
            return value

    class Prepared:
        def open(self):
            advance(11)
            return self

        def __enter__(self):
            return self

        def __exit__(self, *args):
            advance(19)

        def enable_replay_recovery_if_needed(self):
            pass

        def epochs(self, stop):
            assert stop == 8
            advance(17)
            return self.enumerate_views()

        def enumerate_views(self):
            rows = np.arange(4096*8, dtype=np.float32).reshape(-1, 8)
            for section in range(8):
                advance(13)
                batches = (rows*np.float32(.5)).reshape(128, 32, 8)
                staged.append(batches[0])
                yield section, View(batches)
            if empty_final_view:
                advance(31)
                yield 8, View([])
            advance(29)

        def capture_state(self):
            advance(23)
            assert len(delivered) == 1024

    def prepare(_):
        advance(7)
        return Prepared()

    monkeypatch.setattr(Dataset, "prepare", prepare)
    monkeypatch.setattr(benchmark, "perf_counter", lambda: now[0])
    window, total = benchmark._timed(lambda: benchmark._dataset_window("indexed-implicit", "new"))
    assert window["first_result_seconds"] == 32  # setup 17 + staging 13 + delivery 2
    assert window["section_first_result_seconds"] == [32]+[15]*7
    assert window["costs"]["section_first_result_seconds"] == window["section_first_result_seconds"]
    assert window["delivery_intervals"] == [dict(count=127, seconds=386)]*8
    assert window["steady_seconds"] == 3088
    no_result = window["costs"]["epoch_enumeration_without_result_seconds"]
    assert no_result == ([36, 29] if empty_final_view else [29])
    assert (len(staged), len(delivered), len(eof)) == (8, 1024, 9 if empty_final_view else 8)
    assert window["work"] == 1024 and window["timing_domain"]["count"] == 1016
    assert window["prepare_seconds"] == 7 and window["open_seconds"] == 11
    assert window["costs"]["final_capture_seconds"] == 23
    assert total == now[0] == 3314+(36 if empty_final_view else 0)
    assert total == (window["prepare_seconds"]+window["open_seconds"]
                     +sum(window["section_first_result_seconds"])+window["steady_seconds"]
                     +sum(no_result)+window["costs"]["final_capture_seconds"]+19)


@pytest.mark.parametrize("workload,sections,tail_count", [
    ("indexed-implicit", 8, 1016), ("indexed-explicit", 64, 960)])
@pytest.mark.parametrize("mode", ["matched-native", "optimized-native"])
def test_native_dataset_producer_cost_domains_with_synthetic_clock(monkeypatch, workload, sections, tail_count, mode):
    """Run native NumPy producer loops once; qualify copied metadata, not repeats."""
    import numpy as np
    ticks, validations, batches = count(), [], []
    monkeypatch.setattr(benchmark, "perf_counter", lambda: float(next(ticks)))
    original_equal, original_concatenate = np.testing.assert_array_equal, np.concatenate

    def concatenate(values, *args, **kwargs):
        batches.append([value.shape for value in values])
        return original_concatenate(values, *args, **kwargs)

    def validate(actual, expected):
        validations.append(actual.copy())
        original_equal(actual, expected)

    monkeypatch.setattr(np, "concatenate", concatenate)
    monkeypatch.setattr(np.testing, "assert_array_equal", validate)
    window, total = benchmark._timed(lambda: benchmark._dataset_window(workload, mode))
    assert window["status"] == "ok" and window["work"] == 1024
    assert batches == [[(32, 8)]*1024]
    assert len(validations) == 1
    expected = np.tile(np.arange(4096*8, dtype=np.float32).reshape(-1, 8)*np.float32(.5), (8, 1))
    original_equal(validations[0], expected)
    assert window["prepare_seconds"] == 1 and window["open_seconds"] is None
    if mode == "matched-native":
        assert window["timing_domain"] == dict(name="post-first-section-delivery-v1",
                                               unit="batches", count=tail_count, sections=sections)
        assert window["section_first_result_seconds"] == [1]*sections
        assert window["first_result_seconds"] == 1
        assert window["delivery_intervals"] == [dict(count=tail_count//sections, seconds=1)]*sections
        assert window["steady_seconds"] == sections
        assert window["delivery_domain"] is window["delivery_seconds"] is window["steady_blocker"] is None
    else:
        assert window["timing_domain"] is window["steady_seconds"] is None
        assert window["steady_blocker"]
        assert window["first_result_seconds"] is None and window["section_first_result_seconds"] == []
        assert window["delivery_intervals"] == []
        assert window["delivery_seconds"] == 1
        assert window["delivery_domain"] == dict(name="full-vectorized-dataset-delivery-v1",
                                                 unit="batches", count=1024, sections=sections)
    window.update(end_to_end_seconds=total,
                  end_to_end_domain=benchmark._expected_domain(workload, mode, "end_to_end"))
    assert total > window["prepare_seconds"]+(window["steady_seconds"] or window["delivery_seconds"])
    # Copies satisfy the schema's fixed populations without executing warmups/repetitions.
    cell = benchmark._aggregate_windows(workload, mode,
        [dict(copy.deepcopy(window), elapsed_seconds=total) for _ in range(2)],
        [copy.deepcopy(window) for _ in range(5)])
    assert benchmark._valid_protocol(cell)
    new = _synthetic_cell(workload)
    assert (benchmark.compare(cell, new)["ratio"] is None) == (mode == "optimized-native")
    assert benchmark.compare(cell, cell, metric="end_to_end")["ratio"] == 1
    if mode == "optimized-native":
        assert cell["summary"] is cell["throughput_per_second"] is None
    else:
        assert cell["throughput_per_second"] == tail_count/sections


@pytest.mark.parametrize("key,value", [("warmups", None), ("warmups", [1, 2]),
                                      ("windows", None), ("samples", None),
                                      ("end_to_end_seconds", None)])
def test_malformed_required_evidence_is_a_blocker_not_an_aggregation_crash(synthetic_report, key, value):
    _, report = synthetic_report
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


@pytest.mark.exhaustive_only
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


@pytest.mark.exhaustive_only
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


@pytest.mark.exhaustive_only
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
