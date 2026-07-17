import json
import importlib.util
from pathlib import Path
import sys

import pytest

BENCHMARK_PATH = Path(__file__).parents[1] / "benchmarks" / "dispatch_code_performance.py"
SPEC = importlib.util.spec_from_file_location("dryml_dispatch_code_performance_benchmark", BENCHMARK_PATH)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


@pytest.mark.parametrize(
    "value",
    [0, -1, 1.5, True, "", " 1", "1 ", "+1", "1.0", "1_0", "1001"],
)
def test_pure_sample_count_rejects_non_positive_or_unbounded_values(value):
    with pytest.raises(ValueError):
        benchmark.positive_sample_count(value, maximum=benchmark.PURE_MAX, name="pure samples")


def test_sample_defaults_and_maxima_are_explicit():
    assert benchmark.PURE_DEFAULT == 20
    assert benchmark.PURE_MAX == 1000
    assert benchmark.MANAGED_DEFAULT == 5
    assert benchmark.MANAGED_MAX == 100
    assert benchmark.positive_sample_count("100", maximum=100, name="managed samples") == 100
    with pytest.raises(ValueError):
        benchmark.positive_sample_count("101", maximum=100, name="managed samples")


def test_scenario_registry_is_fixed_unique_and_covers_required_families():
    names = [scenario.name for scenario in benchmark.scenarios()]

    assert len(names) == len(set(names))
    assert {"annotations.fragments_0", "annotations.fragments_1", "annotations.fragments_16", "annotations.fragments_64"} <= set(names)
    assert {"trace.zero", "trace.one", "trace.repeated_16"} <= set(names)
    assert "worker.local_subprocess_trivial" in names
    assert "world.plan_four_workers" in names


def test_pure_benchmark_emits_bounded_versioned_json_and_operation_counts():
    result = benchmark.run_benchmark(mode="pure", pure_samples=1, managed_samples=1)
    encoded = json.dumps(result, allow_nan=False)
    scenarios = {scenario["name"]: scenario for scenario in result["scenarios"]}

    assert result["schema"] == benchmark.SCHEMA
    assert result["schema_version"] == 1
    assert result["status"] == "success"
    assert len(encoded.encode()) < benchmark.RESULT_LIMIT_BYTES
    analysis_operations = scenarios["analysis.source_ast_static_calls"]["samples"][0]["operations"]
    assert analysis_operations["source_extractions"] >= 2
    assert analysis_operations["ast_parses"] == 2
    assert scenarios["dispatch.function_plan_no_probe"]["samples"][0]["operations"].get("code_probe_calls", 0) == 0
    assert scenarios["dispatch.function_plan_no_probe"]["samples"][0]["operations"].get("environment_probe_calls", 0) == 0
    assert scenarios["trace.repeated_16"]["samples"][0]["metrics"]["observed_calls"] == 16
    assert scenarios["world.plan_four_workers"]["samples"][0]["metrics"]["workers"] == 4
    assert scenarios["world.plan_one_worker"]["samples"][0]["operations"]["planning_metadata_snapshots"] == 3
    assert scenarios["world.plan_four_workers"]["samples"][0]["operations"]["planning_metadata_snapshots"] == 6
    assert str(benchmark.ROOT) not in encoded
    assert str(benchmark.FIXTURE_DIR) not in encoded


def test_managed_probe_records_launch_and_unavailable_child_stages():
    result = benchmark.run_benchmark(
        mode="managed",
        pure_samples=1,
        managed_samples=1,
        scenario_names=("probe.python_executable",),
    )
    sample = result["scenarios"][0]["samples"][0]

    assert sample["operations"] == {"code_probe_calls": 1, "managed_process_launches": 1}
    assert sample["stages"]["managed_child_analysis"] == {"available": False, "seconds": None}
    assert sample["cleanup"] == "complete"


def test_unknown_scenario_fails_closed():
    with pytest.raises(ValueError, match="unknown benchmark scenario"):
        benchmark.run_benchmark(
            mode="pure", pure_samples=1, managed_samples=1,
            scenario_names=("missing",),
        )


def test_candidate_sha_rejects_an_unrelated_enclosing_repository(tmp_path, monkeypatch):
    expected_root = tmp_path / "standalone-source"
    expected_root.mkdir()

    def fake_check_output(command, **_kwargs):
        if command[-1] == "--show-toplevel":
            return str(tmp_path)
        raise AssertionError("identity lookup must stop after the root mismatch")

    monkeypatch.setattr(benchmark.subprocess, "check_output", fake_check_output)

    assert benchmark._git_sha(expected_root) is None


def test_failed_scenario_cleans_temporary_store(tmp_path, monkeypatch):
    created = []
    original = benchmark.tempfile.TemporaryDirectory

    class RecordingTemporaryDirectory:
        def __init__(self, *args, **kwargs):
            self.inner = original(dir=tmp_path, *args, **kwargs)

        def __enter__(self):
            path = self.inner.__enter__()
            created.append(benchmark.Path(path))
            return path

        def __exit__(self, *exc_info):
            return self.inner.__exit__(*exc_info)

    def fail(_context):
        raise RuntimeError("synthetic benchmark failure")

    monkeypatch.setattr(benchmark.tempfile, "TemporaryDirectory", RecordingTemporaryDirectory)
    monkeypatch.setattr(
        benchmark,
        "scenarios",
        lambda: (benchmark.Scenario("failure", "warm_in_process", fail),),
    )

    with pytest.raises(RuntimeError, match="synthetic benchmark failure"):
        benchmark.run_benchmark(
            mode="pure", pure_samples=1, managed_samples=1,
            scenario_names=("failure",),
        )

    assert len(created) == 1
    assert not created[0].exists()
