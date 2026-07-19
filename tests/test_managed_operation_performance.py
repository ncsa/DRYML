from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


BENCHMARK_PATH = (
    Path(__file__).parents[1] / "benchmarks" / "managed_operation_performance.py"
)
SPEC = importlib.util.spec_from_file_location(
    "dryml_managed_operation_performance_benchmark", BENCHMARK_PATH
)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


def test_subprocess_shard_measurement_reports_bounded_buffer_and_peak_rss():
    result = benchmark.measure_shard_scaling(
        small_rows=128,
        large_rows=2048,
        row_bytes=4096,
        shard_rows=128,
    )
    small = result["small"]
    large = result["large"]

    assert large["payload_bytes"] == 16 * small["payload_bytes"]
    assert small["configured_buffer_bytes"] == large["configured_buffer_bytes"]
    assert small["file_count"] == small["shard_count"] + 1
    assert large["file_count"] == large["shard_count"] + 1
    assert small["sequence_index_entries"] == small["shard_count"]
    assert large["sequence_index_entries"] == large["shard_count"]
    assert small["manifest_entries"] == small["file_count"]
    assert large["manifest_entries"] == large["file_count"]
    assert small["shard_hash_passes"] == small["shard_count"]
    assert large["shard_hash_passes"] == large["shard_count"]
    assert small["manifest_hash_passes"] == small["file_count"]
    assert large["manifest_hash_passes"] == large["file_count"]
    assert small["manifest_bytes"] > 0
    assert large["manifest_bytes"] > small["manifest_bytes"]
    assert small["peak_rss_bytes"] > 0
    assert large["peak_rss_bytes"] > 0
    assert result["invariants"]["rss_growth_below_payload_growth"]


@pytest.mark.parametrize(
    "rows,row_bytes,shard_rows",
    [
        (0, 1, 1),
        (1, 0, 1),
        (1, 1, 0),
        (1_000_000, 1024, 1),
    ],
)
def test_shard_worker_rejects_unbounded_inputs(rows, row_bytes, shard_rows):
    with pytest.raises(ValueError):
        benchmark._measure_shards(rows, row_bytes, shard_rows)


def test_shard_worker_rejects_configured_buffer_larger_than_payload_limit():
    shard_rows = benchmark.SHARD_PAYLOAD_LIMIT_BYTES // 1024 + 1

    with pytest.raises(ValueError, match="buffer"):
        benchmark._measure_shards(1, 1024, shard_rows)


def test_shard_scaling_rejects_unbounded_buffer_before_subprocess(monkeypatch):
    shard_rows = benchmark.SHARD_PAYLOAD_LIMIT_BYTES // 1024 + 1
    calls = []

    def unexpected_subprocess(*args):
        calls.append(args)
        raise AssertionError("shard subprocess must not start")

    monkeypatch.setattr(benchmark, "_shard_subprocess", unexpected_subprocess)

    with pytest.raises(ValueError, match="buffer"):
        benchmark.measure_shard_scaling(
            small_rows=1,
            large_rows=2,
            row_bytes=1024,
            shard_rows=shard_rows,
        )

    assert calls == []


def test_benchmark_module_import_does_not_require_resource():
    script = f"""
import builtins
import importlib.util
import sys

original_import = builtins.__import__
def import_without_resource(name, *args, **kwargs):
    if name == 'resource':
        raise ModuleNotFoundError('resource is unavailable')
    return original_import(name, *args, **kwargs)

builtins.__import__ = import_without_resource
spec = importlib.util.spec_from_file_location('resource_free_benchmark', {str(BENCHMARK_PATH)!r})
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
print(module.SCHEMA)
"""

    output = subprocess.check_output([sys.executable, "-c", script], text=True)

    assert output.strip() == benchmark.SCHEMA


def test_peak_rss_dispatches_to_windows_measurement(monkeypatch):
    monkeypatch.setattr(benchmark.platform, "system", lambda: "Windows")
    monkeypatch.setattr(benchmark, "_peak_rss_windows", lambda: 12345)

    assert benchmark._peak_rss_bytes() == 12345


def test_active_lookup_counts_do_not_scale_with_realization_history():
    short = benchmark.measure_active_lookup(history=2)
    long = benchmark.measure_active_lookup(history=24)

    assert short["history_count"] == 2
    assert long["history_count"] == 24
    assert short["active_lookup_realization_reads"] == (
        long["active_lookup_realization_reads"]
    )
    assert short["history_scans"] == long["history_scans"] == 0
    assert short["record_scans"] == long["record_scans"] == 0


def test_events_adapter_and_export_emit_machine_readable_operation_counts():
    result = benchmark.measure_events_adapter_export(event_count=1000)

    assert result["events"]["submitted"] == 1000
    assert result["events"]["retained"] == result["events"]["capacity"] == 32
    assert result["events"]["checkpoint_requests"] == 1000
    assert result["events"]["coalesced_checkpoint_controls"] == 1
    assert result["adapter"]["source_bytes"] > 0
    assert result["adapter"]["target_bytes"] > 0
    assert result["adapter"]["adapter_records"] == 1
    assert result["adapter"]["intermediate_bytes"] == 0
    assert result["export"]["records_copied"] > 0
    assert result["export"]["products_copied"] > 0
    assert result["export"]["temporary_bytes_after"] == 0


def test_benchmark_result_is_versioned_bounded_json():
    result = benchmark.run_benchmark(
        small_rows=64,
        large_rows=256,
        row_bytes=1024,
        shard_rows=32,
        history=3,
        event_count=64,
    )
    encoded = json.dumps(result, sort_keys=True, allow_nan=False).encode()

    assert result["schema"] == "dryml.managed_operation_performance.v1"
    assert result["schema_version"] == 1
    assert result["status"] == "success"
    assert len(encoded) <= benchmark.RESULT_LIMIT_BYTES
