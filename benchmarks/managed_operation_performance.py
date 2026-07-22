#!/usr/bin/env python3
"""Emit deterministic structural measurements for managed-operation bounds."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

from dryml.artifacts import CachedDataset
from dryml.artifacts.representations.numpy_sequence import write_numpy_sequence
from dryml.core2 import Object
from dryml.core2.store.dir import DirStore
from dryml.data import ArrayDataset
from dryml.formats.canonical import canonical_json_bytes
from dryml.managed import (
    CallbackCoordinator,
    ControlRequest,
    ManagedCallback,
    ManagedOutput,
    current_operation_context,
    managed,
    transfer_realizations,
)
from dryml.managed.events import EventBuffer, OperationEvent
from dryml.managed.store import OperationControl
from dryml.records import (
    AdapterRecord,
    DataRecord,
    ProductManifest,
    ProductManifestEntry,
    ProductRootManifest,
    RecordStoreIO,
    make_representation_spec,
)


SCHEMA = "dryml.managed_operation_performance.v1"
SCHEMA_VERSION = 1
RESULT_LIMIT_BYTES = 2 * 1024 * 1024
SHARD_PAYLOAD_LIMIT_BYTES = 256 * 1024 * 1024
SHARD_WORKER_TIMEOUT_SECONDS = 120
ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = Path(__file__).resolve()
_LOOKUP_REPRESENTATION = make_representation_spec(
    "dryml.benchmark.lookup",
    version="1",
    storage_kinds=("product-dir",),
)


class LookupOperation(Object):
    """Tiny managed producer used only for direct-active lookup counts."""

    @managed(outputs=(ManagedOutput("result", primary=True, kind="data"),))
    def compute(self):
        current_operation_context().write_output(
            "result",
            "value.bin",
            (b"value",),
            representation=_LOOKUP_REPRESENTATION,
        )


def measure_shard_scaling(
    *,
    small_rows: int = 256,
    large_rows: int = 4096,
    row_bytes: int = 4096,
    shard_rows: int = 128,
) -> dict:
    """Measure two isolated writers with one fixed logical shard buffer."""

    for name, value in (
        ("small_rows", small_rows),
        ("large_rows", large_rows),
        ("row_bytes", row_bytes),
        ("shard_rows", shard_rows),
    ):
        _positive_bounded(value, name)
    _validate_shard_buffer(row_bytes, shard_rows)
    if large_rows <= small_rows:
        raise ValueError("large_rows must exceed small_rows")
    if large_rows * row_bytes > SHARD_PAYLOAD_LIMIT_BYTES:
        raise ValueError("shard measurement payload exceeds 256 MiB")
    small = _shard_subprocess(small_rows, row_bytes, shard_rows)
    large = _shard_subprocess(large_rows, row_bytes, shard_rows)
    payload_growth = large["payload_bytes"] - small["payload_bytes"]
    rss_growth = max(0, large["peak_rss_bytes"] - small["peak_rss_bytes"])
    return {
        "small": small,
        "large": large,
        "invariants": {
            "fixed_configured_buffer": (
                small["configured_buffer_bytes"] == large["configured_buffer_bytes"]
            ),
            "files_scale_with_shards": all((
                small["file_count"] == small["shard_count"] + 1,
                large["file_count"] == large["shard_count"] + 1,
            )),
            "rss_growth_below_payload_growth": rss_growth < payload_growth,
        },
    }


def measure_active_lookup(*, history: int = 24) -> dict:
    """Count direct active reads while retained realization history grows."""

    _positive_bounded(history, "history", maximum=128)
    with tempfile.TemporaryDirectory(prefix="dryml-managed-lookup-") as temp:
        store = DirStore(Path(temp) / "store", query_index="none")
        producer = LookupOperation()
        producer.compute(store=store)
        for _index in range(history - 1):
            producer.compute.rerun(store=store)

        counts = Counter()
        original_read = OperationControl._read_realization
        original_history = OperationControl.history
        original_records = RecordStoreIO.iter_records

        def read_realization(self, *args, **kwargs):
            counts["active_lookup_realization_reads"] += 1
            return original_read(self, *args, **kwargs)

        def scan_history(self, *args, **kwargs):
            counts["history_scans"] += 1
            return original_history(self, *args, **kwargs)

        def scan_records(self, *args, **kwargs):
            counts["record_scans"] += 1
            return original_records(self, *args, **kwargs)

        OperationControl._read_realization = read_realization
        OperationControl.history = scan_history
        RecordStoreIO.iter_records = scan_records
        try:
            status = producer.compute.status(store=store)
            results = producer.compute.results(store=store)
        finally:
            OperationControl._read_realization = original_read
            OperationControl.history = original_history
            RecordStoreIO.iter_records = original_records

        return {
            "history_count": len(producer.compute.history(store=store)),
            "active_realization_id": status.active_realization_id,
            "active_output_count": len(results),
            "active_lookup_realization_reads": counts[
                "active_lookup_realization_reads"
            ],
            "history_scans": counts["history_scans"],
            "record_scans": counts["record_scans"],
        }


def measure_events_adapter_export(*, event_count: int = 1000) -> dict:
    """Measure bounded events plus one streaming adapter and exact export."""

    _positive_bounded(event_count, "event_count", maximum=100_000)
    events = EventBuffer(max_events=32)
    callback_counts = Counter()

    def request_checkpoint(_event):
        callback_counts["checkpoint_requests"] += 1
        return ControlRequest.CHECKPOINT

    checkpoints = CallbackCoordinator(
        (
            ManagedCallback(
                request_checkpoint,
                controls={ControlRequest.CHECKPOINT},
            ),
        )
    )
    for sequence in range(1, event_count + 1):
        event = OperationEvent(sequence, "safe_point")
        events.append(event)
        checkpoints.publish(event)

    with tempfile.TemporaryDirectory(prefix="dryml-managed-adapter-export-") as temp:
        root = Path(temp)
        source = DirStore(root / "source", query_index="none")
        destination = DirStore(root / "destination", query_index="none")
        cached = CachedDataset(
            ArrayDataset(np.arange(256, dtype=np.int64).reshape(128, 2))
        )
        completed = cached.compute(
            store=source,
            representation="numpy-sequence",
            shard_rows=16,
        )
        source_record = DataRecord.from_envelope(
            source.records.read_record(completed.outputs["data"].record_id)
        )
        converted = cached.request_representation("parquet", store=source)
        if converted.status != "ok":
            message = converted.issues[0].message if converted.issues else converted.status
            raise RuntimeError(f"managed adapter benchmark failed: {message}")
        target_record = DataRecord.from_envelope(
            source.records.read_record(converted.target_records[-1].record_id)
        )
        report = transfer_realizations(
            source,
            destination,
            cached.compute.result,
        )
        adapter_records = tuple(source.records.find_records(kind=AdapterRecord.kind))
        temporary_bytes = sum(
            path.stat().st_size
            for path in root.rglob("*")
            if path.is_file() and _is_temporary_path(path)
        )
        return {
            "events": {
                "submitted": event_count,
                "retained": len(events.snapshot()),
                "capacity": 32,
                "checkpoint_requests": callback_counts["checkpoint_requests"],
                "coalesced_checkpoint_controls": int(
                    checkpoints.poll() is ControlRequest.CHECKPOINT
                ),
            },
            "adapter": {
                "source_bytes": _record_bytes(source_record),
                "target_bytes": _record_bytes(target_record),
                "intermediate_bytes": 0,
                "adapter_records": len(adapter_records),
            },
            "export": {
                "records_copied": len(report.records),
                "products_copied": len(report.products),
                "definitions_copied": len(report.definitions),
                "temporary_bytes_after": temporary_bytes,
            },
        }


def run_benchmark(
    *,
    small_rows: int = 256,
    large_rows: int = 4096,
    row_bytes: int = 4096,
    shard_rows: int = 128,
    history: int = 24,
    event_count: int = 1000,
) -> dict:
    """Run every bounded scenario and return one versioned JSON-ready object."""

    result = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "success",
        "configuration": {
            "small_rows": small_rows,
            "large_rows": large_rows,
            "row_bytes": row_bytes,
            "shard_rows": shard_rows,
            "history": history,
            "event_count": event_count,
        },
        "shard_scaling": measure_shard_scaling(
            small_rows=small_rows,
            large_rows=large_rows,
            row_bytes=row_bytes,
            shard_rows=shard_rows,
        ),
        "active_lookup": measure_active_lookup(history=history),
        "events_adapter_export": measure_events_adapter_export(
            event_count=event_count
        ),
    }
    if len(json.dumps(result, sort_keys=True, allow_nan=False).encode()) > RESULT_LIMIT_BYTES:
        raise RuntimeError("managed operation benchmark result exceeds output bound")
    return result


def _measure_shards(rows: int, row_bytes: int, shard_rows: int) -> dict:
    for name, value in (
        ("rows", rows),
        ("row_bytes", row_bytes),
        ("shard_rows", shard_rows),
    ):
        _positive_bounded(value, name)
    _validate_shard_buffer(row_bytes, shard_rows)
    if rows * row_bytes > SHARD_PAYLOAD_LIMIT_BYTES:
        raise ValueError("shard measurement payload exceeds 256 MiB")
    row = np.zeros((row_bytes,), dtype=np.uint8)
    with tempfile.TemporaryDirectory(prefix="dryml-managed-shards-") as temp:
        root = Path(temp)
        index = write_numpy_sequence(
            (row for _index in range(rows)),
            root,
            shard_rows=shard_rows,
            shard_bytes=max(row_bytes, shard_rows * row_bytes),
        )
        entries = []
        hash_passes = 0
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            entries.append(
                ProductManifestEntry(
                    path.relative_to(root).as_posix(),
                    path.stat().st_size,
                    _sha256(path),
                )
            )
            hash_passes += 1
        manifest = ProductManifest(tuple(entries))
        manifest_bytes = len(canonical_json_bytes(manifest.to_json()))
        return {
            "rows": rows,
            "row_bytes": row_bytes,
            "payload_bytes": rows * row_bytes,
            "configured_buffer_bytes": shard_rows * row_bytes,
            "shard_count": len(index.shards),
            "sequence_index_entries": len(index.shards),
            "manifest_entries": len(manifest.entries),
            "index_bytes": (root / "index.json").stat().st_size,
            "manifest_bytes": manifest_bytes,
            "shard_hash_passes": len(index.shards),
            "manifest_hash_passes": hash_passes,
            "file_count": len(manifest.entries),
            "peak_rss_bytes": _peak_rss_bytes(),
        }


def _shard_subprocess(rows: int, row_bytes: int, shard_rows: int) -> dict:
    output = subprocess.check_output(
        [
            sys.executable,
            str(_SCRIPT),
            "--worker-shards",
            "--rows",
            str(rows),
            "--row-bytes",
            str(row_bytes),
            "--shard-rows",
            str(shard_rows),
        ],
        cwd=ROOT,
        text=True,
        timeout=SHARD_WORKER_TIMEOUT_SECONDS,
    )
    value = json.loads(output)
    if not isinstance(value, dict):
        raise RuntimeError("shard subprocess returned malformed JSON")
    return value


def _peak_rss_bytes() -> int:
    value = _peak_rss_windows() if platform.system() == "Windows" else _peak_rss_posix()
    if type(value) is not int or value <= 0:
        raise RuntimeError("peak RSS measurement must be a positive byte count")
    return value


def _peak_rss_posix() -> int:
    import resource

    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def _peak_rss_windows() -> int:
    import ctypes
    from ctypes import wintypes

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = (
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        )

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    get_current_process = kernel32.GetCurrentProcess
    get_current_process.argtypes = ()
    get_current_process.restype = wintypes.HANDLE
    get_process_memory_info = psapi.GetProcessMemoryInfo
    get_process_memory_info.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(ProcessMemoryCounters),
        wintypes.DWORD,
    )
    get_process_memory_info.restype = wintypes.BOOL

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    if not get_process_memory_info(
        get_current_process(), ctypes.byref(counters), counters.cb
    ):
        error = ctypes.get_last_error()
        raise OSError(error, "GetProcessMemoryInfo failed")
    return int(counters.PeakWorkingSetSize)


def _record_bytes(record: DataRecord) -> int:
    try:
        if set(record.manifest) == {"entries"}:
            return sum(
                entry.size for entry in ProductManifest.from_json(record.manifest).entries
            )
        return ProductRootManifest.from_json(record.manifest).total_size
    except Exception as exc:
        raise RuntimeError("benchmark record has no bounded product size") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_temporary_path(path: Path) -> bool:
    return any((
        path.name.startswith(".staging-"),
        ".partial-" in path.name,
        path.suffix == ".tmp",
    ))


def _positive_bounded(value: int, name: str, *, maximum: int = 1_000_000) -> int:
    if type(value) is not int or value < 1 or value > maximum:
        raise ValueError(f"{name} must be between 1 and {maximum}")
    return value


def _validate_shard_buffer(row_bytes: int, shard_rows: int) -> None:
    if shard_rows * row_bytes > SHARD_PAYLOAD_LIMIT_BYTES:
        raise ValueError("configured shard buffer exceeds 256 MiB")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-shards", action="store_true")
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--large-rows", type=int, default=4096)
    parser.add_argument("--row-bytes", type=int, default=4096)
    parser.add_argument("--shard-rows", type=int, default=128)
    parser.add_argument("--history", type=int, default=24)
    parser.add_argument("--event-count", type=int, default=1000)
    args = parser.parse_args(argv)
    if args.worker_shards:
        result = _measure_shards(args.rows, args.row_bytes, args.shard_rows)
    else:
        result = run_benchmark(
            small_rows=args.rows,
            large_rows=args.large_rows,
            row_bytes=args.row_bytes,
            shard_rows=args.shard_rows,
            history=args.history,
            event_count=args.event_count,
        )
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
