"""Regenerate deterministic CachedDataset v1 Store fixtures with pinned writers."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from uuid import UUID

import netCDF4
import pyarrow

from dryml.artifacts import CachedDataset
from dryml.artifacts._cache_model import _VERSION as CACHE_FORMAT_VERSION
from dryml.core import ObjectId, ObjectRef, Repo
from dryml.core.metadata import SnapshotCapture
from dryml.core.repo_plan import apply_exact_reference_identity
from dryml.core.store.dir import DirStore
import dryml.core.snapshot_capture as snapshot_capture_module
from dryml.core.utils.graph.path import GraphPath
from dryml.environments import (
    DrymlRuntimeRecord,
    EnvironmentRecord,
    PackageRecord,
    PlatformRecord,
    PythonRecord,
)
from dryml.managed import ManagedConfig
from tests.cached_dataset_fixture_types import (
    FixtureDataset,
)


ROOT = Path(__file__).resolve().parent
STORE_ROOT = ROOT / "dir-store"
CONTROL_ROOT = ROOT / ".control"
SAVED_AT = datetime(2026, 9, 25, 12, 0, 0, tzinfo=timezone.utc)


def _environment() -> EnvironmentRecord:
    """Return fixed synthetic environment evidence without machine-local paths."""

    return EnvironmentRecord(
        python=PythonRecord(
            "3.12.0", "CPython", executable="/opt/dryml-fixture/bin/python",
            prefix="/opt/dryml-fixture", base_prefix="/opt/python",
        ),
        platform=PlatformRecord(
            "FixtureOS", "1", "fixture", "fixture64", "FixtureOS-1-fixture64",
            os_name="posix", sys_platform="fixtureos", implementation_name="cpython",
            implementation_version="3.12.0", platform_python_implementation="CPython",
        ),
        distributions={
            "netcdf4": PackageRecord("netCDF4", "1.7.4"),
            "pyarrow": PackageRecord("pyarrow", "25.0.1"),
        },
        dryml=DrymlRuntimeRecord(
            version="0.3.0b1", git_revision="cached-dataset-v1-fixture",
            execution_protocol="4",
            schema_versions={
                "store": "3", "cached_dataset": str(CACHE_FORMAT_VERSION),
            },
            features=("cached-dataset",),
        ),
        kind="synthetic",
        tags=("fixture",),
        details={"environment_name": "cached-dataset-v1"},
    )


def _fix_identity(value, nonce: int) -> None:
    """Assign one stable fixture ObjectId and lineage timestamp."""

    identity = ObjectId._trusted(
        ("fixture", "cacheddataset", "v1"),
        UUID(f"00000000-0000-0000-0000-{nonce:012d}"),
    )
    apply_exact_reference_identity(
        value,
        ObjectRef(value.definition, {GraphPath(): identity}),
        lineage_facts={GraphPath(): SAVED_AT},
    )


def _authority_hashes(root: Path) -> dict[str, str]:
    """Hash every checked-in Store authority file by relative path."""

    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file() and ".dryml" not in path.parts and not path.name.endswith(".lock")
    }


def _build_cache(repo: Repo, control: DirStore, codec: str, kind: str, nonce: int):
    """Compute one fixed cache and return its completed StateRef."""

    source = FixtureDataset(kind)
    cache = CachedDataset(source)
    _fix_identity(cache, nonce)
    return cache.compute(
        codec=codec,
        target_chunk_bytes=512,
        managed=ManagedConfig(state_repo=repo, control_store=control),
    )


def main() -> None:
    """Write the committed Store and semantic/hash manifest."""

    required = {"pyarrow": "25.0.1", "netCDF4": "1.7.4"}
    observed = {"pyarrow": pyarrow.__version__, "netCDF4": netCDF4.__version__}
    if observed != required:
        raise RuntimeError(
            f"CachedDataset v1 fixtures require exact writer versions {required}; "
            f"found {observed}."
        )
    shutil.rmtree(STORE_ROOT, ignore_errors=True)
    shutil.rmtree(CONTROL_ROOT, ignore_errors=True)
    store = DirStore(STORE_ROOT, query_index="none")
    control = DirStore(CONTROL_ROOT, query_index="none")
    repo = Repo(store)
    original_capture = snapshot_capture_module.capture_snapshot

    def fixed_capture(plan, *, observer=None, clock=None):
        captured = original_capture(
            plan, observer=_environment, clock=lambda: SAVED_AT,
        )
        return SnapshotCapture(
            captured.lineages, SAVED_AT, captured.environment,
            captured.environment_status, captured.requirements,
            captured.requirements_status, captured.requirements_coverage,
            captured.diagnostics,
        )

    snapshot_capture_module.capture_snapshot = fixed_capture
    try:
        states = {}
        nonce = 1
        for codec in ("numpy", "parquet", "netcdf"):
            states[codec] = {}
            for kind in ("fidelity", "empty"):
                states[codec][kind] = _build_cache(
                    repo, control, codec, kind, nonce,
                ).digest()
                nonce += 1
        states["numpy"]["polynomial"] = _build_cache(
            repo, control, "numpy", "polynomial", nonce,
        ).digest()
        nonce += 1
        states["parquet"]["polynomial"] = _build_cache(
            repo, control, "parquet", "polynomial", nonce,
        ).digest()
    finally:
        snapshot_capture_module.capture_snapshot = original_capture
        repo.close(flush=False)
        control.close()
        store.close()

    shutil.rmtree(CONTROL_ROOT, ignore_errors=True)
    retained_snapshots = {
        digest for codec_states in states.values() for digest in codec_states.values()
    }
    snapshots_root = STORE_ROOT / "snapshots"
    for snapshot in tuple(snapshots_root.glob("*/*")):
        if snapshot.name not in retained_snapshots:
            shutil.rmtree(snapshot)
    for shard in tuple(snapshots_root.iterdir()):
        if shard.is_dir() and not any(shard.iterdir()):
            shard.rmdir()
    shutil.rmtree(STORE_ROOT / "managed", ignore_errors=True)
    shutil.rmtree(STORE_ROOT / ".dryml", ignore_errors=True)
    (STORE_ROOT / ".writer.lock").unlink(missing_ok=True)
    for path in ROOT.glob(".dryml-bootstrap-*.lock"):
        path.unlink()
    manifest = {
        "fixture_version": 1,
        "store_format_version": 3,
        "cache_format": "dryml.cached-dataset",
        "cache_format_version": CACHE_FORMAT_VERSION,
        "generator": "generate.py",
        "writer_qualification": {
            "pyarrow": pyarrow.__version__,
            "netCDF4": netCDF4.__version__,
        },
        "compatibility_scope": (
            "Reader compatibility for committed completed NumPy, Parquet, and NetCDF "
            "payloads generated with PyArrow 25.0.1 and netCDF4 1.7.4; assertion "
            "tests never regenerate codec bytes."
        ),
        "states": states,
        "authority_sha256": _authority_hashes(STORE_ROOT),
    }
    (ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="ascii",
    )


if __name__ == "__main__":
    main()
