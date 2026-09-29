"""Regenerate the checked-in synthetic Store v3 directory and archive fixtures."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import shutil
from uuid import UUID
import zipfile

from dryml.core import (
    ObjectId,
    ObjectRef,
    Repo,
    SaveAnnotations,
    SnapshotCapture,
    timestamp_to_seconds,
)
from dryml.core.metadata import encode_metadata_mapping
from dryml.core.repo_plan import apply_exact_reference_identity
import dryml.core.snapshot_capture as snapshot_capture_module
from dryml.core.store.dir import DirStore
from dryml.core.utils.graph.path import GraphPath
from dryml.environments import (
    DrymlRuntimeRecord,
    EnvironmentRecord,
    PackageRecord,
    PlatformRecord,
    PythonRecord,
)
from dryml.records import decode_record
from tests.store_v3_fixture_types import (
    ConflictingRequirementValue,
    EmptyRequirementValue,
    RoutedStateRoot,
    ValuedRequirementValue,
)


FIXTURE_ROOT = Path(__file__).resolve().parent
DIR_STORE = FIXTURE_ROOT / "dir-store"
ZIP_STORE = FIXTURE_ROOT / "zip-store.zip"
SAVED_AT = datetime(2024, 2, 3, 4, 5, 6, 250000, tzinfo=timezone.utc)
CREATED_AT = SAVED_AT - timedelta(days=1)


def _environment() -> EnvironmentRecord:
    """Return fixed public synthetic environment evidence."""

    return EnvironmentRecord(
        python=PythonRecord(
            "3.12.1",
            "CPython",
            executable="/opt/dryml-fixture/bin/python",
            prefix="/opt/dryml-fixture",
            base_prefix="/opt/python",
        ),
        platform=PlatformRecord(
            "FixtureOS",
            "1.0",
            "fixture-kernel",
            "fixture64",
            "FixtureOS-1.0-fixture64",
            os_name="posix",
            sys_platform="fixtureos",
            implementation_name="cpython",
            implementation_version="3.12.1",
            platform_python_implementation="CPython",
        ),
        distributions={
            "syntheticpkg": PackageRecord(
                "syntheticpkg",
                "2.1.0",
                location="/opt/dryml-fixture/lib/python/site-packages",
                installer="fixture",
                editable=False,
            ),
        },
        dryml=DrymlRuntimeRecord(
            version="0.3.0b1",
            git_revision="fixture-revision",
            execution_protocol="3",
            schema_versions={"store": "3"},
            features=("metadata",),
        ),
        kind="synthetic",
        tags=("fixture",),
        details={"environment_name": "dryml-public-fixture", "root": "/opt/dryml-fixture"},
    )


def _fix_identity(value, nonce: str, *, offset_seconds: int) -> None:
    """Replace a fresh Serializable identity with deterministic fixture evidence."""

    identity = ObjectId._trusted(("fixture",), UUID(nonce))
    reference = ObjectRef(value.definition, {GraphPath(): identity})
    apply_exact_reference_identity(
        value,
        reference,
        lineage_facts={GraphPath(): CREATED_AT + timedelta(seconds=offset_seconds)},
    )


def _placement(store: DirStore, state) -> dict:
    """Decode one fixture placement record for manifest expectations."""

    path = store.get_snapshot_directory(state) / "placement.json"
    return decode_record(json.loads(path.read_text(encoding="ascii"))).data


def _metadata_expectation(repo: Repo, store: DirStore, state) -> dict:
    """Return stable JSON expectations for one published snapshot."""

    metadata = repo.get_snapshot_metadata(state, store=store)
    placement = _placement(store, state)
    return {
        "state_ref_digest": state.digest(),
        "object_ref_digest": state.object.digest(),
        "saved_at_unix_seconds": timestamp_to_seconds(metadata.saved_at),
        "environment_status": metadata.environment_status,
        "environment": None if metadata.environment is None else metadata.environment.to_data(),
        "requirements_status": metadata.requirements_status,
        "requirements_coverage": metadata.requirements_coverage,
        "requirements": None if metadata.requirements is None else metadata.requirements.to_data(),
        "diagnostics": [list(value) for value in metadata.diagnostics],
        "captured_object_annotations": None if metadata.captured_object_annotations is None else encode_metadata_mapping(metadata.captured_object_annotations),
        "captured_state_annotations": None if metadata.captured_state_annotations is None else encode_metadata_mapping(metadata.captured_state_annotations),
        "current_object_annotations": None if repo.get_metadata(state.object, store=store) is None else encode_metadata_mapping(repo.get_metadata(state.object, store=store)),
        "current_state_annotations": None if repo.get_metadata(state, store=store) is None else encode_metadata_mapping(repo.get_metadata(state, store=store)),
        "lineages": {
            str(path): {
                "object_ref_digest": lineage.object_ref.digest(),
                "creation_status": lineage.creation_status,
                "created_at_unix_seconds": None if lineage.created_at is None else timestamp_to_seconds(lineage.created_at),
            }
            for path, lineage in metadata.lineages.items()
        },
        "local_paths": [
            {"display": str(GraphPath.from_data(item["path"])), "data": item["path"]}
            for item in placement["local"]
        ],
        "children": [
            {
                "path": str(GraphPath.from_data(item["path"])),
                "path_data": item["path"],
                "state_ref_digest": item["state_ref_digest"],
            }
            for item in placement["children"]
        ],
    }


def _authority_hashes(root: Path) -> dict[str, str]:
    """Hash every framework-owned fixture file relative to its Store root."""

    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file() and ".dryml" not in path.parts
    }


def _write_archive(source: Path, destination: Path) -> None:
    """Write a deterministic committed ZipStore archive from one Store root."""

    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(source.rglob("*")):
            if not path.is_file() or ".dryml" in path.parts:
                continue
            info = zipfile.ZipInfo(path.relative_to(source).as_posix(), (2024, 2, 3, 4, 5, 6))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def main() -> None:
    """Regenerate fixtures and their semantic expectation manifest."""

    if DIR_STORE.exists():
        shutil.rmtree(DIR_STORE)
    ZIP_STORE.unlink(missing_ok=True)

    store = DirStore(DIR_STORE)
    repo = Repo(store, save_routing="per-object")
    environment = _environment()
    original_capture = snapshot_capture_module.capture_snapshot

    def capture_with_fixed_clock(plan, *, observer=None):
        return original_capture(plan, observer=observer, clock=lambda: SAVED_AT)

    snapshot_capture_module.capture_snapshot = capture_with_fixed_clock
    try:
        empty = EmptyRequirementValue("empty-payload", repo=repo)
        _fix_identity(empty, "00000000-0000-0000-0000-000000000001", offset_seconds=1)
        empty_state = repo.save_object(empty, _snapshot_observer=lambda: environment)

        valued = ValuedRequirementValue("valued-payload", repo=repo)
        _fix_identity(valued, "00000000-0000-0000-0000-000000000002", offset_seconds=2)
        valued_state = repo.save_object(
            valued,
            annotations=SaveAnnotations(
                object={
                    "nested": {"stage": "beta", "values": [1, True, None]},
                    "tuple": ("preserved", 2),
                },
                state={},
            ),
            _snapshot_observer=lambda: environment,
        )
        repo.set_metadata(valued_state.object, {"current": "edited-after-capture"})
        repo.delete_metadata(valued_state)

        conflicting = ConflictingRequirementValue("conflict-payload", repo=repo)
        _fix_identity(conflicting, "00000000-0000-0000-0000-000000000003", offset_seconds=3)
        conflict_state = repo.save_object(
            conflicting,
            annotations=SaveAnnotations(object={}),
            _snapshot_observer=lambda: environment,
        )

        routed = RoutedStateRoot(valued_state, repo=repo)

        def capture_unavailable_requirements(plan, *, observer=None):
            captured = capture_with_fixed_clock(plan, observer=observer)
            return SnapshotCapture(
                captured.lineages,
                captured.saved_at,
                captured.environment,
                captured.environment_status,
                None,
                "unavailable",
                "incomplete",
                (*captured.diagnostics, (
                    "dryml.environments.requirement_collection_unavailable",
                    "environment requirement collection unavailable",
                )),
            )

        snapshot_capture_module.capture_snapshot = capture_unavailable_requirements
        routed_state = repo.save_object(
            routed,
            annotations=SaveAnnotations(state={"route": {"kind": "exact-child"}}),
            _snapshot_observer=lambda: environment,
        )
    finally:
        snapshot_capture_module.capture_snapshot = original_capture

    states = {
        "empty": empty_state,
        "value": valued_state,
        "conflict": conflict_state,
        "unavailable_routed": routed_state,
    }
    expectations = {
        name: _metadata_expectation(repo, store, state)
        for name, state in states.items()
    }
    store.close()
    shutil.rmtree(DIR_STORE / ".dryml", ignore_errors=True)
    (DIR_STORE / ".writer.lock").unlink(missing_ok=True)
    for lock_path in FIXTURE_ROOT.glob(".dryml-bootstrap-*.lock"):
        lock_path.unlink()
    _write_archive(DIR_STORE, ZIP_STORE)
    manifest = {
        "fixture_version": 1,
        "store_format_version": 3,
        "record_contract_version": "1.1",
        "saved_at_unix_seconds": timestamp_to_seconds(SAVED_AT),
        "snapshots": expectations,
        "authority_sha256": _authority_hashes(DIR_STORE),
        "zip_sha256": hashlib.sha256(ZIP_STORE.read_bytes()).hexdigest(),
        "compatibility_scope": "framework-owned Store v3 records and placement; author payload codecs are excluded",
    }
    (FIXTURE_ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )


if __name__ == "__main__":
    main()
