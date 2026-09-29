"""Compatibility contracts for the checked-in synthetic Store v3 fixtures."""

import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import dill
import pytest

from dryml.core import Repo, field, timestamp_to_seconds
from dryml.core.metadata import encode_metadata_mapping
from dryml.core.store.dir import DirStore
from dryml.core.store.store import StoreAuthorityError
from dryml.core.store.zip import ZipStore
from dryml.core.utils.graph.path import GraphPath


FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "store_v3"


def _manifest() -> dict:
    """Read detached expectations without opening a fixture Store."""

    return json.loads((FIXTURE_ROOT / "manifest.json").read_text(encoding="ascii"))


def _fixture_hashes() -> tuple[dict[str, str], str]:
    """Return current source fixture hashes for immutability assertions."""

    root = FIXTURE_ROOT / "dir-store"
    authority = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    archive = hashlib.sha256((FIXTURE_ROOT / "zip-store.zip").read_bytes()).hexdigest()
    return authority, archive


def _open_copy(tmp_path: Path, kind: str):
    """Copy one read-only fixture and return an independently writable Store."""

    if kind == "directory":
        destination = tmp_path / "dir-store"
        shutil.copytree(FIXTURE_ROOT / "dir-store", destination)
        return DirStore.open_existing(destination)
    destination = tmp_path / "zip-store.zip"
    shutil.copy2(FIXTURE_ROOT / "zip-store.zip", destination)
    return ZipStore.open_existing(destination)


def _encoded_or_none(values):
    """Encode a present mapping while retaining absent versus empty semantics."""

    return None if values is None else encode_metadata_mapping(values)


def _assert_semantics(store, manifest: dict) -> None:
    """Compare decoded Store authority with all semantic manifest fields."""

    repo = Repo(store)
    states = {
        record.state_ref.digest(): record.state_ref
        for record in store.iter_state_ref_records()
    }
    assert set(states) == {
        expected["state_ref_digest"]
        for expected in manifest["snapshots"].values()
    }

    for expected in manifest["snapshots"].values():
        state = states[expected["state_ref_digest"]]
        metadata = repo.get_snapshot_metadata(state, store=store)
        assert state.object.digest() == expected["object_ref_digest"]
        assert timestamp_to_seconds(metadata.saved_at) == expected["saved_at_unix_seconds"]
        assert metadata.environment_status == expected["environment_status"]
        assert metadata.environment.to_data() == expected["environment"]
        assert metadata.requirements_status == expected["requirements_status"]
        assert metadata.requirements_coverage == expected["requirements_coverage"]
        assert (
            None if metadata.requirements is None else metadata.requirements.to_data()
        ) == expected["requirements"]
        assert [list(item) for item in metadata.diagnostics] == expected["diagnostics"]
        assert _encoded_or_none(metadata.captured_object_annotations) == expected["captured_object_annotations"]
        assert _encoded_or_none(metadata.captured_state_annotations) == expected["captured_state_annotations"]
        assert _encoded_or_none(repo.get_metadata(state.object, store=store)) == expected["current_object_annotations"]
        assert _encoded_or_none(repo.get_metadata(state, store=store)) == expected["current_state_annotations"]
        assert {
            str(path): {
                "object_ref_digest": lineage.object_ref.digest(),
                "creation_status": lineage.creation_status,
                "created_at_unix_seconds": None if lineage.created_at is None else timestamp_to_seconds(lineage.created_at),
            }
            for path, lineage in metadata.lineages.items()
        } == expected["lineages"]
        for child in expected["children"]:
            assert state.at(GraphPath.from_data(child["path_data"])).digest() == child["state_ref_digest"]


def test_fixture_manifest_hashes_cover_directory_and_committed_archive():
    """The manifest names every immutable authority byte and the exact archive."""

    manifest = _manifest()
    authority, archive = _fixture_hashes()

    assert manifest["store_format_version"] == 3
    assert manifest["record_contract_version"] == "1.1"
    assert authority == manifest["authority_sha256"]
    assert archive == manifest["zip_sha256"]
    with zipfile.ZipFile(FIXTURE_ROOT / "zip-store.zip") as source:
        assert {
            info.filename: hashlib.sha256(source.read(info)).hexdigest()
            for info in source.infolist()
        } == manifest["authority_sha256"]


@pytest.mark.parametrize("kind", ("directory", "archive"))
def test_fixture_copies_decode_to_semantic_manifest_without_changing_sources(tmp_path, kind):
    """Directory and archive copies expose equal metadata and exact references."""

    manifest = _manifest()
    before = _fixture_hashes()
    store = _open_copy(tmp_path, kind)
    try:
        _assert_semantics(store, manifest)
    finally:
        store.close()
    assert _fixture_hashes() == before


@pytest.mark.parametrize("kind", ("directory", "archive"))
def test_inspection_and_query_do_not_open_payloads_but_validation_does(tmp_path, kind, monkeypatch):
    """Metadata inspection/query use authority only; payload validation is separate."""

    manifest = _manifest()
    store = _open_copy(tmp_path, kind)
    repo = Repo(store)
    expected_digests = {
        value["state_ref_digest"] for value in manifest["snapshots"].values()
    }
    original_validate = store._validate_local_state_dir

    def fail_payload_read(*_args, **_kwargs):
        raise AssertionError("metadata inspection opened local payload bytes")

    monkeypatch.setattr(store, "_validate_local_state_dir", fail_payload_read)
    try:
        assert {state.digest() for state in repo.references().state_refs()} == expected_digests
        assert {
            state.digest()
            for state in repo.references().where(
                field("snapshot", "requirements_status").eq("conflict")
            ).state_refs()
        } == {manifest["snapshots"]["conflict"]["state_ref_digest"]}
        for state in repo.references().state_refs():
            assert repo.get_snapshot_metadata(state, store=store).state_ref == state
        monkeypatch.setattr(store, "_validate_local_state_dir", original_validate)
        states = {
            state.digest(): state for state in repo.references().state_refs()
        }
        for expected in manifest["snapshots"].values():
            state = states[expected["state_ref_digest"]]
            for path_record in expected["local_paths"]:
                path = GraphPath.from_data(path_record["data"])
                assert store.validate_local_state(state, path).state_hash == state.states[path]
    finally:
        store.close()


def test_corrupt_and_v2_copies_fail_without_rewriting_input(tmp_path):
    """Malformed authority and unsupported v2 gates are rejected in place."""

    corrupt_root = tmp_path / "corrupt"
    shutil.copytree(FIXTURE_ROOT / "dir-store", corrupt_root)
    metadata_path = next(corrupt_root.glob("snapshots/*/*/metadata.json"))
    metadata_path.write_bytes(metadata_path.read_bytes() + b"\ncorrupt")
    corrupt_before = metadata_path.read_bytes()
    corrupt = DirStore.open_existing(corrupt_root)
    try:
        with pytest.raises(StoreAuthorityError, match="snapshot JSON authority|metadata association"):
            tuple(corrupt.iter_state_ref_records())
    finally:
        corrupt.close()
    assert metadata_path.read_bytes() == corrupt_before

    v2_root = tmp_path / "v2"
    shutil.copytree(FIXTURE_ROOT / "dir-store", v2_root)
    gate = v2_root / "store-format.record"
    v2_bytes = (
        b"DRYML-STORE-RECORD/store-format/1\n"
        + dill.dumps({
            "schema": "store-format",
            "version": 1,
            "format_version": 2,
        }, protocol=5)
    )
    gate.write_bytes(v2_bytes)
    with pytest.raises(StoreAuthorityError, match="Store format"):
        DirStore.open_existing(v2_root)
    assert gate.read_bytes() == v2_bytes

    archive = tmp_path / "corrupt.zip"
    shutil.copy2(FIXTURE_ROOT / "zip-store.zip", archive)
    archive.write_bytes(archive.read_bytes()[:64])
    archive_before = archive.read_bytes()
    with pytest.raises(StoreAuthorityError, match="archive"):
        ZipStore.open_existing(archive)
    assert archive.read_bytes() == archive_before
