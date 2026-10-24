"""Intact old-format refusal and separately authored current semantic fixtures."""

import hashlib
import json
from pathlib import Path
import shutil
import zipfile
from dataclasses import replace

import dill
import pytest

from dryml.core import (Object, Repo, SaveAnnotations, Serializable, field,
                        timestamp_from_seconds, timestamp_to_seconds)
from dryml.core.metadata import encode_metadata_mapping, decode_metadata_mapping
from dryml.environments import EnvironmentRecord, EnvironmentRequirement
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


def _copy_old(tmp_path: Path, kind: str):
    """Copy frozen authority without opening it or changing source bytes."""

    if kind == "directory":
        destination = tmp_path / "dir-store"
        shutil.copytree(FIXTURE_ROOT / "dir-store", destination)
        return destination
    destination = tmp_path / "zip-store.zip"
    shutil.copy2(FIXTURE_ROOT / "zip-store.zip", destination)
    return destination


class CurrentPayload(Serializable):
    """Deliberately authored current-format payload, independent of old records."""

    def __init__(self, value):
        self.value = value

    def save_state_to_dir_imp(self, directory, *, codec):
        Path(directory, "value.txt").write_text(self.value, encoding="ascii")


class CurrentWrapper(Object):
    """Identity-free root preserving an exact materializing child projection."""

    def __init__(self, child):
        self.child = child


def _bytes(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def _current(tmp_path, kind, monkeypatch):
    """Author current records with the frozen manifest's meaningful metadata facts.

    The manifest is only detached expected data. No old authority is decoded,
    rewritten or admitted. Capture evidence is controlled at its owning seam.
    """
    import dryml.core.snapshot_capture as capture
    store = DirStore(tmp_path / "current", query_index="none")
    repo = Repo(store)
    expected = _manifest()["snapshots"]
    original = capture.capture_snapshot
    nodes, live = {}, {}
    for name in ("value", "empty", "conflict", "unavailable_routed"):
        facts = expected[name]
        def observe(plan, *, observer=None, facts=facts):
            result = original(plan, observer=lambda: EnvironmentRecord.from_data(facts["environment"]))
            lineages = {path: replace(lineage, creation_status=facts["lineages"][str(path)]["creation_status"],
                        created_at=timestamp_from_seconds(facts["lineages"][str(path)]["created_at_unix_seconds"]))
                        for path, lineage in result.lineages.items()}
            return replace(result, lineages=lineages,
                           saved_at=timestamp_from_seconds(facts["saved_at_unix_seconds"]),
                           requirements=None if facts["requirements"] is None else EnvironmentRequirement.from_data(facts["requirements"]),
                           requirements_status=facts["requirements_status"],
                           requirements_coverage=facts["requirements_coverage"],
                           diagnostics=tuple(tuple(d) for d in facts["diagnostics"]))
        monkeypatch.setattr(capture, "capture_snapshot", observe)
        value = CurrentWrapper(live["value"], repo=repo) if name == "unavailable_routed" else CurrentPayload(name, repo=repo)
        annotations = SaveAnnotations(**{scope: None if facts[f"captured_{scope}_annotations"] is None
                                         else decode_metadata_mapping(facts[f"captured_{scope}_annotations"])
                                         for scope in ("object", "state")})
        nodes[name] = repo.save_object(value, annotations=annotations, alias=name)
        live[name] = value
        for scope, target in (("object", nodes[name].object_view()), ("state", nodes[name])):
            current = facts[f"current_{scope}_annotations"]
            if current is None:
                repo.delete_metadata(target)
            else:
                repo.set_metadata(target, decode_metadata_mapping(current))
    monkeypatch.setattr(capture, "capture_snapshot", original)
    if kind == "directory":
        store.close()
        return DirStore.open_existing(tmp_path / "current"), nodes
    path = tmp_path / "current.zip"
    archive = ZipStore(path)
    try:
        keys = {state.key: state for node in nodes.values() for state in node.state_spec().values()}
        for key, state in keys.items():
            with store.read_state(key) as source:
                stage = archive.create_local_state_staging()
                shutil.copytree(source.data_directory, Path(stage, "data"), dirs_exist_ok=True)
                manifest = source.manifest
            installed = archive.install_state(state.creation_cdef, stage, codec=manifest.codec,
                                              codec_version=manifest.codec_version,
                                              interpretation=manifest.interpretation)
            assert installed.state.key == key and installed.durability == "buffered"
        for name, node in nodes.items():
            record = archive.publish_node(node, kind="saved")
            from dryml.core.store.records import NodeNameRecord
            archive.compare_and_set_node_name(NodeNameRecord(name, "saved", record.digest), expected=None)
        # Transport only current immutable capture/lineage metadata. Durable user
        # roots/grants stay with the directory authority that actually owns them.
        records = [(p.relative_to(store.base_dir), p.read_bytes())
                   for scope in ("capture", "lineage")
                   for p in Path(store.base_dir, "metadata", scope).rglob("*.record")]
        from dryml.core.store.authority import PreparedChange
        archive._prepare_and_apply(lambda: ([PreparedChange(str(p), None, b) for p, b in records], ("metadata",), None))
        for node in nodes.values():
            for target in (node.object_view(), node):
                values = store.read_metadata(target)
                if values is not None:
                    archive.write_metadata(target, values)
        archive.commit()
    finally:
        archive.close()
        store.close()
    return ZipStore.open_existing(path), nodes


def _encoded_or_none(values):
    """Encode a present mapping while retaining absent versus empty semantics."""

    return None if values is None else encode_metadata_mapping(values)


def _assert_semantics(store, manifest: dict, nodes) -> None:
    """Compare decoded Store authority with all semantic manifest fields."""

    repo = Repo(store)
    states = repo.query().saved().collect()
    assert len(states) == 4
    for name, expected in manifest["snapshots"].items():
        record = store.read_node_name(name)
        state = store.read_node(record.root_digest, kind="saved")
        assert state.graph_equal(nodes[name])
        metadata = repo.get_snapshot_metadata(state, store=store)
        assert metadata.state_ref.graph_equal(state)
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
        assert _encoded_or_none(repo.get_metadata(state.object_view(), store=store)) == expected["current_object_annotations"]
        assert _encoded_or_none(repo.get_metadata(state, store=store)) == expected["current_state_annotations"]
        assert {
            str(path): {
                "creation_status": lineage.creation_status,
                "created_at_unix_seconds": None if lineage.created_at is None else timestamp_to_seconds(lineage.created_at),
            }
            for path, lineage in metadata.lineages.items()
        } == {p: {k: v for k, v in row.items() if k != "object_ref_digest"} for p, row in expected["lineages"].items()}
        for path, lineage in metadata.lineages.items():
            assert lineage.object_ref.graph_equal(state.at(path).object_view())
        for child in expected["children"]:
            assert state.at(GraphPath.from_data(child["path_data"])).graph_equal(nodes["value"])
    value = repo.get_snapshot_metadata(nodes["value"], store=store).captured_object_annotations
    assert type(value["nested"]["values"][0]) is int
    assert type(value["nested"]["values"][1]) is bool
    assert value["nested"]["values"][2] is None and type(value["tuple"]) is tuple


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
def test_fixture_copies_decode_to_semantic_manifest_without_changing_sources(tmp_path, kind, monkeypatch):
    """Refuse old bytes intact, then independently prove current semantic fidelity."""

    manifest = _manifest()
    before = _fixture_hashes()
    copied = _copy_old(tmp_path, kind)
    copied_before = _bytes(copied) if kind == "directory" else copied.read_bytes()
    backend = DirStore if kind == "directory" else ZipStore
    for opener in (backend.open_existing, backend):
        with pytest.raises(StoreAuthorityError) as raised:
            opener(copied)
        assert "Unsupported Store format version 3" in str(raised.value.__cause__)
        assert (_bytes(copied) if kind == "directory" else copied.read_bytes()) == copied_before
    store, nodes = _current(tmp_path, kind, monkeypatch)
    try:
        _assert_semantics(store, manifest, nodes)
    finally:
        store.close()
    assert _fixture_hashes() == before


@pytest.mark.parametrize("kind", ("directory", "archive"))
def test_inspection_and_query_do_not_open_payloads_but_validation_does(tmp_path, kind, monkeypatch):
    """Metadata inspection/query use authority only; payload validation is separate."""

    manifest = _manifest()
    before = _fixture_hashes()
    store, nodes = _current(tmp_path, kind, monkeypatch)
    repo = Repo(store)
    original_validate = store._authenticate_state_data

    def fail_payload_read(*_args, **_kwargs):
        raise AssertionError("metadata inspection opened local payload bytes")

    monkeypatch.setattr(store, "_authenticate_state_data", fail_payload_read)
    try:
        states = repo.query().saved().collect()
        assert len(states) == len(nodes)
        conflicts = repo.query().where(
                field("snapshot", "requirements_status").eq("conflict")
            ).saved().collect()
        assert len(conflicts) == 1 and next(iter(conflicts)).graph_equal(nodes["conflict"])
        _assert_semantics(store, manifest, nodes)
        with pytest.raises(AssertionError, match="opened local payload"):
            with store.read_state(nodes["value"].state.key):
                pass
        monkeypatch.setattr(store, "_authenticate_state_data", original_validate)
        for name in ("value", "empty", "conflict"):
            with store.read_state(nodes[name].state.key) as source:
                assert Path(source.data_directory, "value.txt").read_text() == name
                assert source.manifest.key == nodes[name].state.key
    finally:
        store.close()
    assert _fixture_hashes() == before


def test_corrupt_and_v2_copies_fail_without_rewriting_input(tmp_path, monkeypatch):
    """Malformed authority and unsupported v2 gates are rejected in place."""

    from tests.core.test_repo_aliases import _saved_digest
    corrupt, nodes = _current(tmp_path, "directory", monkeypatch)
    metadata_path = Path(corrupt._node_metadata_path("capture", _saved_digest(nodes["value"])))
    metadata_path.write_bytes(metadata_path.read_bytes() + b"\ncorrupt")
    corrupt_before = metadata_path.read_bytes()
    try:
        with pytest.raises(StoreAuthorityError):
            Repo(corrupt).get_snapshot_metadata(nodes["value"])
    finally:
        corrupt.close()
    assert metadata_path.read_bytes() == corrupt_before
    assert _fixture_hashes()[0] == _manifest()["authority_sha256"]

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
    before = _bytes(v2_root)
    with pytest.raises(StoreAuthorityError):
        DirStore.open_existing(v2_root)
    assert gate.read_bytes() == v2_bytes and _bytes(v2_root) == before

    archive = tmp_path / "corrupt.zip"
    shutil.copy2(FIXTURE_ROOT / "zip-store.zip", archive)
    archive.write_bytes(archive.read_bytes()[:64])
    archive_before = archive.read_bytes()
    with pytest.raises(StoreAuthorityError, match="archive"):
        ZipStore.open_existing(archive)
    assert archive.read_bytes() == archive_before
