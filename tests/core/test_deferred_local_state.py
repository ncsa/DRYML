from pathlib import Path
import builtins

import pytest

pytestmark = pytest.mark.usefixtures("fixed_snapshot_environment")

from dryml.core import Repo, SaveRouting, Selector, Serializable
from dryml.core.repo import RepoLoadError, RepoSaveError
from dryml.core.store.dir import DirStore
from dryml.core.store.records import StoreRecordError
from dryml.core.store.store import StoreCapabilityError
from dryml.core.store.zip import ZipStore
from dryml.core.utils.graph.path import GraphPath


class DeferredPayload(Serializable):
    """Serializable fixture that defers one immutable payload file."""

    reads = 0

    def __init__(self, value=b"payload"):
        self.value = value

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        Path(dest_dir, "metadata").write_text("metadata", encoding="ascii")
        Path(dest_dir, "payload.bin").write_bytes(self.value)

    def deferred_state_payload_paths(self, data_dir, *, codec):
        return ("payload.bin",)

    def restore_state_from_dir_imp(self, src_dir, *, codec):
        Path(src_dir, "metadata").read_text(encoding="ascii")


class AllEagerPayload(Serializable):
    """Serializable fixture relying on the default all-eager hook."""

    def __init__(self, value=b"payload"):
        self.value = value

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        Path(dest_dir, "payload.bin").write_bytes(self.value)

    def restore_state_from_dir_imp(self, src_dir, *, codec):
        self.value = Path(src_dir, "payload.bin").read_bytes()


def test_deferred_manifest_authenticates_inventory_paths_and_delays_exact_load_hash(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store")
    writer = Repo(store)
    state = writer.save_object(DeferredPayload(repo=writer))
    source = store.open_local_state(state, GraphPath())

    assert source.manifest.version == 3
    assert source.manifest.deferred_paths == ("payload.bin",)

    payload = Path(source.handle, "data", "payload.bin")
    payload.write_bytes(b"corrupt")
    with pytest.raises(StoreRecordError):
        source.manifest.validate_payload(Path(source.handle, "data"))

    original_open = builtins.open

    def forbid_payload_reads(path, *args, **kwargs):
        if Path(path) == payload and "r" in (args[0] if args else kwargs.get("mode", "r")):
            raise AssertionError("exact restore read deferred payload bytes")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", forbid_payload_reads)
    loaded = Repo(store).load_state_ref(state, reuse_live="never")

    assert isinstance(loaded, DeferredPayload)


def test_default_serializable_hook_writes_an_all_eager_v3_manifest(tmp_path):
    store = DirStore(tmp_path / "store")
    writer = Repo(store)
    state = writer.save_object(AllEagerPayload(repo=writer))
    source = store.open_local_state(state, GraphPath())

    assert source.manifest.version == 3
    assert source.manifest.deferred_paths == ()
    Path(source.handle, "data", "payload.bin").write_bytes(b"corrupt")
    with pytest.raises(RepoLoadError, match="preflight is incomplete"):
        Repo(store).load_state_ref(state, reuse_live="never")


@pytest.mark.parametrize("replacement", [None, b"short", "directory"])
def test_deferred_payload_type_and_size_fail_before_restore(tmp_path, replacement):
    store = DirStore(tmp_path / "store")
    writer = Repo(store)
    state = writer.save_object(DeferredPayload(repo=writer))
    source = store.open_local_state(state, GraphPath())
    payload = Path(source.handle, "data", "payload.bin")
    payload.unlink()
    if replacement == "directory":
        payload.mkdir()
    elif replacement is not None:
        payload.write_bytes(replacement)

    with pytest.raises(RepoLoadError, match="preflight is incomplete"):
        Repo(store).load_state_ref(state, reuse_live="never")


def test_deferred_payload_corruption_waits_for_exact_restore_but_not_fork(tmp_path):
    source_store = DirStore(tmp_path / "source")
    target_store = DirStore(tmp_path / "target")
    writer = Repo([source_store, target_store])
    state = writer.save_object(DeferredPayload(repo=writer), store=source_store)
    source = source_store.open_local_state(state, GraphPath())
    Path(source.handle, "data", "payload.bin").write_bytes(b"corrupt")

    assert isinstance(Repo(source_store).load_state_ref(state, reuse_live="never"), DeferredPayload)
    with pytest.raises(RepoLoadError, match="Fork source lacks"):
        writer.fork_state_ref(state, store=target_store, source_store=source_store)


def test_every_exact_restore_entry_point_uses_deferred_validation(tmp_path):
    store = DirStore(tmp_path / "store")
    writer = Repo(store)
    value = DeferredPayload(repo=writer)
    state = writer.save_object(value)
    source = store.open_local_state(state, GraphPath())
    Path(source.handle, "data", "payload.bin").write_bytes(b"corrupt")

    selected = Repo([DirStore(tmp_path / "other"), store]).load_state_ref(
        state, reuse_live="never", source_store=store,
    )
    boundary, = Repo(store).materialize_boundary(
        (state,), cache="none", reuse_live="never",
    )
    writer.restore_state_ref_into(value, state)

    assert isinstance(selected, DeferredPayload)
    assert isinstance(boundary, DeferredPayload)
    assert value.last_state_ref is state


def test_zip_store_rejects_deferred_manifest_before_snapshot_install(tmp_path):
    source = DirStore(tmp_path / "source")
    target = ZipStore(tmp_path / "target.zip")
    writer = Repo(source)
    state = writer.save_object(DeferredPayload(repo=writer))

    with pytest.raises(StoreCapabilityError, match="does not support deferred"):
        Repo([source, target]).fork_state_ref(state, store=target, source_store=source)
    assert tuple(target.iter_state_ref_records()) == ()
    target.close()


def test_zip_store_exact_load_remains_eager_for_all_eager_manifests(tmp_path):
    store = ZipStore(tmp_path / "store.zip")
    writer = Repo(store)
    state = writer.save_object(AllEagerPayload(repo=writer))
    source = store.open_local_state(state, GraphPath())
    Path(source.handle, "data", "payload.bin").write_bytes(b"corrupt")

    with pytest.raises(RepoLoadError, match="preflight is incomplete"):
        Repo(store).load_state_ref(state, reuse_live="never")
    store.close()


def test_unsupported_replica_rejects_deferred_state_after_staging_before_install(tmp_path, monkeypatch):
    primary = DirStore(tmp_path / "primary")
    replica = ZipStore(tmp_path / "replica.zip")
    staged = []
    original_create = primary.create_local_state_staging

    def record_staging():
        result = original_create()
        staged.append(result)
        return result

    monkeypatch.setattr(primary, "create_local_state_staging", record_staging)
    repo = Repo(
        [primary, replica],
        save_routing=SaveRouting(
            ((Selector(DeferredPayload), primary), (Selector(DeferredPayload), replica)),
            match_mode="all",
        ),
    )

    with pytest.raises(RepoSaveError, match="Save publication failed") as raised:
        repo.save_object(DeferredPayload(repo=repo))

    assert staged
    assert isinstance(raised.value.__cause__, StoreCapabilityError)
    assert tuple(primary.iter_state_ref_records()) == ()
    assert tuple(replica.iter_state_ref_records()) == ()
    replica.close()
