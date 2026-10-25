"""Current full-key authentication supersedes snapshot-v3 deferred proofs.

Codec interpretation can record deferred logical files, but StateVariantManifest
v1 authenticates every byte before use. Metadata inspection remains payload-free.
No old manifest or snapshot-cut proof is promoted into current authority.
"""

from pathlib import Path
import builtins
from contextlib import contextmanager
import zipfile

import pytest

from dryml.core import Repo, SaveRouting, Selector, Serializable
from dryml.core.node_prepare import NodePreparation
from dryml.core.repo import RepoLoadError
from dryml.core.store.dir import DirStore
from dryml.core.store.records import LocalStateManifest
from dryml.core.store.store import StoreCapabilityError, StoreAuthorityError
from dryml.core.store.zip import ZipStore
from dryml.core.materialization import build_exact_state_load_plan, execute_exact_state_load_plan

pytestmark = pytest.mark.usefixtures("fixed_snapshot_environment")


class DeferredPayload(Serializable):
    """Codec declaring one logically deferred file; Core still authenticates it."""
    reads = 0

    def __init__(self, value=b"payload"):
        self.value = value

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        Path(dest_dir, "metadata").write_text("metadata", encoding="ascii")
        Path(dest_dir, "payload.bin").write_bytes(self.value)

    def deferred_state_payload_paths(self, data_dir, *, codec):
        return ("payload.bin",)

    def restore_state_from_dir_imp(self, src_dir, *, codec):
        type(self).reads += 1
        Path(src_dir, "metadata").read_text(encoding="ascii")


class AllEagerPayload(Serializable):
    """Codec relying on the default all-eager interpretation."""

    def __init__(self, value=b"payload"):
        self.value = value

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        Path(dest_dir, "payload.bin").write_bytes(self.value)

    def restore_state_from_dir_imp(self, src_dir, *, codec):
        self.value = Path(src_dir, "payload.bin").read_bytes()


def _manifest(store, saved):
    with store.read_state(saved.state.key) as source:
        return source.manifest


def test_current_manifest_authenticates_even_logically_deferred_files(tmp_path):
    store = DirStore(tmp_path / "store")
    state = Repo(store).save_object(DeferredPayload())
    manifest = _manifest(store, state)
    assert manifest.version == 1
    assert manifest.interpretation["deferred_paths"] == ["payload.bin"]
    assert manifest.deferred_paths == ()
    Path(store.state_data_path(state.state.key), "payload.bin").write_bytes(b"corrupt")
    DeferredPayload.reads = 0
    with pytest.raises(RepoLoadError, match="preflight is incomplete"):
        Repo(store).load_state_ref(state, reuse_live="never")
    assert DeferredPayload.reads == 0


def test_default_serializable_hook_writes_an_all_eager_current_manifest(tmp_path):
    store = DirStore(tmp_path / "store")
    state = Repo(store).save_object(AllEagerPayload())
    manifest = _manifest(store, state)
    assert manifest.version == 1 and manifest.interpretation["deferred_paths"] == []
    Path(store.state_data_path(state.state.key), "payload.bin").write_bytes(b"corrupt")
    with pytest.raises(RepoLoadError, match="preflight is incomplete"):
        Repo(store).load_state_ref(state, reuse_live="never")


@pytest.mark.parametrize("replacement", [None, b"short", "directory"])
def test_deferred_payload_type_and_size_fail_before_restore(tmp_path, replacement):
    store = DirStore(tmp_path / "store")
    state = Repo(store).save_object(DeferredPayload())
    payload = Path(store.state_data_path(state.state.key), "payload.bin")
    payload.unlink()
    if replacement == "directory":
        payload.mkdir()
    elif replacement is not None:
        payload.write_bytes(replacement)
    DeferredPayload.reads = 0
    with pytest.raises(RepoLoadError, match="preflight is incomplete"):
        Repo(store).load_state_ref(state, reuse_live="never")
    assert DeferredPayload.reads == 0


def test_corrupt_selected_payload_rejects_restore_and_cross_store_reuse(tmp_path):
    source, target = (DirStore(tmp_path / name) for name in ("source", "target"))
    repo = Repo([source, target])
    state = repo.save_object(DeferredPayload(), store=source)
    Path(source.state_data_path(state.state.key), "payload.bin").write_bytes(b"corrupt")
    with pytest.raises(RepoLoadError):
        repo.load_state_ref(state, reuse_live="never", source_store=source)
    with pytest.raises(StoreAuthorityError):
        repo.save_object(DeferredPayload(), states=state.state_spec(), store=target, source_store=source)
    assert not tuple(Path(target.base_dir, "membership", "saved").glob("*/*.record"))


def test_every_exact_restore_entry_point_authenticates_current_payload(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    value = DeferredPayload()
    state = repo.save_object(value)
    plan = build_exact_state_load_plan(repo, state)
    assert not hasattr(plan.actions[0], "payload")
    Path(store.state_data_path(state.state.key), "payload.bin").write_bytes(b"corrupt")
    DeferredPayload.reads = 0
    for load in (
        lambda: Repo([DirStore(tmp_path / "other"), store]).load_state_ref(state, reuse_live="never", source_store=store),
        lambda: Repo(store).materialize_boundary((state,), cache="none", reuse_live="never"),
        lambda: repo.restore_state_ref_into(value, state),
        lambda: execute_exact_state_load_plan(repo, plan, reuse_live="never"),
    ):
        with pytest.raises(RepoLoadError):
            load()
    assert DeferredPayload.reads == 0 and not getattr(value, "_restore_failed", False)


def test_zip_store_rejects_deferred_manifest_before_snapshot_install(tmp_path):
    target = ZipStore(tmp_path / "target.zip")
    try:
        with pytest.raises(StoreCapabilityError, match="durable"):
            Repo(target).save_object(DeferredPayload())
        assert not Path(target.archive_path).exists()
    finally:
        target.close()


def test_zip_store_exact_load_remains_eager_for_all_eager_manifests(tmp_path):
    """A compatible committed archive lends current bytes only until closure."""
    source = DirStore(tmp_path / "source", query_index="none")
    state = Repo(source).save_object(AllEagerPayload())
    archive = tmp_path / "store.zip"
    with zipfile.ZipFile(archive, "w") as output:
        for path in Path(source.base_dir).rglob("*"):
            if path.is_file():
                output.write(path, path.relative_to(source.base_dir))
    source.close()
    store = ZipStore.open_existing(archive)
    data = Path(store.state_data_path(state.state.key))
    try:
        assert Repo(store).load_state_ref(state, reuse_live="never").value == b"payload"
        Path(data, "payload.bin").write_bytes(b"corrupt")
        with pytest.raises(RepoLoadError):
            Repo(store).load_state_ref(state, reuse_live="never")
    finally:
        store.close()
    assert not data.exists()


def test_unsupported_replica_rejects_before_staging_or_install(tmp_path, monkeypatch):
    primary = DirStore(tmp_path / "primary")
    replica = ZipStore(tmp_path / "replica.zip")
    monkeypatch.setattr(primary, "create_local_state_staging", lambda: pytest.fail("unsupported target reached capture"))
    repo = Repo([primary, replica], save_routing=SaveRouting(
        ((Selector(DeferredPayload), primary), (Selector(DeferredPayload), replica)), match_mode="all"))
    try:
        with pytest.raises(StoreCapabilityError):
            repo.save_object(DeferredPayload())
        assert not tuple(Path(primary.base_dir, "membership", "saved").glob("*/*.record"))
        assert not Path(replica.archive_path).exists()
    finally:
        replica.close()


def test_metadata_reads_no_payload_and_cannot_authorize_later_use(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store", query_index="none")
    state = Repo(store).save_object(DeferredPayload())
    Path(store.state_data_path(state.state.key), "payload.bin").write_bytes(b"corrupt")
    original = builtins.open

    def forbid_data(path, *args, **kwargs):
        assert Path(path).parent.name != "data", "metadata inspection read payload"
        return original(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "open", forbid_data)
        digest = NodePreparation((state,)).digest(state, domain="dryml-saved-root/v1")
        assert store.read_node(digest, kind="saved").graph_equal(state)
        assert store.inspect_states(state.state.key.cdef_digest)[0].available
    for defer in (False, True):
        with pytest.raises(RepoLoadError):
            build_exact_state_load_plan(Repo(store), state, _defer_payload=defer)


def test_current_reader_override_preserves_its_error(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store", query_index="none")
    state = Repo(store).save_object(DeferredPayload())
    error = StoreCapabilityError("custom current reader")
    @contextmanager
    def reject(*args, **kwargs):
        raise error
        yield
    monkeypatch.setattr(store, "read_states", reject)
    with pytest.raises(RepoLoadError) as caught:
        Repo(store).load_state_ref(state, reuse_live="never")
    assert caught.value.__cause__ is error


@pytest.mark.parametrize("version", (2, 3))
def test_old_snapshot_manifests_are_rejected_intact_before_hooks(tmp_path, version):
    store = DirStore(tmp_path / "store", query_index="none")
    state = Repo(store).save_object(DeferredPayload())
    current = _manifest(store, state)
    manifest_path = Path(store.state_data_path(state.state.key)).parent / "manifest.record"
    old = LocalStateManifest("pkl", "0" * 64, "1" * 64, "2" * 64, current.files, version=version).to_bytes()
    manifest_path.write_bytes(old)
    DeferredPayload.reads = 0
    before = Path(store.base_dir, "authority.record").read_bytes()
    with pytest.raises(RepoLoadError):
        build_exact_state_load_plan(Repo(store), state, _defer_payload=True)
    assert DeferredPayload.reads == 0
    assert manifest_path.read_bytes() == old
    assert Path(store.base_dir, "authority.record").read_bytes() == before
