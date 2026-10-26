"""Journal budgets cover supported public results and interrupted byte transitions."""

from pathlib import Path
from dataclasses import replace
from uuid import uuid4

import pytest

from dryml.core import Definition, Repo
from dryml.core.object import Pickleable
from dryml.core.store.dir import DirStore
from dryml.core.store import authority as journal
from dryml.core.store.records import (MutationIntentRecord, StoreRecordError,
                                      _json_bytes, _encode_authority_bytes as encode,
                                      _decode_authority_bytes as decode)
from dryml.core.store.store import StoreAuthorityError
from dryml.managed import ManagedConfig, managed_operation
from dryml.managed.control import ManagedControlStore
from dryml.managed.result_codec import MAX_RESULT_BYTES, encode_result
from dryml.managed.results import publish_result, read_result_bytes, result_path, ResultReference


class LargeResultValue(Pickleable):
    """Exercise real managed completion, final state capture and result storage."""

    def __init__(self, value=0):
        self.value = value

    @managed_operation(resumable=True)
    def advance(self, *, managed):
        self.value += 1
        return "x" * (7 * 1024 * 1024)


@pytest.mark.parametrize("query_index", ["none", "sqlite"])
def test_public_managed_large_result_save_reopen_and_next_mutation(tmp_path, query_index):
    root = tmp_path / "store"
    store = DirStore(root, query_index=query_index)
    repo = Repo(store)
    live = LargeResultValue(repo=repo)
    result = live.advance(managed=ManagedConfig(state_repo=repo))
    assert result.status.state == "completed"
    assert result.value == "x" * (7 * 1024 * 1024)
    assert result.final_state_ref.state is not None
    assert not store.authority_cut().pending
    receipts = [MutationIntentRecord.from_bytes(p.read_bytes())
                for p in Path(store.base_dir, "mutations").glob("*.record")]
    receipt, = [r for r in receipts if any(s.startswith("managed-result:") for s in r.intent["scopes"])]
    assert receipt.intent["status"] == "acknowledged"
    assert len(receipt.to_bytes()) > 16 * 1024 * 1024
    repo.close(flush=False)
    reopened = DirStore(root, query_index=query_index)
    assert reopened.recover_authority() == ()
    assert result.value == "x" * (7 * 1024 * 1024)
    cut = reopened.authority_cut()
    reopened.install_creation(Definition(LargeResultValue, value=2).concretize())
    assert reopened.authority_cut().epoch == cut.epoch + 1
    assert not reopened.authority_cut().pending


@pytest.mark.parametrize("query_index", ["none", "sqlite"])
def test_old_pending_large_result_is_classified_and_repaired_losslessly(tmp_path, query_index):
    import hashlib
    store = DirStore(tmp_path / "store", query_index=query_index)
    ManagedControlStore(store, Repo(store)).initialize()
    if query_index == "sqlite":
        store.open_query_index().maintain_store()
    payload = encode_result("x" * (7 * 1024 * 1024))
    reference = ResultReference(hashlib.sha256(payload).hexdigest())
    relative = store._relative(result_path(store, reference.key))
    cut, identity = store.authority_cut(), str(uuid4())
    # Recreate the old writer's admitted pending frame and installed result,
    # without routing it through the new preparation preflight.
    record = MutationIntentRecord(_json_bytes({
        "mutation_id": identity, "store_uuid": cut.store_uuid, "base_epoch": cut.epoch,
        "scopes": ["managed-result:" + reference.key],
        "transitions": [journal.PreparedChange(relative, None, payload).to_data()],
        "confirmed": [], "projection": [], "epoch": None, "status": "pending",
        "persistence_confirmed": False}))
    assert len(record.to_bytes()) < 16 * 1024 * 1024
    store._atomic_write(result_path(store, reference.key), payload)
    store._atomic_write(journal._path(store, identity), record.to_bytes())
    store._atomic_write(Path(store.base_dir, "authority.record"),
                        replace(cut, pending=(identity,)).to_bytes())
    reopened = DirStore(store.base_dir, query_index=query_index)
    recovered, = reopened.recover_authority()
    assert recovered.intent["status"] == "classified"
    assert recovered.intent["confirmed"] == [relative]
    assert decode(recovered.intent["projection"][0]["value"]) == payload
    assert reopened.authority_cut().epoch == cut.epoch + 1
    assert reopened.recover_authority() == (recovered,)
    reopened.replay_mutation(identity)
    assert not reopened.authority_cut().pending
    assert read_result_bytes(reopened, reference) == payload
    reopened.install_creation(Definition(LargeResultValue, value=4).concretize())


@pytest.mark.parametrize("query_index", ["none", "sqlite"])
def test_near_eight_mib_result_budget_is_still_supported(tmp_path, query_index):
    store = DirStore(tmp_path / "store", query_index=query_index)
    ManagedControlStore(store, Repo(store)).initialize()
    payload = encode_result("x" * (MAX_RESULT_BYTES - 1024))
    assert MAX_RESULT_BYTES - 1024 < len(payload) <= MAX_RESULT_BYTES
    reference = publish_result(store, payload)
    assert read_result_bytes(store, reference) == payload
    assert not store.authority_cut().pending


def test_old_pending_mutable_current_and_history_keep_initial_projection(tmp_path):
    store = DirStore(tmp_path / "store", query_index="none")
    old, new = b"o" * (4 * 1024 * 1024), b"n" * (1024 * 1024)
    paths = ("managed/budget/current.json", "managed/budget/history/1.json")
    changes = [journal.PreparedChange(path, old, new) for path in paths]
    cut, identity = store.authority_cut(), str(uuid4())
    from types import SimpleNamespace
    record = journal._intent(SimpleNamespace(cut=cut, changes=changes, scopes=("budget-test",)), identity)
    assert len(record.to_bytes()) < 16 * 1024 * 1024
    store._atomic_write(Path(store.base_dir, paths[0]), new)
    store._atomic_write(Path(store.base_dir, paths[1]), old)
    store._atomic_write(journal._path(store, identity), record.to_bytes())
    store._atomic_write(Path(store.base_dir, "authority.record"),
                        replace(cut, pending=(identity,)).to_bytes())
    reopened = DirStore(store.base_dir, query_index="none")
    recovered, = reopened.recover_authority()
    assert recovered.intent["confirmed"] == [paths[0]]
    assert {row["path"]: decode(row["value"]) for row in recovered.intent["projection"]} == {
        paths[0]: new, paths[1]: old}
    assert len(recovered.to_bytes()) > 16 * 1024 * 1024
    assert reopened.authority_cut().epoch == cut.epoch + 1
    reopened.replay_mutation(identity)
    assert not reopened.authority_cut().pending
    assert all(Path(store.base_dir, path).read_bytes() == new for path in paths)


@pytest.mark.parametrize("query_index", ["none", "sqlite"])
def test_large_mutable_current_and_history_budget_refuses_before_effects(tmp_path, query_index, monkeypatch):
    store = DirStore(tmp_path / "store", query_index=query_index)
    ManagedControlStore(store, Repo(store)).initialize()
    if query_index == "sqlite":
        store.open_query_index().maintain_store()
    before, after = b"a" * (8 * 1024 * 1024), b"b" * (8 * 1024 * 1024)
    paths = ("managed/budget/current.json", "managed/budget/history/1.json")
    for path in paths:
        store._atomic_write(Path(store.base_dir, path), before)
    cut = store.authority_cut()
    original_files = {p.relative_to(store.base_dir): p.read_bytes()
                      for p in Path(store.base_dir).rglob("*") if p.is_file()}
    changes = tuple(journal.PreparedChange(path, before, after) for path in paths)
    # Pending fits the journal limit; adding the duplicated observed values does
    # not. Even equal before/after values must reserve their projection copy.
    def prepare():
        for path in paths:
            assert journal.raw(store, path) == before
        return changes, ("budget-test",), None
    def forbidden(*args, **kwargs):
        pytest.fail("over-budget preparation reached a writer/dirty effect")
    monkeypatch.setattr(store, "_atomic_write", forbidden)
    if store.open_query_index() is not None:
        monkeypatch.setattr(store.open_query_index(), "_mark_dirty", forbidden)
    with pytest.raises(StoreRecordError, match="Oversized.*mutation-intent"):
        store._prepare_and_apply(prepare)
    assert store.authority_cut() == cut
    assert {p.relative_to(store.base_dir): p.read_bytes()
            for p in Path(store.base_dir).rglob("*") if p.is_file()} == original_files


def _facts(store, changes):
    return journal.PreparedMutation(store.authority_cut(), (), (), tuple(changes),
                                    ("budget-test",), store._prepared_fact_owner)


def test_preflight_reserves_largest_intermediate_and_acknowledged_frame(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store", query_index="none")
    changes = [journal.PreparedChange("budget/current", b"old", b"x" * 2000),
               journal.PreparedChange("budget/current", b"x" * 2000, b"small")]
    facts = _facts(store, changes)
    pending = journal._intent(facts, str(uuid4()))
    data = pending.intent
    data.update(confirmed=["budget/current"], epoch=data["base_epoch"] + 1,
                projection=[{"path": "budget/current", "kind": "file", "value": encode(b"x" * 2000)}],
                status="classified")
    classified = MutationIntentRecord(_json_bytes(data)).to_bytes()
    data["status"] = "acknowledged"
    data["persistence_confirmed"] = True
    acknowledged = MutationIntentRecord(_json_bytes(data)).to_bytes()
    assert len(pending.to_bytes()) < len(classified) < len(acknowledged)
    monkeypatch.setattr(MutationIntentRecord, "max_bytes", len(classified) - 1)
    with pytest.raises(StoreRecordError, match="Oversized"):
        _facts(store, changes)
    monkeypatch.setattr(MutationIntentRecord, "max_bytes", len(acknowledged) - 1)
    with pytest.raises(StoreRecordError, match="Oversized"):
        _facts(store, changes)
    monkeypatch.setattr(MutationIntentRecord, "max_bytes", len(acknowledged))
    assert _facts(store, changes).changes == tuple(changes)


@pytest.mark.parametrize("boundary", ["before-intent", "after-result", "classification", "acknowledgement"])
def test_large_result_io_failure_retains_exact_cut_and_repair(tmp_path, monkeypatch, boundary):
    store = DirStore(tmp_path / "store", query_index="none")
    ManagedControlStore(store, Repo(store)).initialize()
    cut = store.authority_cut()
    payload = encode_result("x" * (7 * 1024 * 1024))
    original = store._atomic_write
    fired = []
    def fail(path, data):
        relative = store._relative(path)
        if boundary == "before-intent" and relative == "mutation-prepare.record":
            fired.append(True)
            raise OSError("injected budget IO")
        if relative.startswith("mutations/") and boundary in {"classification", "acknowledgement"}:
            status = MutationIntentRecord.from_bytes(data).intent["status"]
            if status == {"classification": "classified", "acknowledgement": "acknowledged"}[boundary]:
                fired.append(True)
                raise OSError("injected budget IO")
        original(path, data)
        if boundary == "after-result" and relative.startswith("managed/results/"):
            fired.append(True)
            raise OSError("injected budget IO")
    with monkeypatch.context() as patch:
        patch.setattr(store, "_atomic_write", fail)
        with pytest.raises(journal.StoreMutationError, match="injected budget IO") as raised:
            publish_result(store, payload)
    assert fired
    error = raised.value
    assert error.cut == store.authority_cut()
    if boundary == "before-intent":
        assert store.authority_cut() == cut
        assert not Path(store.base_dir, "managed", "results").exists()
        assert not Path(store.base_dir, "mutation-prepare.record").exists()
        assert not list(Path(store.base_dir, "mutations").glob("*.record"))
    else:
        assert error.mutation_id in store.authority_cut().pending
        record = store._read_file(journal._path(store, error.mutation_id), MutationIntentRecord)
        assert record.intent["status"] != "acknowledged"
        reopened = DirStore(store.base_dir, query_index="none")
        recovered, = reopened.recover_authority()
        assert recovered.intent["epoch"] == cut.epoch + 1
        assert recovered.intent["confirmed"]
        row, = recovered.intent["projection"]
        assert decode(row["value"]) == payload
        assert Path(reopened.base_dir, row["path"]).read_bytes() == payload
        reopened.replay_mutation(error.mutation_id)
        assert not reopened.authority_cut().pending
    assert publish_result(store, payload)


def test_journal_encoder_decoder_and_witness_share_bound_and_closed_grammar(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store", query_index="none")
    record = journal._intent(_facts(store, [journal.PreparedChange("budget/file", None, b"x" * 2000)]), str(uuid4()))
    payload = record.to_bytes()
    path = journal._path(store, record.intent["mutation_id"])
    store._atomic_write(path, payload)
    monkeypatch.setattr(MutationIntentRecord, "max_bytes", len(payload) - 1)
    with pytest.raises(StoreRecordError):
        record.to_bytes()
    with pytest.raises(StoreRecordError):
        MutationIntentRecord.from_bytes(payload)
    with pytest.raises(StoreAuthorityError):
        store._read_file(path, MutationIntentRecord)
    with pytest.raises(StoreAuthorityError, match="Oversized"):
        journal.raw(store, store._relative(path))
    monkeypatch.setattr(MutationIntentRecord, "max_bytes", len(payload))
    assert store._read_file(path, MutationIntentRecord) == record
    assert journal.raw(store, store._relative(path)) == payload
    data = record.to_data()
    data["intent"]["extra"] = True
    with pytest.raises(StoreRecordError):
        MutationIntentRecord.from_data(data)
    with pytest.raises(StoreRecordError):
        MutationIntentRecord.from_bytes(payload + b"\n")
