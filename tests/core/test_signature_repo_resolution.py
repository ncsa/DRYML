"""Focused authority-selection proofs for the shared core signature interpreter."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import pytest

from dryml.core import ConcreteDefinition, Definition, Object, Repo, Serializable
from dryml.core.cdef_graph import EdgeKind
from dryml.core.links import DefLink
from dryml.core.repo import RepoLoadError
from dryml.core.quoted import QuotedDef, SelectorSpec
from dryml.core.reference_values import ObjectRef, StateRef
from dryml.core.selector import Selector
from dryml.core.signatures import (
    AutoRef,
    Mat,
    Ref,
    ReferenceSelection,
    SignatureError,
    compile_signature,
    signature_context,
)
from dryml.core.store.dir import DirStore
from dryml.core.store.records import ClaimRecord, DeclarationRecord, DefinitionRecord, StateRefRecord
from dryml.core.store.store import StoreCapabilityError


class SignatureCounter(Serializable):
    """Minimal stateful fixture whose state receipt changes with ``value``."""

    def __init__(self, value=0):
        self.value = value

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        Path(dest_dir, "value").write_text(str(self.value), encoding="ascii")


class SignatureStateless(Object):
    """Minimal stateless Object fixture for automatic authority selection."""

    def __init__(self, value=0):
        self.value = value


class SignatureWrapper(Object):
    """Stateless root whose materializing child controls automatic selection."""

    def __init__(self, child):
        self.child = child


def _plan(annotation):
    """Compile one single-slot signature with the supplied role annotation."""

    def target(value):
        return value

    target.__annotations__["value"] = annotation
    return compile_signature(target)


def _stateful_reference(repo: Repo):
    """Declare, build, and save one exact Counter reference for selection tests."""

    reference = repo.declare_object(SignatureCounter(1).definition)
    live = repo.build_object_ref(reference)
    state = repo.save_object(live)
    return reference, live, state


def _replicate_declaration(store: DirStore, reference: ObjectRef, claim: ClaimRecord) -> None:
    """Install equal declaration authority in a separate Store without building."""

    store.write_definition_record(DefinitionRecord(reference.definition), stored_root=False)
    store.write_claim_record(claim)
    store.write_declaration_record(DeclarationRecord(reference))


def test_automatic_and_exact_live_projections_do_not_query_snapshots(tmp_path):
    """Automatic Ref uses local graph/receipt facts while exact live views stay local."""

    repo = Repo(DirStore(tmp_path / "store"))
    stateless = SignatureStateless()
    stateful = SignatureCounter(1)

    assert _plan(Ref[AutoRef]).prepare_args((stateless,), {}).authority["value"] == stateless.definition
    assert _plan(Ref[AutoRef]).prepare_args((stateful,), {}).authority["value"] == stateful.object_ref
    assert _plan(Ref[ConcreteDefinition]).prepare_args((stateful,), {}).authority["value"] == stateful.definition
    assert _plan(Ref[ObjectRef]).prepare_args((stateful,), {}).authority["value"] == stateful.object_ref
    with pytest.raises(SignatureError, match="unavailable"):
        _plan(Ref[StateRef]).prepare_args((stateless,), {})

    reference, live, state = _stateful_reference(repo)
    live.value = 99
    assert _plan(Ref[AutoRef]).prepare_args((live,), {}).authority["value"] == state
    assert _plan(Ref[StateRef]).prepare_args((live,), {}).authority["value"] == state
    assert reference == live.object_ref


def test_automatic_selection_counts_stateful_descendants_but_not_ref_data(tmp_path):
    """Automatic selection follows materializing topology without loading payloads."""

    repo = Repo(DirStore(tmp_path / "store"))
    stateful = SignatureCounter(2, repo=repo)
    materializing = SignatureWrapper(stateful, repo=repo)
    reference_only = SignatureWrapper(
        DefLink.finalized(EdgeKind.REF, stateful.definition), repo=repo
    )

    assert _plan(Ref[AutoRef]).prepare_args((materializing,), {}).authority["value"] == materializing.object_ref
    assert _plan(Ref[AutoRef]).prepare_args((reference_only,), {}).authority["value"] == reference_only.definition


def test_state_selection_deduplicates_replicas_and_rejects_ambiguity(tmp_path):
    """Exact ObjectRef snapshot lookup accepts replicas but never chooses a snapshot."""

    first = DirStore(tmp_path / "first")
    second = DirStore(tmp_path / "second")
    repo = Repo(first)
    reference, live, state = _stateful_reference(repo)
    _replicate_declaration(second, reference, first.read_claim_record(reference.digest()))
    second.write_state_ref_record(StateRefRecord(state))
    replicas = Repo((first, second))

    assert _plan(Ref[StateRef]).prepare_args((reference,), {}, repo=replicas).authority["value"] == state
    with signature_context(repo=replicas):
        assert _plan(Ref[StateRef]).prepare_args((reference,), {}).authority["value"] == state
    second_state = StateRef(
        reference, {path: "pkl-" + "b" * 64 for path in reference.objects}
    )
    second.write_state_ref_record(StateRefRecord(second_state))
    with pytest.raises(SignatureError, match="ambiguous"):
        _plan(Ref[ObjectRef | StateRef]).prepare_args((reference,), {}, repo=replicas)
    with pytest.raises(SignatureError, match="ambiguous"):
        _plan(Ref[StateRef | ObjectRef]).prepare_args((reference,), {}, repo=replicas)


@pytest.mark.parametrize("status", ("available", "claimed", "completed"))
def test_declaration_claim_statuses_remain_reference_eligible(tmp_path, status):
    """Ref identity selection validates every ClaimRecord status without acquiring it."""

    store = DirStore(tmp_path / "store")
    repo = Repo(store, clock=lambda: 10.0)
    reference = repo.declare_object(SignatureCounter(1).definition)
    old = store.read_claim_record(reference.digest())
    if status == "available":
        claim = ClaimRecord(reference.digest(), old.generation, status)
    elif status == "claimed":
        claim = ClaimRecord(reference.digest(), old.generation + 1, status, "other", 20.0)
    else:
        claim = ClaimRecord(reference.digest(), old.generation + 1, status, state_ref_digest="a" * 64)
    store.write_claim_record(claim)

    result = _plan(Ref[ObjectRef]).prepare_args((reference.definition,), {}, repo=repo)
    assert result.authority["value"] == reference
    assert store.read_claim_record(reference.digest()) == claim


def test_retained_live_claim_remains_reference_eligible_without_acquisition(tmp_path):
    """A real live construction lease is selection evidence, never Ref permission to build."""

    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    reference = repo.declare_object(SignatureCounter(1).definition)
    live = repo.build_object_ref(reference)

    selected = _plan(Ref[ObjectRef]).prepare_args((reference.definition,), {}, repo=repo)

    assert selected.authority["value"] == reference
    assert store.read_claim_record(reference.digest()).status == "claimed"
    assert live._claim_lease is not None


def test_explicit_selection_and_definition_strengthening_guards(tmp_path):
    """Slot selections disambiguate only declared topology and Definition does not add defaults."""

    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    cdef = SignatureCounter(1).definition
    first = repo.declare_object(cdef)
    second = repo.declare_object(cdef)
    plan = _plan(Ref[ObjectRef])
    with pytest.raises(SignatureError, match="ambiguous"):
        plan.prepare_args((cdef,), {}, repo=repo)
    selected = plan.prepare_args(
        (cdef,), {}, repo=repo, selections={"value": ReferenceSelection(second, store)}
    )
    assert selected.authority["value"] == second
    unrelated = repo.declare_object(SignatureCounter(2).definition)
    with pytest.raises(SignatureError, match="topology"):
        plan.prepare_args((cdef,), {}, repo=repo, selections={"value": unrelated})

    assert _plan(Ref[ConcreteDefinition]).prepare_args((Definition(SignatureCounter, 1),), {}).authority["value"] == cdef
    with pytest.raises(SignatureError, match="unavailable"):
        _plan(Ref[ConcreteDefinition]).prepare_args((Definition(SignatureCounter),), {})


def test_source_guards_and_quotation_data_are_preserved(tmp_path):
    """Live receipt and quotation restrictions survive composed authority conversion."""

    repo = Repo(DirStore(tmp_path / "store"))
    reference, live, state = _stateful_reference(repo)
    no_receipt = repo.build_object_ref(repo.declare_object(SignatureCounter(2).definition))
    with pytest.raises(SignatureError, match="unavailable"):
        _plan(Ref[StateRef]).prepare_args((no_receipt,), {}, repo=repo)
    assert _plan(Ref[StateRef]).prepare_args((reference,), {}, repo=repo).authority["value"] == state

    quoted = QuotedDef(Definition(SignatureCounter, 1))
    assert _plan(Ref[Definition]).prepare_args((quoted,), {}).authority["value"] == quoted.value
    assert isinstance(_plan(Ref[QuotedDef]).prepare_args((reference,), {}).authority["value"], QuotedDef)
    selector_spec = SelectorSpec(Selector(Definition(SignatureCounter)))
    assert _plan(Ref[Selector]).prepare_args((selector_spec,), {}).authority["value"] == selector_spec.selector
    with pytest.raises(SignatureError, match="quoted input"):
        _plan(Mat[ConcreteDefinition]).prepare_args((quoted,), {})
    assert _plan(Ref[ConcreteDefinition | ObjectRef | StateRef]).prepare_args(
        (reference,), {}, repo=repo
    ).authority["value"] == state
    assert live.object_ref == reference


def test_state_only_and_store_failures_are_not_treated_as_absence(tmp_path, monkeypatch):
    """State-only records serve explicit refs, while corrupt/missing reads remain failures."""

    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    reference, _, state = _stateful_reference(repo)
    state_only = DirStore(tmp_path / "state-only")
    state_only.write_state_ref_record(StateRefRecord(state))
    assert _plan(Ref[StateRef]).prepare_args((reference,), {}, repo=Repo(state_only)).authority["value"] == state

    read_claim_record = store.read_claim_record
    monkeypatch.setattr(store, "read_claim_record", lambda digest: None)
    with pytest.raises(RepoLoadError, match="lacks authoritative ClaimRecord"):
        repo.reference_evidence(reference.definition)
    monkeypatch.setattr(store, "read_claim_record", read_claim_record)
    monkeypatch.setattr(store, "iter_state_ref_records", lambda: (_ for _ in ()).throw(OSError("offline")))
    with pytest.raises(RepoLoadError, match="could not be read"):
        repo.reference_evidence(reference.definition)


def test_evidence_fences_duplicate_handles_once_and_reject_unsupported(tmp_path):
    """One ordered evidence cut fences duplicate Store handles once and fails closed otherwise."""

    class CountingStore(DirStore):
        fences = 0

        @contextmanager
        def authority_read_fence(self):
            type(self).fences += 1
            with super().authority_read_fence():
                yield

    store = CountingStore(tmp_path / "store")
    repo = Repo((store, store))
    reference = repo.declare_object(SignatureCounter(1).definition)
    evidence = repo.reference_evidence(reference.definition)
    assert len(evidence.declarations) == 1
    assert CountingStore.fences == 1

    class UnsupportedStore(CountingStore):
        def authority_fence_key(self):
            raise StoreCapabilityError("no stable snapshot")

    with pytest.raises(StoreCapabilityError, match="stable snapshot"):
        Repo(UnsupportedStore(tmp_path / "unsupported")).reference_evidence(reference.definition)


def test_immutable_snapshot_store_uses_its_read_fence_not_a_writer_fence(tmp_path):
    """Read-only snapshot authority can select/restore without mutable fencing."""

    source = DirStore(tmp_path / "store")
    writable = Repo(source)
    reference, _, state = _stateful_reference(writable)

    class ImmutableSnapshotStore(DirStore):
        def authority_fence_key(self):
            return f"immutable:{self.base_dir}"

        @contextmanager
        def authority_read_fence(self):
            yield

        def writer_lock(self):
            raise AssertionError("immutable snapshot selection must not take a writer fence")

    readonly = Repo(ImmutableSnapshotStore.open_existing(tmp_path / "store"))

    assert _plan(Ref[StateRef]).prepare_args((reference,), {}, repo=readonly).authority["value"] == state
