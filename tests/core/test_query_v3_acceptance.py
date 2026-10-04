"""Compact Query V3 acceptance coverage for U8's cross-layer contract.

The named cases form an acceptance matrix: AE1/4 cover authority and stored
membership, AE6-8 metadata restrictions, AE10-12 traversal, AE17-20 algebra
and terminals, AE21 recovery, and AE22-24 effects, diagnostics, and imports.
Focused U1-U6 modules retain the larger boundary matrices.
"""

from pathlib import Path
import sys

import pytest

from dryml.core import ConcreteDefinition, Definition, Object, Repo, SKIP_ARGS, SaveAnnotations, Serializable
from dryml.core.bound_args import BoundArguments
from dryml.core.cdef_graph import EdgeKind
from dryml.core.links import DefLink
from dryml.core.query import EdgePolicy, IdentitySet, RelationshipKind, field
from dryml.core.query.sqlite import sqlite_available
from dryml.core.store.dir import DirStore
from dryml.core.symbol import ImportRef


class AcceptanceLeaf(Serializable):
    """Stateful saved leaf whose payload must remain unopened by queries."""

    def __init__(self, name):
        self.name = name

    def save_state_to_dir_imp(self, directory, *, codec):
        Path(directory, "name").write_text(self.name, encoding="ascii")

    def restore_state_from_dir_imp(self, directory, *, codec):
        self.name = Path(directory, "name").read_text(encoding="ascii")


class AcceptanceRoot(Object):
    """Saved root retaining a selected reference edge to an exact child state."""

    def __init__(self, child):
        self.child = child


def test_ae1_ae8_ae22_authority_metadata_and_query_effects(tmp_path, monkeypatch):
    """V3 finds stored identities and metadata without construction or payload I/O."""

    store = DirStore(tmp_path / "store", query_index="memory")
    writer = Repo(store)
    state = writer.save_object(
        AcceptanceLeaf("selected", repo=writer),
        annotations=SaveAnnotations(object={"team": "vision"}),
    )
    reopened = Repo(DirStore(store.base_dir, query_index="memory"))
    monkeypatch.setattr(
        reopened.default_store,
        "_validate_local_state_dir",
        lambda *_args, **_kwargs: pytest.fail("query opened a saved payload"),
    )

    selected = (
        reopened.query()
        .sel(Definition(AcceptanceLeaf, "selected"))
        .where(field("object", "team").eq("vision"))
        .state_refs()
        .stored()
        .collect()
    )

    assert selected.one() == state
    assert reopened._num_constructions == 0
    assert selected.sources(state)
    assert "source" in repr(selected.diagnostic())


def test_ae10_ae20_traversal_algebra_fixed_sets_and_terminals(tmp_path):
    """Explicit roots preserve reference paths through bounded fixed-set algebra."""

    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    child = repo.save_object(AcceptanceLeaf("child", repo=repo))
    root = repo.save_object(
        AcceptanceRoot(DefLink.finalized(EdgeKind.REF, child), repo=repo)
    )
    root_query = repo.query().sel(root.definition).cdefs().stored()
    occurrences = root_query.nested(child, edges=EdgePolicy.ALL).through(
        RelationshipKind.REFERENCE
    )
    fixed = IdentitySet((child.object,), bounded=True)
    combined = occurrences.targets().union(fixed).take(1)

    assert occurrences.one().target == child
    assert occurrences.owners().cdefs().one() == root.definition
    assert combined.bounded and combined.requested_limit == 1
    assert combined.count() == 1


@pytest.mark.skipif(not sqlite_available(), reason="sqlite3 is unavailable")
def test_ae21_ae23_save_load_and_missing_sidecar_recover_from_authority(tmp_path):
    """A removed derived sidecar cannot change V3 answers or exact reload identity."""

    store = DirStore(tmp_path / "store", query_index="sqlite")
    repo = Repo(store)
    state = repo.save_object(AcceptanceLeaf("saved", repo=repo))
    index = store.open_query_index()
    index.rebuild()
    before = store.read_state_ref_record(state.digest()).to_bytes()
    index.path.unlink()

    reopened = Repo(DirStore(store.base_dir, query_index="sqlite"))
    assert reopened.query().sel(state).state_refs().stored().one() == state
    assert reopened.load_state_ref(state, reuse_live="never").name == "saved"
    assert store.read_state_ref_record(state.digest()).to_bytes() == before


def test_ae24_trusted_selector_import_is_explicit_and_backend_free(tmp_path, monkeypatch):
    """Ordinary subclass selection may import trusted code without probing a backend."""

    module = "query_v3_trusted_fixture"
    (tmp_path / f"{module}.py").write_text(
        "class Parent:\n    pass\n\nclass Child(Parent):\n    pass\n",
        encoding="ascii",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop(module, None)
    selector = Definition(ImportRef(module, "Parent"), SKIP_ARGS)
    candidate = ConcreteDefinition._from_bound_record(
        ImportRef(module, "Child"), BoundArguments(()),
    )
    frameworks_before = {
        name for name in ("jax", "tensorflow", "torch") if name in sys.modules
    }

    assert IdentitySet((candidate,)).query().sel(selector).one() == candidate
    assert module in sys.modules
    assert {
        name for name in ("jax", "tensorflow", "torch") if name in sys.modules
    } == frameworks_before
