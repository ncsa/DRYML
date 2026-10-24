"""Managed completion refuses partial current-Store saved-root publication."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.usefixtures("fixed_managed_snapshot_environment")

from dryml.core import Repo, SaveRouting, Selector
from dryml.core.object import Pickleable
from dryml.core.repo import RepoSaveError
from dryml.core.node import require_saved
from dryml.core.node_prepare import NodePreparation
from dryml.core.store.dir import DirStore
from dryml.managed import ManagedConfig, managed_operation
from dryml.managed import runtime


class ManagedReportValue(Pickleable):
    """Managed receiver whose final save exercises replica report completion."""

    def __init__(self, value=0):
        self.value = value

    @managed_operation()
    def increment(self, *, managed):
        """Mutate the receiver before the managed final-publication boundary."""

        self.value += 1
        return self.value


class ManagedReportGraph(Pickleable):
    """Own shared children for multi-node checkpoint and grant-confirmation proofs."""

    def __init__(self, children):
        """Retain the supplied child sequence without changing its aliases."""
        self.children = children

    @managed_operation()
    def checkpoint_children(self, *, managed):
        """Checkpoint the full owned graph and select it as the final state."""
        selected = managed.checkpoint()
        managed.final_state(selected)
        return len(self.children)


def test_multinode_grant_confirmation_walks_once_and_public_retention_survives(tmp_path, monkeypatch):
    """Confirm each exact root once while retaining every shared owned state key."""
    from dryml.core import node
    from dryml.core.utils.graph.path import Index, Parameter

    store = DirStore(tmp_path / "state", query_index="none")
    repo = Repo(store)
    first = ManagedReportValue(1, repo=repo)
    second = ManagedReportValue(2, repo=repo)
    graph = ManagedReportGraph([first, second, first], repo=repo)
    original_confirm = runtime._confirm_root
    original_walk = node._owned_nodes
    confirmations = []

    def confirm(ledger, selected, **kwargs):
        expected = {state.key for _, state in selected.state_spec().items()}
        assert len(expected) == 3
        walks = []

        def counted(scope):
            walks.append(scope)
            return original_walk(scope)

        with monkeypatch.context() as patch:
            patch.setattr(node, "_owned_nodes", counted)
            original_confirm(ledger, selected, **kwargs)
        assert walks == [selected]
        confirmations.append(expected)

    monkeypatch.setattr(runtime, "_confirm_root", confirm)
    result = graph.checkpoint_children(managed=ManagedConfig(state_repo=repo, control_store=store))
    assert result.value == 3 and confirmations
    saved = result.final_state_ref
    keys = {state.key for state in saved.state_spec().values()}
    assert confirmations == [keys] * len(confirmations)
    assert saved.at((Parameter("children"), Index(0))) is saved.at((Parameter("children"), Index(2)))
    retained = repo.retain(saved)
    assert {item.key for item in repo.release(retained)} == keys
    assert all(store.delete_state(key).status == "retained" for key in keys)
    loaded = repo.load_state_ref(saved, reuse_live="never")
    assert [child.value for child in loaded.children] == [1, 2, 1]
    assert loaded.children[0] is loaded.children[2]


def test_managed_completion_rejects_a_partial_snapshot_report(tmp_path, monkeypatch):
    """A completed replica cannot make managed final association succeed alone."""

    first = DirStore(tmp_path / "first")
    second = DirStore(tmp_path / "second")
    repo = Repo(
        (first, second),
        save_routing=SaveRouting(
            ((Selector(ManagedReportValue), first), (Selector(ManagedReportValue), second)),
            match_mode="all",
        ),
    )
    value = ManagedReportValue(repo=repo)

    def no_association(*args, **kwargs):
        raise AssertionError("partial publication must not associate a final state")

    monkeypatch.setattr(runtime, "_associate_state", no_association)
    monkeypatch.setattr(
        second,
        "publish_save_batch",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("second replica failed")),
    )

    with pytest.raises(RepoSaveError) as raised:
        value.increment(managed=ManagedConfig(state_repo=repo, control_store=first))

    report = raised.value.report
    assert report is not None
    publications = [item for item in report.publications if item.phase == "snapshot"]
    assert [(item.store, item.status) for item in publications] == [
        (first, "completed"), (second, "failed"),
    ]
    published, failed = publications
    assert require_saved(published.state_ref) is published.state_ref
    assert failed.state_ref.graph_equal(published.state_ref)
    digest = NodePreparation((published.state_ref,)).digest(published.state_ref, domain="dryml-saved-root/v1")
    assert first.read_node(digest, kind="saved").graph_equal(published.state_ref)
    assert second.read_node(digest, kind="saved") is None
    assert Repo(first).load_state_ref(published.state_ref, reuse_live="never").value == 1
    status = value.increment.status(state_repo=repo, control_store=first)
    assert (status.state, status.checkpoint, status.final) == ("failed", None, None)
    assert raised.value.failed_result.status == status
