"""Retry unpublished query results without replaying visible iterator prefixes."""

from contextlib import closing

import pytest

from dryml.core import Definition, Node, Object, ObjectId, Ref, Serializable
from dryml.core.query import QueryError
from dryml.core.query.source import _NodePage
from dryml.core.store.dir import DirStore


class CutItem(Object):
    """Stateless query root with a visible label for snapshot comparisons."""

    def __init__(self, label):
        self.label = label


class QuotedOwner(Object):
    """Reference-only recipe whose subject is opaque to scalar postings."""

    def __init__(self, subject: Ref[Node]):
        self.subject = subject


class SpecializedQuotedOwner(QuotedOwner):
    """Subclass admitted only by soft class matching, not an exact recipe."""


class QuotedPair(Object):
    """Independent quoted slots must keep their subject identities correlated."""

    def __init__(self, first: Ref[Node], second: Ref[Node]):
        self.first, self.second = first, second


class IdentifiedSubject(Serializable):
    """Quoted subject with an identity pin independent of recipe equality."""

    def __init__(self, label):
        self.label = label


def test_unquoted_recipe_lowering_does_not_encode_constructor_values(tmp_path, monkeypatch):
    import numpy as np
    from dryml.core import node_codec
    from dryml.core.query.sqlite.lowering import _store_discovery_relation

    with closing(DirStore(tmp_path / "store")) as store:
        recipe = Definition(CutItem, np.zeros(1024, dtype=np.float32)).concretize()
        query = store.query().saved().sel(recipe)
        monkeypatch.setattr(node_codec, "_describe_node", lambda *a: pytest.fail("unquoted lowering encoded metadata"))
        _store_discovery_relation(query)


def test_quoted_identity_prefilter_correlates_slots_and_keeps_exact_residual(tmp_path):
    from dryml.core.query.model import QueryIndexUnavailable

    with closing(DirStore(tmp_path / "store")) as store:
        first, second, other = [Definition(IdentifiedSubject, "same").concretize().with_identity(ObjectId())
                                for _ in range(3)]
        wanted = Definition(QuotedPair, Ref(first), Ref(second)).concretize()
        for pair in (wanted, Definition(QuotedPair, Ref(second), Ref(first)).concretize(),
                     Definition(QuotedPair, Ref(other), Ref(second)).concretize()):
            store.publish_node(pair, kind="saved")
        result = store.query().saved().sel(wanted).collect()
        assert result.one().graph_equal(wanted)
        assert result.diagnostic().cdef_blobs_decoded == 1

        different = Definition(IdentifiedSubject, "different").concretize().with_identity(first.object_id)
        store.publish_node(Definition(QuotedPair, Ref(different), Ref(second)).concretize(), kind="saved")
        result = store.query().saved().sel(wanted).collect()
        assert result.one().graph_equal(wanted)
        assert result.diagnostic().cdef_blobs_decoded == 2
        with pytest.raises(QueryIndexUnavailable):
            store.query().saved().sel(wanted).require_indexed().collect()


@pytest.mark.parametrize("shape", ("mapping", "tuple", "frozenset"))
def test_quoted_identity_prefilter_preserves_container_matching_and_opacity(tmp_path, shape):
    with closing(DirStore(tmp_path / "store")) as store:
        first, second = [Definition(IdentifiedSubject, "same").concretize().with_identity(ObjectId())
                         for _ in range(2)]

        def container(a, b):
            return ({"second": b.quote(), "first": a.quote()} if shape == "mapping" else
                    (a.quote(), b.quote()) if shape == "tuple" else frozenset((a.quote(), b.quote())))

        wanted = Definition(CutItem, container(first, second)).concretize()
        store.publish_node(wanted, kind="saved")
        pattern = Definition(CutItem, container(second, first) if shape == "frozenset"
                             else container(first, second)).concretize()
        assert store.query().saved().sel(pattern).one().graph_equal(wanted)
        assert store.query().object_refs().sel(pattern).one().graph_equal(wanted)
        assert store.query().object_refs().object_id(first.object_id).count() == 0


@pytest.mark.parametrize("domain", ("saved", "object_refs"))
def test_exact_recipe_class_prefilter_avoids_decoding_unrelated_saved_graphs(tmp_path, domain):
    with closing(DirStore(tmp_path / "store")) as store:
        for index in range(4):
            store.publish_node(Definition(CutItem, str(index)).concretize(), kind="saved")
        subject = Definition(CutItem, "subject").concretize()
        wanted = Definition(QuotedOwner, Ref(subject)).concretize()
        store.publish_node(wanted, kind="saved")
        store.publish_node(Definition(SpecializedQuotedOwner, Ref(subject)).concretize(), kind="saved")
        query = getattr(store.query(), domain)()
        result = query.sel(wanted).collect()
        assert result.one().graph_equal(wanted)
        assert result.diagnostic().cdef_blobs_decoded == 1

        soft = query.sel(Definition(QuotedOwner)).collect()
        assert len(soft) == 2
        from dryml.core.query.model import QueryIndexUnavailable
        with pytest.raises(QueryIndexUnavailable):
            query.sel(wanted).require_indexed().collect()


@pytest.mark.parametrize("terminal", ["collect", "count", "iter"])
def test_cut_drift_before_first_output_restarts_from_fresh_authority(tmp_path, monkeypatch, terminal):
    with closing(DirStore(tmp_path / "store")) as store, closing(DirStore.open_existing(tmp_path / "store")) as peer:
        store.publish_node(Definition(CutItem, "before").concretize())
        expand = _NodePage.expand
        moved = []

        def change_after_expansion(page, *args, **kwargs):
            result = expand(page, *args, **kwargs)
            if not moved:
                moved.append(True)
                peer.publish_node(Definition(CutItem, "after").concretize())
            return result

        monkeypatch.setattr(_NodePage, "expand", change_after_expansion)
        query = store.query().cdefs().stored()
        result = getattr(query, terminal)()
        if terminal == "count":
            assert result == 2
        else:
            assert {node.parameters["label"] for node in result} == {"before", "after"}
        assert moved == [True]


def test_started_iterator_refuses_cut_drift_without_replaying_delivered_results(tmp_path):
    with closing(DirStore(tmp_path / "store")) as store, closing(DirStore.open_existing(tmp_path / "store")) as peer:
        for label in ("one", "two", "three"):
            store.publish_node(Definition(CutItem, label).concretize())
        with closing(store.query().cdefs().stored().page_size(1).iter()) as items:
            first = next(items)
            assert first.parameters["label"] in {"one", "two", "three"}
            peer.publish_node(Definition(CutItem, "four").concretize())
            with pytest.raises(QueryError, match="changed|continuation"):
                next(items)
            assert list(items) == []


@pytest.mark.parametrize("budget", [3, 4])
def test_collect_discards_partial_rows_but_keeps_spent_admissions(tmp_path, monkeypatch, budget):
    from dryml.core.node_prepare import NodePreparation
    from dryml.core.query import QueryVerifyBudgetExceeded
    from dryml.core.query.query import IdentityQuery

    with closing(DirStore(tmp_path / "store")) as store, closing(DirStore.open_existing(tmp_path / "store")) as peer:
        for label in ("one", "two", "three"):
            store.publish_node(Definition(CutItem, label).concretize())
        original = IdentityQuery._iter_terminal_members
        removed = []

        def move_after_first(query, *args, **kwargs):
            with closing(original(query, *args, **kwargs)) as items:
                for item in items:
                    yield item
                    if not removed:
                        node = item[1][0]
                        removed.append(node.parameters["label"])
                        digest = NodePreparation((node,)).digest(node, domain="dryml-recipe-root/v1")
                        peer.withdraw_node(digest, kind="recipe")
                        peer.publish_node(Definition(CutItem, "four").concretize())

        monkeypatch.setattr(IdentityQuery, "_iter_terminal_members", move_after_first)
        query = store.query().cdefs().stored().page_size(1).max_verify(budget)
        if budget == 3:
            with pytest.raises(QueryVerifyBudgetExceeded):
                query.collect()
        else:
            result = query.collect()
            assert {node.parameters["label"] for node in result} == ({"one", "two", "three", "four"} - set(removed))
        assert len(removed) == 1


def test_index_replacement_before_output_uses_a_fresh_generation(tmp_path, monkeypatch):
    with closing(DirStore(tmp_path / "store")) as store:
        store.publish_node(Definition(CutItem, "retained").concretize())
        original = _NodePage.expand
        moved = []

        def replace_index(page, *args, **kwargs):
            result = original(page, *args, **kwargs)
            if not moved:
                moved.append(True)
                store.rebuild_query_index()
            return result

        monkeypatch.setattr(_NodePage, "expand", replace_index)
        assert store.query().cdefs().stored().collect().one().parameters["label"] == "retained"


def test_continuous_cut_drift_exhausts_without_returning_partial_results(tmp_path, monkeypatch):
    from dryml.core.store import authority

    with closing(DirStore(tmp_path / "store")) as store, closing(DirStore.open_existing(tmp_path / "store")) as peer:
        store.publish_node(Definition(CutItem, "initial").concretize())
        original = _NodePage.expand
        calls = []

        def move_every_attempt(page, *args, **kwargs):
            result = original(page, *args, **kwargs)
            calls.append(True)
            peer.publish_node(Definition(CutItem, str(len(calls))).concretize())
            return result

        monkeypatch.setattr(_NodePage, "expand", move_every_attempt)
        monkeypatch.setattr(authority, "_pause_cut_retry", lambda attempt: None)
        with pytest.raises(QueryError, match="changed repeatedly"):
            store.query().cdefs().stored().collect()
        assert len(calls) == 8


@pytest.mark.parametrize("failure", [QueryError, KeyboardInterrupt])
def test_non_cut_failures_are_not_retried(tmp_path, monkeypatch, failure):
    with closing(DirStore(tmp_path / "store")) as store:
        store.publish_node(Definition(CutItem, "retained").concretize())
        calls = []
        error = failure("invalid metadata or interruption")

        def fail(*args, **kwargs):
            calls.append(True)
            raise error

        monkeypatch.setattr(_NodePage, "expand", fail)
        with pytest.raises(failure) as raised:
            store.query().cdefs().stored().collect()
        assert raised.value is error and calls == [True]
