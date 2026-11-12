"""Work bounds for exact metadata decoding during snapshot-heavy queries."""

import pytest

from dryml.core import Definition, Node, Object, Ref, Repo, Serializable
from dryml.core.store.dir import DirStore


class DecodeValue(Object):
    """Small metadata carrier with shared children and typed leaf values."""

    def __init__(self, value):
        self.value = value


class DecodeQuoted(Object):
    """Opaque saved Node configuration retained through an explicit Ref role."""

    def __init__(self, value: Ref[Node]):
        self.value = value


class DecodeState(Serializable):
    """Identified nested object repeated in independent snapshot contributions."""

    def __init__(self, value):
        self.value = value
        self.progress = 0

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        from pathlib import Path
        Path(dest_dir, "progress").write_text(str(self.progress), encoding="ascii")

    def restore_state_from_dir_imp(self, src_dir, *, codec):
        from pathlib import Path
        self.progress = int(Path(src_dir, "progress").read_text(encoding="ascii"))


@pytest.mark.parametrize("explicit_domain", (False, True))
def test_identified_object_query_uses_its_declaration_not_all_containing_snapshots(tmp_path, fixed_snapshot_environment, monkeypatch, explicit_domain):
    repo = Repo(DirStore(tmp_path / "store"))
    try:
        child = DecodeState(DecodeValue("target", repo=repo), repo=repo)
        parent = DecodeState(child, repo=repo)
        for index in range(6):
            parent.progress = index
            saved = repo.save(parent)
        expected = saved.parameters["value"].object_view()
    finally:
        repo.close(flush=False)
    repo = Repo(DirStore.open_existing(tmp_path / "store"))
    try:
        def no_payload(*args, **kwargs):
            raise AssertionError("object discovery opened private state")
        monkeypatch.setattr(repo.default_store, "read_states", no_payload)
        pattern = Definition(DecodeState, Definition(DecodeValue, "target"))
        query = repo.query().sel(pattern)
        reference = query.collect().object_projection()
        result = query.object_refs().collect() if explicit_domain else query.object_projection()
        assert result.one().graph_equal(expected)
        assert result.one().graph_equal(reference.one())
        assert result.sources(result.one()) == reference.sources(reference.one())
        assert result.diagnostic().cdef_blobs_decoded == 1
        assert repo._num_constructions == 0
    finally:
        repo.close(flush=False)


def test_exact_transport_values_do_not_repeat_canonical_folding(monkeypatch):
    from dryml.core import node_codec
    from dryml.core.node_prepare import NodePreparation

    shared = Definition(DecodeValue, {"enabled": True, "count": 1, "weight": 1.0})
    root = Definition(DecodeValue, [Definition(DecodeValue, [shared, index]) for index in range(16)])
    packet = node_codec.encode_node_graph(NodePreparation((root,)))
    folds = []
    original = node_codec._canonical_values

    def fold(*args, **kwargs):
        folds.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(node_codec, "_canonical_values", fold)
    restored = node_codec.decode_node_graph(packet)
    assert restored.graph_equal(root)
    assert not folds


@pytest.mark.parametrize("left,right", ((True, 1), (1, 1.0), (0.0, -0.0)))
def test_transport_syntax_shortcut_keeps_json_leaf_types_and_signed_zero(left, right):
    from dryml.core.node_codec import _same_value_syntax
    assert not _same_value_syntax({"value": left}, {"value": right})


def test_object_declaration_path_validates_nested_identity_dependencies(tmp_path, fixed_snapshot_environment):
    repo = Repo(DirStore(tmp_path / "store"))
    try:
        grandchild = DecodeState("leaf", repo=repo)
        child = DecodeState(grandchild, repo=repo)
        parent = DecodeState(child, repo=repo)
        saved = repo.save(parent)
        expected = saved.parameters["value"].object_view()
        pattern = Definition(DecodeState, Definition(DecodeState, "leaf"))
        result = repo.query().object_refs().sel(pattern).collect()
        assert result.one().graph_equal(expected)
        assert result.diagnostic().cdef_blobs_decoded == 1
        missing = saved.parameters["value"].parameters["value"].object_id
        from pathlib import Path
        Path(repo.default_store._node_path("declarations/required", repo.default_store._anchor_key(missing))).unlink()
        from dryml.core.query.source import InventoryCapabilityError
        with pytest.raises(InventoryCapabilityError, match="required declaration"):
            repo.query().object_refs().sel(pattern).collect()
    finally:
        repo.close(flush=False)


@pytest.mark.parametrize("hint_kind", ("unknown", "other-member"))
def test_object_identity_hint_is_only_an_address_not_membership_evidence(tmp_path, fixed_snapshot_environment, monkeypatch, hint_kind):
    import json
    from dryml.core import ObjectId
    from dryml.core._value_json import _value_to_json
    from dryml.core.query.source import InventoryCapabilityError
    from dryml.core.query.sqlite import lowering

    repo = Repo(DirStore(tmp_path / "store"))
    try:
        saved = repo.save(DecodeState(DecodeState("target", repo=repo), repo=repo))
        original = lowering._store_discovery_page_sql
        identity = ObjectId() if hint_kind == "unknown" else saved.object_id
        hint = json.dumps({"tag": "leaf", "value": _value_to_json(identity, grammar=3)})

        def wrong_hint(*args, **kwargs):
            sql, parameters = original(*args, **kwargs)
            if kwargs.get("object_hints"):
                _, rest = sql.split(" FROM store_projection_rows p", 1)
                sql = "SELECT p.source,p.row_key,p.payload,? FROM store_projection_rows p" + rest
                parameters = (hint, *parameters)
            return sql, parameters

        monkeypatch.setattr(lowering, "_store_discovery_page_sql", wrong_hint)
        message = "source identity membership" if hint_kind == "unknown" else "semantic digest"
        with pytest.raises(InventoryCapabilityError, match=message):
            repo.query().object_refs().sel(Definition(DecodeState, "target")).collect()
    finally:
        repo.close(flush=False)


def test_object_declaration_path_checks_conflicts_in_nonmatching_stores(tmp_path):
    from dryml.core import ObjectId
    from dryml.core.query import QueryError

    first, second = DirStore(tmp_path / "first"), DirStore(tmp_path / "second")
    repo = Repo((first, second))
    try:
        identity = ObjectId()
        first.publish_node(Definition(DecodeState, "target").concretize().with_identity(identity))
        second.publish_node(Definition(DecodeState, "other").concretize().with_identity(identity))
        with pytest.raises(QueryError, match="authority conflicts"):
            repo.query().object_refs().sel(Definition(DecodeState, "target")).collect()
    finally:
        repo.close(flush=False)


def test_multiple_matching_objects_in_one_snapshot_share_root_validation(tmp_path, fixed_snapshot_environment, monkeypatch):
    repo = Repo(DirStore(tmp_path / "store"))
    try:
        parent = DecodeState([DecodeState("target", repo=repo), DecodeState("target", repo=repo)], repo=repo)
        repo.save(parent)
        original = repo.default_store._publication_identities
        visits = []

        def counted(node):
            visits.append(node)
            return original(node)

        monkeypatch.setattr(repo.default_store, "_publication_identities", counted)
        result = repo.query().object_refs().sel(Definition(DecodeState, "target")).collect()
        assert len(result) == 2 and len(visits) == 1
    finally:
        repo.close(flush=False)


def test_object_declaration_fast_path_keeps_stateless_fallback(tmp_path, fixed_snapshot_environment):
    repo = Repo(DirStore(tmp_path / "store"))
    try:
        saved = repo.save(DecodeValue("target", repo=repo))
        assert saved.object_id is None
        result = repo.query().object_refs().sel(Definition(DecodeValue, "target")).one()
        assert result.graph_equal(saved.object_view())
    finally:
        repo.close(flush=False)


def test_pinned_artifact_completion_avoids_containing_snapshot_hydration(tmp_path, fixed_snapshot_environment, monkeypatch):
    from dryml.managed import ManagedConfig
    from tests.models.test_experiment_artifacts import ConstantValue
    from dryml.core.query.source import _NodePage

    repo = Repo(DirStore(tmp_path / "store"))
    try:
        value = ConstantValue(repo=repo)
        result = value.compute(managed=ManagedConfig(state_repo=repo))
        parent = DecodeState(value, repo=repo)
        for index in range(6):
            parent.progress = index
            repo.save(parent)
        observed = []
        expand = _NodePage.expand

        def count(page, capture, *args, **kwargs):
            answer = expand(page, capture, *args, **kwargs)
            observed.append((capture.candidate_rows_read, capture.cdef_blobs_decoded))
            return answer

        monkeypatch.setattr(_NodePage, "expand", count)
        selected = ConstantValue.find_completed_state_ref(result.final_state_ref.object_view(), repo=repo)
        assert selected.graph_equal(result.final_state_ref)
        assert observed and max(rows for rows, _ in observed) <= 2

        from dryml.core.node_prepare import NodePreparation
        digest = NodePreparation((selected,)).digest(selected, domain="dryml-saved-root/v1")
        repo.default_store.withdraw_node(digest, kind="saved")
        embedded = ConstantValue.find_completed_state_ref(result.final_state_ref.object_view(), repo=repo)
        assert embedded.graph_equal(result.final_state_ref)
    finally:
        repo.close(flush=False)


def test_object_projection_retains_state_sensitive_selection_and_explicit_domains(tmp_path, fixed_snapshot_environment):
    from dataclasses import replace
    from dryml.core import QueryPredicate

    repo = Repo(DirStore(tmp_path / "store"))
    try:
        value = DecodeState("target", repo=repo)
        first = repo.save(value)
        value.progress = 1
        second = repo.save(value)
        constrained = Definition(DecodeState, "target").with_state(first.state)
        for query in (
            repo.query().sel(constrained),
            repo.query().sel(QueryPredicate(constrained)),
            repo.query().saved().sel(Definition(DecodeState, "target")),
            replace(repo.query().sel(Definition(DecodeState, "target")), take_limit=1),
        ):
            expected = query.collect().object_projection()
            actual = query.object_projection()
            assert len(expected) == len(actual)
            assert all(any(value.graph_equal(other) for other in actual) for value in expected)
            assert actual.bounded == expected.bounded
            assert actual.requested_limit == expected.requested_limit
        assert repo.query().sel(constrained).object_projection().one().graph_equal(first.object_view())
        assert repo.query().object_refs().sel(constrained).count() == 0
        assert not first.graph_equal(second)

        holder = Definition(DecodeQuoted, Ref(first)).concretize()
        repo.default_store.publish_node(holder, kind="saved")
        selected = repo.query().sel(holder).object_projection()
        assert selected.one().graph_equal(holder.object_view())
    finally:
        repo.close(flush=False)


def test_projection_fusion_proof_refuses_state_and_alias_sensitive_syntax():
    from dryml.core import Ref
    from dryml.core.params import AnyValue
    from dryml.core.query.node_pattern import _projection_invariant_chain

    leaf = Definition(DecodeValue, "leaf")
    assert _projection_invariant_chain(Definition(DecodeState, Ref(leaf)))
    for pattern in (
        Definition(DecodeState, [leaf, leaf]),
        Definition(DecodeState, [leaf, Definition(DecodeValue, "other")]),
        Definition(DecodeState, Ref(leaf.quote())),
        Definition(DecodeState, AnyValue()),
    ):
        assert not _projection_invariant_chain(pattern)
    values = ("leaf",)
    for _ in range(11):
        values = values, values
    assert not _projection_invariant_chain(Definition(DecodeValue, values))


def test_default_projection_fusion_preserves_stateless_embedded_and_cache_sources(tmp_path, fixed_snapshot_environment):
    from dryml.core.query import IdentityQuery
    from dryml.core.query.source import RepoSource

    repo = Repo(DirStore(tmp_path / "store"))
    try:
        root = DecodeValue(DecodeValue("target", repo=repo), repo=repo)
        repo.save(root)
        pattern = Definition(DecodeValue, "target")
        for source in (repo.query(), repo.default_store.query(), IdentityQuery(RepoSource(repo))):
            query = source.sel(pattern)
            expected, actual = query.collect().object_projection(), query.object_projection()
            assert len(expected) == len(actual) == 1
            assert actual.one().graph_equal(expected.one())
            assert actual.sources(actual.one()) == expected.sources(expected.one())
    finally:
        repo.close(flush=False)
