"""Incremental query terminals retain detached authority and explicit lifetimes."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager
from dataclasses import replace
import gc
import threading
import weakref

import pytest

from dryml.core import Definition, Object, ObjectId, ObjectRef, Repo, SaveAnnotations, Serializable, StateRef
from dryml.core.query import IdentitySet, QueryError, QueryIndexUnavailable, QueryVerifyBudgetExceeded, QueryWouldScanError, field
from dryml.core.query import query as query_module
from dryml.core.query.identity import identity_key
from dryml.core.query.query import IdentityQuery
from dryml.core.query.source import SourceCapture, _SourceRecapture
from dryml.core.store.dir import DirStore
from dryml.core.store.records import DefinitionRecord
from dryml.core.utils.graph.path import GraphPath


class IterationLeaf(Serializable):
    """Small complete identity with optional synthetic ObjectIds and state hashes."""

    def __init__(self, name):
        self.name = name

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        pass


class IterationRoot(Object):
    """Retain multiple child paths for incremental relationship traversal."""

    def __init__(self, children):
        self.children = children


def _values():
    return tuple(Definition(IterationLeaf, name).concretize() for name in ('a', 'b', 'c'))


def test_identity_iterator_is_deferred_incremental_and_not_a_collect_wrapper(monkeypatch):
    """The first request evaluates one match, and early close avoids later matches."""
    fixed = IdentitySet(_values())
    expected = tuple(fixed)
    query = fixed.query().sel(Definition(IterationLeaf))
    calls = []
    original = IdentityQuery._matches

    def counted(self, restriction, value, *args, **kwargs):
        calls.append(value)
        return original(self, restriction, value, *args, **kwargs)

    monkeypatch.setattr(IdentityQuery, '_matches', counted)
    monkeypatch.setattr(IdentityQuery, 'collect', lambda *_: pytest.fail('iterator called collect'))
    monkeypatch.setattr(IdentitySet, '_from_entries', lambda *_a, **_k: pytest.fail('built final result'))
    stream = query.iter()
    assert iter(stream) is stream
    assert calls == []
    assert next(stream) == expected[0]
    assert calls == [expected[0]]
    stream.close()
    assert calls == [expected[0]]
    assert list(stream) == []


def test_iterator_captures_on_first_next_and_holds_no_store_fence_between_yields(tmp_path, monkeypatch):
    """Pre-start writes participate; writes during consumption affect only later terminals."""
    store = DirStore(tmp_path / 'store', query_index='memory')
    first, second, third = _values()
    store.write_definition_record(DefinitionRecord(first))
    query = store.query().cdefs().stored()
    active = 0
    original = store.authority_read_fence

    @contextmanager
    def fence():
        nonlocal active
        with original():
            active += 1
            try:
                yield
            finally:
                active -= 1

    monkeypatch.setattr(store, 'authority_read_fence', fence)
    stream = query.iter()
    store.write_definition_record(DefinitionRecord(second))
    expected = tuple(query.collect())
    head = next(stream)
    assert active == 0
    store.write_definition_record(DefinitionRecord(third))
    assert (head, *stream) == expected
    assert len(tuple(query.iter())) == 3


def test_closing_an_unstarted_iterator_never_captures_or_refreshes(tmp_path, monkeypatch):
    """Closing before first advancement releases the plan without executing it."""
    store = DirStore(tmp_path / 'store')
    monkeypatch.setattr(
        SourceCapture, '__init__', lambda *_: pytest.fail('unstarted iterator acquired a capture'),
    )
    store.query().refresh(True).iter().close()
    store.query().nested().iter().close()
    assert list(store.query().refresh(True).nested().max_occurrences(0).iter()) == []


@pytest.mark.parametrize('finish', ('close', 'exhaust'))
def test_iterator_releases_its_capture_on_close_or_exhaustion(monkeypatch, finish):
    """A suspended iterator owns its snapshot until explicit close or exhaustion."""
    captures = []
    original = SourceCapture.__init__

    def track(self, *args, **kwargs):
        original(self, *args, **kwargs)
        captures.append(weakref.ref(self))

    monkeypatch.setattr(SourceCapture, '__init__', track)
    stream = IdentitySet(_values()).query().iter()
    next(stream)
    assert captures[-1]() is not None
    if finish == 'close':
        stream.close()
    else:
        tuple(stream)
    gc.collect()
    assert captures[-1]() is None


def test_identity_iterator_explicitly_closes_its_delegated_matching_generator(monkeypatch):
    """Closing owns subiterator cleanup even when another reference retains it."""
    original = IdentityQuery._iter_terminal_members
    delegated = []

    def track(self, *args, **kwargs):
        items = original(self, *args, **kwargs)
        delegated.append(items)
        return items

    monkeypatch.setattr(IdentityQuery, '_iter_terminal_members', track)
    stream = IdentitySet(_values()).query().iter()
    next(stream)
    stream.close()
    assert delegated[0].gi_frame is None


def test_iterator_keeps_complete_identity_order_deduplication_and_reference_values():
    """All identity kinds retain the same deterministic order as fixed collection."""
    cdef = _values()[0]
    first = ObjectRef(cdef, {GraphPath(): ObjectId()})
    second = ObjectRef(cdef, {GraphPath(): ObjectId()})
    state = StateRef(first, {GraphPath(): 'pkl-' + '1' * 64})
    query = IdentitySet((state, second, cdef, first, first)).query()
    assert tuple(query.iter()) == tuple(query.collect())
    assert set(map(identity_key, query.object_refs().iter())) == {identity_key(first), identity_key(second)}
    assert tuple(query.state_refs().iter()) == (state,)
    with pytest.raises(TypeError):
        iter(query)


def test_repo_reference_iteration_keeps_multiple_realizations_without_loading(tmp_path, monkeypatch):
    """The notebook pattern returns distinct ObjectRefs sharing one definition."""
    repo = Repo(DirStore(tmp_path / 'store', query_index='memory'))
    objects = [IterationLeaf('same', repo=repo) for _ in range(2)]
    expected = {identity_key(repo.save_object(obj).object) for obj in objects}
    assert len(expected) == 2
    monkeypatch.setattr(repo, 'load_object', lambda *_a, **_k: pytest.fail('query loaded an Object'))
    monkeypatch.setattr(repo, 'load_state_ref', lambda *_a, **_k: pytest.fail('query loaded a payload'))
    query = repo.query().sel(objects[0].definition).object_refs()
    assert set(map(identity_key, query.iter())) == expected


def test_identity_iterator_preserves_graph_distinctions_under_digest_collisions(monkeypatch):
    """Collision buckets retain graph topology and the collection's stable tie order."""
    from dryml.core.definition import ConcreteDefinition

    child = _values()[0]
    shared = Definition(IterationRoot, (child, child)).concretize()
    separate = Definition(IterationRoot, (_values()[0], _values()[0])).concretize()
    assert shared == separate and not shared.graph_equal(separate)
    monkeypatch.setattr(ConcreteDefinition, 'graph_hash', lambda _: '0' * 64)
    query = IdentitySet((separate, shared)).query()
    assert tuple(map(identity_key, query.iter())) == tuple(map(identity_key, query.collect()))
    assert len(tuple(query.iter())) == 2


def test_iterators_preserve_algebra_closure_projection_and_fixed_prefixes():
    """Intermediate set-producing operators retain their existing semantics."""
    first, second, third = _values()
    left, right = IdentitySet((first, second)).query(), IdentitySet((second, third)).query()
    for query in (left.union(right), left.intersection(right), left.take(1).query()):
        assert tuple(query.iter()) == tuple(query.collect())
    root = Definition(IterationRoot, (first, first, second)).concretize()
    roots = IdentitySet((root,)).query()
    for query in (roots.closure(), roots.nested(first).owners(), roots.nested(first).targets()):
        assert tuple(query.iter()) == tuple(query.collect())


@pytest.mark.parametrize('policy,exception', (('forbid', QueryWouldScanError), ('indexed', QueryIndexUnavailable)))
def test_iterator_defers_scan_policy_failures_until_first_next(tmp_path, policy, exception):
    """Laziness does not bypass source/index policy or perform a forbidden read."""
    query = DirStore(tmp_path / 'store', query_index='memory').query()
    query = query.require_indexed() if policy == 'indexed' else query.scan_policy(policy)
    stream = query.iter()
    with pytest.raises(exception):
        next(stream)
    assert list(stream) == []


def test_iterator_admits_the_candidate_domain_before_emitting_a_prefix():
    """Early consumer termination cannot weaken the complete-inventory budget."""
    query = IdentitySet(_values()).query().max_verify(1)
    with pytest.raises(QueryVerifyBudgetExceeded):
        next(query.iter())


def test_iterator_validates_late_metadata_before_any_output(tmp_path):
    """A valid first selection cannot conceal another eligible record's bad metadata."""
    repo = Repo(DirStore(tmp_path / 'store', query_index='memory'))
    repo.save_object(IterationLeaf('bad', repo=repo), annotations=SaveAnnotations(object={'score': 'bad'}))
    good = repo.save_object(IterationLeaf('good', repo=repo), annotations=SaveAnnotations(object={'score': 1}))
    query = repo.query().sel(good).where(field('object', 'score').lt(3))
    with pytest.raises(Exception, match='numeric field'):
        next(query.iter())


def test_iterator_retries_recapture_only_before_first_output(monkeypatch):
    """A pre-output cut change retries; a later one fails without replaying members."""
    query = IdentitySet(_values()).query()
    original = IdentityQuery._iter_terminal_members
    attempts = 0

    def recapture(self, capture, *args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise _SourceRecapture(True)
        for index, item in enumerate(original(self, capture, *args, **kwargs)):
            if index == 1:
                raise _SourceRecapture(True)
            yield item

    monkeypatch.setattr(IdentityQuery, '_iter_terminal_members', recapture)
    stream = query.iter()
    next(stream)
    assert attempts == 2
    with pytest.raises(QueryError, match='after iteration began'):
        next(stream)
    assert attempts == 2
    assert list(stream) == []


def test_iterator_defers_refresh_and_preserves_shared_request_cadence(tmp_path, monkeypatch):
    """Iterators join the same one-shot lifecycle as other explicit terminals."""
    store = DirStore(tmp_path / 'store')
    store.write_definition_record(DefinitionRecord(_values()[0]))
    store.rebuild_query_index()
    index = store.open_query_index()
    original = index.refresh
    calls = []

    def refresh(policy, *, stats=None):
        calls.append(policy)
        return original(policy, stats=stats)

    monkeypatch.setattr(index, 'refresh', refresh)
    once = store.query().refresh(True)
    stream = once.iter()
    assert calls == []
    assert tuple(stream)
    assert tuple(once.iter())
    assert calls == [True]
    repeated = once.refresh('always')
    assert tuple(repeated.iter())
    assert tuple(repeated.iter())
    assert calls == [True, 'always', 'always']


def test_occurrence_iteration_streams_paths_and_keeps_raw_caps(monkeypatch):
    """Traversal stops when the consumer closes, preserving distinct shared paths."""
    from dryml.core.query import relationships

    child = _values()[0]
    root = Definition(IterationRoot, (child, child, child)).concretize()
    query = IdentitySet((root,)).query().nested(child)
    expected = tuple(query.collect())
    original = relationships.iter_direct_relationships
    visits = []

    def counted(value):
        visits.append(value)
        yield from original(value)

    monkeypatch.setattr(relationships, 'iter_direct_relationships', counted)
    with closing(query.iter()) as stream:
        assert visits == []
        assert next(stream) == expected[0]
        assert visits == [root]
    assert visits == [root]
    assert tuple(query.iter()) == expected
    assert len(tuple(query.max_occurrences(1).iter())) == 1
    assert tuple(query.collect().query().iter()) == expected


def test_occurrence_iterator_propagates_late_depth_failure_without_replaying():
    """A delivered shallow match is not a claim that deeper traversal succeeded."""
    from dryml.core.query.relationships import RelationshipDepthExceeded

    child = _values()[0]
    middle = Definition(IterationRoot, (child,)).concretize()
    root = Definition(IterationRoot, (middle,)).concretize()
    stream = IdentitySet((root,)).query().nested().max_depth(1).iter()
    assert next(stream).target == middle
    with pytest.raises(RelationshipDepthExceeded):
        next(stream)
    assert list(stream) == []


def test_iterator_propagates_interruption_and_closes(monkeypatch):
    """A late interruption leaves a terminally closed iterator, not a retry."""
    query = IdentitySet(_values()).query().sel(Definition(IterationLeaf))
    original = IdentityQuery._matches
    calls = 0

    def interrupt(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt('controlled iteration interruption')
        return original(self, *args, **kwargs)

    monkeypatch.setattr(IdentityQuery, '_matches', interrupt)
    stream = query.iter()
    next(stream)
    with pytest.raises(KeyboardInterrupt, match='controlled iteration'):
        next(stream)
    assert list(stream) == []


def test_independent_iterators_support_concurrent_consumers(tmp_path):
    """Each consumer gets its own capture even when it reuses an immutable plan."""
    store = DirStore(tmp_path / 'store', query_index='memory')
    for value in _values():
        store.write_definition_record(DefinitionRecord(value))
    query = store.query().cdefs()
    expected = tuple(query.collect())
    paused = threading.Barrier(2, timeout=60)

    def consume(_):
        with closing(query.iter()) as stream:
            first = next(stream)
            paused.wait()
            return (first, *stream)

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(consume, range(2))) == [expected, expected]


def test_suspended_iterator_rejects_a_different_consumer_thread():
    """A started iterator cannot transfer its retained execution to another thread."""
    stream = IdentitySet(_values()).query().iter()
    next(stream)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(next, stream)
        with pytest.raises(QueryError, match='process and thread'):
            future.result()
    assert list(stream) == []


def test_suspended_iterator_rejects_a_changed_process_identity(monkeypatch):
    """An inherited started iterator fails before advancing its detached execution."""
    stream = IdentitySet(_values()).query().iter()
    next(stream)
    owner = query_module._iterator_owner()
    monkeypatch.setattr(query_module, '_iterator_owner', lambda: (owner[0] + 1, owner[1]))
    with pytest.raises(QueryError, match='process and thread'):
        next(stream)
    assert list(stream) == []


def test_bounded_identity_plan_iterates_only_its_retained_prefix():
    """An operand-local terminal bound is not erased by the streaming terminal."""
    query = replace(IdentitySet(_values()).query(), take_limit=1)
    assert tuple(query.iter()) == tuple(query.collect())
