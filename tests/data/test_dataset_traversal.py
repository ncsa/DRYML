"""Local session ownership and lazy physical-pass handoff contracts."""

import itertools
import threading

import numpy as np
import pytest

from dryml.core.cardinality import Cardinality
from dryml.core.tensor_spec import TensorSpec
from dryml.data import ArrayDataset, Batch, Dataset, DatasetExhaustedError, GeneratorDataset
import dryml.data as data


_events = []


class Producer:
    def __init__(self, size=None, fail=False, close_fail=False):
        _events.append("open")
        self.values = iter(range(size)) if size is not None else itertools.count()
        self.fail = fail
        self.close_fail = close_fail

    def __iter__(self):
        return self

    def __next__(self):
        _events.append("pull")
        if self.fail:
            raise OSError("producer failed")
        return np.int64(next(self.values))

    def close(self):
        _events.append("close")
        if self.close_fail:
            raise RuntimeError("close failed")


def source(size=None, *, declared=Cardinality.UNKNOWN, **kwargs):
    return GeneratorDataset(Producer, size, cardinality=declared,
                            spec=TensorSpec("int64", shape=(), backend="numpy"), **kwargs)


@pytest.fixture(autouse=True)
def reset_events():
    _events.clear()


def test_ae1_adjacent_sections_keep_one_producer_without_boundary_overpull():
    with data.Epoch(source(), len=3).open() as traversal:
        assert traversal.position == data.TraversalPosition(0, 0, 0, False, False)
        first = traversal.next_epoch()
        assert [next(first), next(first), next(first)] == [0, 1, 2]
        assert traversal.position.section_complete
        assert (traversal.position.epoch, traversal.position.start) == (1, 0)
        assert _events.count("pull") == 3
        assert list(first) == []
        assert list(traversal.next_epoch()) == [3, 4, 5]
        assert _events.count("open") == 1
        assert _events.count("pull") == 6
    assert _events.count("close") == 1


def test_ae2_implicit_passes_reopen_once_each_and_unread_close_is_inert(monkeypatch):
    import dryml.data.source as sources
    original = sources._IndexedDatasetCursor
    opens = []

    def tracked(*args, **kwargs):
        opens.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(sources, "_IndexedDatasetCursor", tracked)
    prepared = ArrayDataset(np.arange(3)).prepare()
    unread = prepared.open()
    assert unread.position.start == 0
    unread.close()
    assert opens == []
    with prepared.open() as traversal:
        assert [(epoch, list(view)) for epoch, view in traversal.epochs(stop=3)] == [
            (0, [0, 1, 2]), (1, [0, 1, 2]), (2, [0, 1, 2])]
        assert not traversal.position.root_complete
    assert len(opens) == 3


def test_unfinished_view_rejects_competition_but_independent_sessions_interleave():
    prepared = data.Epoch(source(), len=2).prepare()
    with prepared.open() as first, prepared.open() as second:
        left, right = first.next_epoch(), second.next_epoch()
        assert next(left) == next(right) == 0
        with pytest.raises(data.TraversalOwnershipError):
            first.next_epoch()
        assert next(right) == 1
        assert list(second.next_epoch()) == [2, 3]
        assert next(left) == 1
        assert list(first.next_epoch()) == [2, 3]
        assert list(left) == []
    assert _events.count("open") == 2


def test_unknown_eof_handoff_is_idempotent_and_precedes_next_pass_acquisition():
    with source(3).open(start=3) as traversal:
        # An empty selected suffix is not an empty physical pass.
        traversal._hold_boundary_handoff = True
        with pytest.raises(StopIteration):
            traversal.next_epoch()
        position = traversal.position
        assert (position.epoch, position.start, position.root_complete) == (1, 0, False)
        assert traversal._boundary_pending
        assert _events.count("open") == 1
        assert _events.count("close") == 1
        with pytest.raises(StopIteration):
            traversal.next_epoch()
        assert traversal.position == position
        traversal._handoff()
        assert _events.count("open") == 1
        assert list(traversal.next_epoch()) == [0, 1, 2]
        assert _events.count("open") == 2
    with source(0).open() as traversal:
        assert list(traversal.epochs()) == []
        assert traversal.position.root_complete


def test_unknown_exact_explicit_eof_has_no_phantom_section():
    with data.Epoch(source(3), len=3).open() as traversal:
        assert list(traversal.next_epoch()) == [0, 1, 2]
        assert not traversal.position.root_complete
        assert _events.count("pull") == 3
        assert list(traversal.epochs()) == []
        assert traversal.position.root_complete
        assert _events.count("pull") == 4


@pytest.mark.parametrize("sectioned", (False, True))
def test_positive_declared_empty_pass_is_a_shortfall_not_terminal_success(sectioned):
    dataset = source(0, declared=3, close_fail=True)
    if sectioned:
        dataset = data.Epoch(dataset, len=2)
    traversal = dataset.open()
    with pytest.raises(DatasetExhaustedError) as caught:
        traversal.next_epoch()
    assert (caught.value.requested, caught.value.yielded) == (3, 0)
    assert any("close failed" in note for note in caught.value.__notes__)
    assert not traversal.position.root_complete
    assert _events.count("close") == 1


@pytest.mark.parametrize("kind", ("source", "chain", "repeat", "zip"))
def test_finite_generator_final_output_normalizes_without_eof_pull(kind):
    src = source(2, declared=2)
    dataset = {"source": lambda: src,
               "chain": lambda: data.Chain(src, source(1, declared=1)),
               "repeat": lambda: data.Repeat(src, 2),
               "zip": lambda: data.Zip(src, source(3, declared=3))}[kind]()
    expected = {"source": [0, 1], "chain": [0, 1, 0],
                "repeat": [0, 1, 0, 1], "zip": [(0, 0), (1, 1)]}[kind]
    prepared = dataset.prepare()
    src.cardinality = Cardinality.UNKNOWN  # finality belongs to the frozen plan
    with prepared.open(acceptance="explicit") as traversal:
        view = traversal.next_epoch()
        for index, value in enumerate(expected):
            candidate = view.reserve()
            assert candidate.value == value
            assert candidate.section_complete is (index == len(expected) - 1)
            traversal.accept(candidate)
        assert (traversal.position.epoch, traversal.position.start) == (1, 0)
        assert not traversal.position.root_complete
        assert _events.count("pull") == (4 if kind == "zip" else len(expected))
        with pytest.raises(StopIteration):
            view.reserve()


@pytest.mark.parametrize("kind", ("chain", "repeat", "zip", "batch"))
def test_finite_composite_cannot_hide_a_generator_source_shortfall(kind):
    src = source(2, declared=3)
    dataset = {"chain": lambda: data.Chain(src, source(1, declared=1)),
               "repeat": lambda: data.Repeat(src, 2),
               "zip": lambda: data.Zip(src, source(4, declared=4)),
               "batch": lambda: data.Batch(src, 2)}[kind]()
    with dataset.open() as traversal:
        with pytest.raises(DatasetExhaustedError) as caught:
            list(traversal.next_epoch())
        assert (caught.value.requested, caught.value.yielded) == (3, 2)
        assert not traversal.position.section_complete


def test_finite_chain_final_output_does_not_bypass_trailing_source_validation():
    """Zero-output trailing operators still owe their source's count validation."""
    dataset = data.Chain(data.Batch(source(1, declared=1), 4),
                         data.Batch(source(2, declared=3), 4, drop_remainder=True))
    assert dataset.yield_cardinality() == Cardinality.finite(1)
    with dataset.open(acceptance="explicit") as traversal:
        view = traversal.next_epoch()
        candidate = view.reserve()
        assert candidate.value == 0 and not candidate.section_complete
        traversal.accept(candidate)
        with pytest.raises(DatasetExhaustedError) as caught:
            view.reserve()
        assert (caught.value.requested, caught.value.yielded) == (3, 2)
        assert not traversal.position.section_complete


def test_open_and_iterate_endpoint_and_pair_semantics():
    dataset = data.Epoch(source(), len=3)
    with dataset.open(epoch=2, start=1) as traversal:
        assert [(epoch, list(view)) for epoch, view in traversal.epochs(stop=3)] == [(2, [7, 8])]
    with dataset.iterate(epoch=2, start=3) as cursor:
        assert list(cursor) == []
        assert cursor.observed_eof
    with pytest.raises(DatasetExhaustedError) as caught:
        list(dataset.iterate(start=4))
    assert (caught.value.requested, caught.value.yielded) == (4, 3)
    _events.clear()
    with dataset.open(epoch=2) as traversal:
        assert list(traversal.epochs(stop=2)) == []
        with pytest.raises(ValueError):
            traversal.epochs(stop=1)
    assert _events == []
    pairs = ArrayDataset((np.arange(3), np.arange(3) + 10))
    assert list(pairs.iterate(start=1)) == [(1, 11), (2, 12)]


@pytest.mark.parametrize("value", [True, np.int64(1), 1.0])
def test_new_controls_require_exact_ints(value):
    dataset = source(3)
    for kwargs in ({"epoch": value}, {"start": value}):
        with pytest.raises(TypeError):
            dataset.open(**kwargs)
        with pytest.raises(TypeError):
            dataset.iterate(**kwargs)
    with dataset.open() as traversal:
        with pytest.raises(TypeError):
            traversal.epochs(stop=value)
    with pytest.raises(TypeError):
        data.Epoch(dataset, len=value)
    assert _events == []


def test_known_infinite_unsectioned_request_rejects_without_acquisition():
    with pytest.raises(ValueError, match="infinite"):
        source(declared=Cardinality.INFINITE).open()
    assert _events == []


def test_failures_and_abandonment_close_once_preserving_primary():
    with pytest.raises(OSError, match="producer failed") as caught:
        with source(fail=True, close_fail=True).open() as traversal:
            traversal.next_epoch()
    assert any("close failed" in note for note in caught.value.__notes__)
    assert _events.count("close") == 1
    with source().open() as traversal:
        view = traversal.next_epoch()
        view.close()
        with pytest.raises(data.TraversalOwnershipError):
            traversal.next_epoch()
    assert _events.count("close") == 2
    primary = KeyboardInterrupt("consumer interrupted")
    with pytest.raises(KeyboardInterrupt) as caught:
        with source(close_fail=True).open() as traversal:
            next(traversal.next_epoch())
            raise primary
    assert caught.value is primary
    assert any("close failed" in note for note in primary.__notes__)


def test_owner_thread_and_recursive_pulls_reject_without_progress():
    with source(3).open() as traversal:
        errors = []

        def foreign():
            try:
                traversal.next_epoch()
            except BaseException as error:
                errors.append(error)

        thread = threading.Thread(target=foreign)
        thread.start()
        thread.join()
        assert isinstance(errors[0], data.TraversalOwnershipError)
        assert _events == []


def test_thousand_implicit_passes_release_physical_children_promptly():
    with source(1).open() as traversal:
        for ordinal, view in traversal.epochs(stop=1000):
            assert list(view) == [0]
            assert len(traversal._resources) <= 1
            assert _events.count("close") == ordinal + 1
    assert _events.count("open") == _events.count("close") == 1000


def test_live_candidates_are_lightweight_single_use_and_owner_bound():
    with data.Epoch(source(), len=1).open(acceptance="explicit") as traversal:
        view = traversal.next_epoch()
        candidate = view.reserve()
        assert candidate.value == 0 and candidate.section_complete
        assert traversal.position.sequence == 0
        assert not hasattr(candidate, "state")
        with pytest.raises(data.TraversalOwnershipError):
            view.reserve()
        traversal.accept(candidate)
        assert traversal.position.sequence == 1
        with pytest.raises(data.TraversalOwnershipError):
            traversal.accept(candidate)


def test_exact_end_enumeration_skips_selected_suffix_and_honors_stop_before_staging():
    dataset = data.Epoch(source(), len=3)
    with dataset.open(start=3) as traversal:
        assert [(ordinal, list(view)) for ordinal, view in traversal.epochs(stop=2)] == [(1, [3, 4, 5])]
    _events.clear()
    with dataset.open(start=3) as traversal:
        assert list(traversal.epochs(stop=1)) == []
        assert _events.count("pull") == 3


def test_invalid_view_access_does_not_poison_the_current_borrow():
    with data.Epoch(source(), len=2).open(acceptance="explicit") as traversal:
        view = traversal.next_epoch()
        with pytest.raises(data.TraversalOwnershipError):
            next(view)
        candidate = view.reserve()
        traversal.accept(candidate)
        traversal.accept(view.reserve())
        assert traversal.position.epoch == 1
    with source(2).open() as traversal:
        view = traversal.next_epoch()
        errors = []

        def foreign():
            try:
                next(view)
            except BaseException as error:
                errors.append(error)

        thread = threading.Thread(target=foreign)
        thread.start()
        thread.join()
        assert isinstance(errors[0], data.TraversalOwnershipError)
        assert list(view) == [0, 1]


def test_public_sessions_retain_map_collator_splitter_and_frozen_controls(monkeypatch):
    import dryml.data.structural as structural
    from tests.data.test_traversal_preparation import Identity, AddTen

    method = Identity()
    dataset = data.Epoch(data.Unbatch(Batch(data.Map(ArrayDataset(np.arange(6)), method), 2)), len=3)
    calls = {"map": 0, "collate": 0, "split": 0}

    def counted(key, function):
        def call(*args, **kwargs):
            calls[key] += 1
            return function(*args, **kwargs)
        return call

    monkeypatch.setattr(method, "find_implementation", counted("map", method.find_implementation))
    monkeypatch.setattr(structural, "collate_for_spec", counted("collate", structural.collate_for_spec))
    monkeypatch.setattr(structural, "split_for_spec", counted("split", structural.split_for_spec))
    prepared = dataset.prepare()
    assert calls == {"map": 1, "collate": 1, "split": 1}
    dataset.length = 2
    dataset.src.src.src.method = AddTen()
    for _ in range(2):
        with prepared.open() as traversal:
            assert [(i, list(view)) for i, view in traversal.epochs()] == [(0, [0, 1, 2]), (1, [3, 4, 5])]
    assert calls == {"map": 1, "collate": 1, "split": 1}


def test_recursive_advance_rejects_at_the_source_boundary(monkeypatch):
    import dryml.data.source as sources
    fresh = sources._fresh_iterable
    with source(2).open() as traversal:
        def acquire(*args, **kwargs):
            with pytest.raises(data.TraversalOwnershipError, match="Recursive"):
                traversal.next_epoch()
            return fresh(*args, **kwargs)

        monkeypatch.setattr(sources, "_fresh_iterable", acquire)
        assert list(traversal.next_epoch()) == [0, 1]


def test_enumerators_are_exclusive_and_limited_enumeration_leaves_owner_usable():
    with data.Epoch(source(), len=1).open() as traversal:
        first, competing = traversal.epochs(stop=2), traversal.epochs(stop=2)
        ordinal, view = next(first)
        with pytest.raises(data.TraversalOwnershipError):
            next(competing)
        assert ordinal == 0 and list(view) == [0]
        assert [(i, list(view)) for i, view in first] == [(1, [1])]
        assert list(traversal.next_epoch()) == [2]


def test_candidate_aliases_foreign_tokens_and_rejection_preserve_accepted_cut():
    from dataclasses import replace

    prepared = data.Epoch(source(), len=3).prepare()
    with prepared.open(acceptance="explicit") as first, prepared.open(acceptance="explicit") as second:
        view = first.next_epoch()
        candidate = view.reserve()
        with pytest.raises(data.TraversalOwnershipError):
            second.accept(candidate)
        first.accept(replace(candidate, value="converted"))
        with pytest.raises(data.TraversalOwnershipError):
            first.accept(candidate)
        position = first.position
        candidate = view.reserve()
        first.reject(candidate)
        assert first.position == position
        with pytest.raises(data.TraversalOwnershipError):
            first.next_epoch()


def test_twenty_sections_have_exact_work_and_no_capture_or_payload_copy(monkeypatch):
    import dill
    import pickle
    import copy

    def forbidden(*args, **kwargs):
        pytest.fail("steady traversal copied or encoded continuation state")

    prepared = data.Epoch(source(), len=7).prepare()
    monkeypatch.setattr(pickle, "dumps", forbidden)
    monkeypatch.setattr(dill, "dumps", forbidden)
    monkeypatch.setattr(copy, "deepcopy", forbidden)
    with prepared.open() as traversal:
        for ordinal, view in traversal.epochs(stop=20):
            assert list(view) == list(range(ordinal * 7, ordinal * 7 + 7))
    assert _events.count("open") == 1
    assert _events.count("pull") == 140


def test_candidate_alias_facts_cannot_replace_the_reserved_frontier():
    from dataclasses import replace

    with data.Epoch(source(), len=1).open(acceptance="explicit") as traversal:
        candidate = traversal.next_epoch().reserve()
        traversal.accept(replace(candidate, section_complete=False))
        assert traversal.position.epoch == 1
        assert traversal.position.start == 0


def test_foreign_view_close_and_recursive_close_leave_valid_owner_usable(monkeypatch):
    import dryml.data.source as sources
    with source(2).open() as traversal:
        original = sources._fresh_iterable

        def acquire(*args, **kwargs):
            with pytest.raises(data.TraversalOwnershipError):
                traversal.close()
            return original(*args, **kwargs)

        monkeypatch.setattr(sources, "_fresh_iterable", acquire)
        view = traversal.next_epoch()
        errors = []

        def foreign():
            try:
                view.close()
            except BaseException as error:
                errors.append(error)

        thread = threading.Thread(target=foreign)
        thread.start()
        thread.join()
        assert isinstance(errors[0], data.TraversalOwnershipError)
        assert list(view) == [0, 1]


def test_known_implicit_bound_normalizes_before_an_extra_pull_and_preserves_handoff():
    dataset = data.Take(source(), 3)
    with dataset.open() as traversal:
        traversal._hold_boundary_handoff = True
        view = traversal.next_epoch()
        assert [next(view), next(view), next(view)] == [0, 1, 2]
        assert (traversal.position.epoch, traversal.position.start) == (1, 0)
        assert traversal.position.section_complete and not traversal.position.root_complete
        assert _events.count("pull") == 3
        assert _events.count("close") == 1
        position = traversal.position
        assert list(view) == []
        assert traversal.position == position
        with pytest.raises(StopIteration):
            traversal.next_epoch()
        assert _events.count("open") == 1


def test_exhausted_explicit_view_remains_exhausted_after_owner_close():
    traversal = data.Epoch(source(), len=1).open(acceptance="explicit")
    view = traversal.next_epoch()
    traversal.accept(view.reserve())
    traversal.close()
    with pytest.raises(StopIteration):
        view.reserve()


def test_accept_cleanup_cannot_recursively_advance_an_implicit_owner(monkeypatch):
    original = Producer.close
    with data.Take(source(), 1).open(acceptance="explicit") as traversal:
        def close(producer):
            with pytest.raises(data.TraversalOwnershipError):
                traversal.next_epoch()
            original(producer)

        monkeypatch.setattr(Producer, "close", close)
        view = traversal.next_epoch()
        traversal.accept(view.reserve())
        assert _events.count("open") == _events.count("close") == 1


def test_foreign_context_exit_cannot_close_the_borrowed_owner():
    with source(2).open() as traversal:
        view = traversal.next_epoch()
        errors = []

        def foreign():
            try:
                with view:
                    next(view)
            except BaseException as error:
                errors.append(error)

        thread = threading.Thread(target=foreign)
        thread.start()
        thread.join()
        assert isinstance(errors[0], data.TraversalOwnershipError)
        assert list(view) == [0, 1]


def test_steady_map_validation_uses_retained_specs_without_decoding_control_graph(monkeypatch):
    from tests.data.test_traversal_preparation import Identity
    from dryml.methods.stream import _StreamOp

    dataset = data.Map(data.Epoch(ArrayDataset(np.arange(4)), len=2), Identity(), preserves_examples=True)
    with dataset.open() as traversal:
        view = traversal.next_epoch()

        def forbidden(op):
            pytest.fail("steady pulls decoded a frozen control-plane spec")

        monkeypatch.setattr(_StreamOp, "spec", property(forbidden))
        assert list(view) == [0, 1]
        assert list(traversal.next_epoch()) == [2, 3]
