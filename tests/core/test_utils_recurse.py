from collections import namedtuple

import pytest

from dryml.core2.errors import CycleError
from dryml.core2.utils.recurse import (
    cycle_detect,
    map_leaves,
    iter_leaves,
    first_leaf,
    leaf_values,
)


Pair = namedtuple("Pair", ["x", "y"])


# ---------------------------------------------------------------------------
# map_leaves
# ---------------------------------------------------------------------------

def test_map_leaves_basic_nested_structure():
    x = {
        "a": 1,
        "b": (2, [3, 4]),
        "c": Pair(5, 6),
    }

    y = map_leaves(x, lambda v: v * 10)

    assert y == {
        "a": 10,
        "b": (20, [30, 40]),
        "c": Pair(50, 60),
    }
    assert isinstance(y["c"], Pair)


def test_map_leaves_predicate_only_maps_matching_leaves():
    x = {
        "a": 1,
        "b": "hello",
        "c": [2, "world", 3],
    }

    y = map_leaves(x, lambda v: v + 100, pred=lambda v: isinstance(v, int))

    assert y == {
        "a": 101,
        "b": "hello",
        "c": [102, "world", 103],
    }


def test_map_leaves_preserves_empty_containers():
    x = {
        "a": [],
        "b": (),
        "c": {},
        "d": Pair([], {}),
    }

    y = map_leaves(x, lambda v: "changed")

    assert y == {
        "a": [],
        "b": (),
        "c": {},
        "d": Pair([], {}),
    }


# ---------------------------------------------------------------------------
# iter_leaves / leaf_values
# ---------------------------------------------------------------------------

def test_iter_leaves_left_to_right_order():
    x = {
        "a": 1,
        "b": (2, [3, 4]),
        "c": Pair(5, 6),
    }

    vals = list(iter_leaves(x))

    assert vals == [1, 2, 3, 4, 5, 6]


def test_iter_leaves_predicate_filters_output():
    x = {
        "a": 1,
        "b": "hello",
        "c": [2, "world", 3],
    }

    vals = list(iter_leaves(x, pred=lambda v: isinstance(v, int)))

    assert vals == [1, 2, 3]


def test_iter_leaves_empty_tree_yields_nothing():
    assert list(iter_leaves([])) == []
    assert list(iter_leaves(())) == []
    assert list(iter_leaves({})) == []
    assert list(iter_leaves(Pair([], {}))) == []


def test_leaf_values_collects_all_leaves():
    x = {
        "a": 1,
        "b": (2, [3, 4]),
        "c": Pair(5, 6),
    }

    assert leaf_values(x) == [1, 2, 3, 4, 5, 6]


def test_leaf_values_with_predicate():
    x = {
        "a": 1,
        "b": "x",
        "c": [2, "y", 3],
    }

    assert leaf_values(x, pred=lambda v: isinstance(v, int)) == [1, 2, 3]


# ---------------------------------------------------------------------------
# first_leaf
# ---------------------------------------------------------------------------

def test_first_leaf_basic():
    x = {
        "a": [],
        "b": ((), [Pair([], 7)]),
        "c": 8,
    }

    assert first_leaf(x) == 7


def test_first_leaf_with_predicate():
    x = {
        "a": "hello",
        "b": [3, 4.5, 6],
        "c": 9.5,
    }

    assert first_leaf(x, pred=lambda v: isinstance(v, float)) == 4.5


def test_first_leaf_raises_on_empty_tree():
    with pytest.raises(ValueError, match="Cannot get first leaf"):
        first_leaf([])

    with pytest.raises(ValueError, match="Cannot get first leaf"):
        first_leaf({})

    with pytest.raises(ValueError, match="Cannot get first leaf"):
        first_leaf(Pair([], ()))


def test_first_leaf_raises_when_no_leaf_matches_predicate():
    x = [1, 2, 3]

    with pytest.raises(ValueError, match="Cannot get first leaf"):
        first_leaf(x, pred=lambda v: isinstance(v, str))


# ---------------------------------------------------------------------------
# cycle_detect configuration
# ---------------------------------------------------------------------------

def test_cycle_detect_rejects_both_arg_pos_and_kwarg_name():
    with pytest.raises(ValueError, match="Specify only one"):
        cycle_detect(arg_pos=0, kwarg_name="x")


def test_cycle_detect_requires_arg_pos_or_kwarg_name():
    with pytest.raises(ValueError, match="Specify one"):
        cycle_detect(arg_pos=None, kwarg_name=None)


# ---------------------------------------------------------------------------
# cycle_detect behavior
# ---------------------------------------------------------------------------

def test_cycle_detect_allows_acyclic_recursive_walk():
    @cycle_detect(arg_pos=0)
    def walk(x):
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(v)
        return "ok"

    x = {"a": [1, 2], "b": (3, {"c": 4})}

    assert walk(x) == "ok"


def test_cycle_detect_raises_on_list_self_cycle():
    @cycle_detect(arg_pos=0)
    def walk(x):
        if isinstance(x, list):
            for v in x:
                walk(v)

    x = []
    x.append(x)

    with pytest.raises(CycleError):
        walk(x)


def test_cycle_detect_raises_on_dict_cycle():
    @cycle_detect(arg_pos=0)
    def walk(x):
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    x = {}
    x["self"] = x

    with pytest.raises(CycleError):
        walk(x)


def test_cycle_detect_does_not_confuse_shared_substructure_with_cycle():
    @cycle_detect(arg_pos=0)
    def walk(x):
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    shared = {"leaf": 1}
    x = [shared, shared]  # repeated reference, but no active-path cycle

    walk(x)  # should not raise


def test_cycle_detect_resets_state_after_cycle_error():
    @cycle_detect(arg_pos=0)
    def walk(x):
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    cyc = []
    cyc.append(cyc)

    with pytest.raises(CycleError):
        walk(cyc)

    acyc = [1, [2, 3], {"a": 4}]
    walk(acyc)  # should not raise after previous failure


def test_cycle_detect_supports_kwarg_name():
    @cycle_detect(arg_pos=None, kwarg_name="x")
    def walk(*, x):
        if isinstance(x, dict):
            for v in x.values():
                walk(x=v)
        elif isinstance(x, list):
            for v in x:
                walk(x=v)

    x = {}
    x["self"] = x

    with pytest.raises(CycleError):
        walk(x=x)


def test_cycle_detect_should_track_can_limit_what_is_tracked():
    @cycle_detect(arg_pos=0, should_track=lambda v: isinstance(v, list))
    def walk(x):
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    x = {"a": [1, 2], "b": {"c": [3, 4]}}

    walk(x)  # should not raise


def test_cycle_detect_should_track_only_lists_still_catches_list_cycles():
    @cycle_detect(arg_pos=0, should_track=lambda v: isinstance(v, list))
    def walk(x):
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    x = []
    x.append(x)

    with pytest.raises(CycleError):
        walk(x)


def test_cycle_detect_message_contains_useful_context():
    @cycle_detect(arg_pos=0)
    def walk(x):
        if isinstance(x, list):
            for v in x:
                walk(v)

    x = []
    x.append(x)

    with pytest.raises(CycleError) as excinfo:
        walk(x)

    msg = str(excinfo.value)
    assert "Val/type that tripped" in msg
    assert "oid:" in msg
    assert "path_oids:" in msg
