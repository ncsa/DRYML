from __future__ import annotations

import pytest

from dryml.core.canonical import (
    is_canonical_key,
    iter_value_children,
    to_canonical,
    thaw_value,
    from_canonical,
    transform_container,
)
from dryml.core.freeze import FrozenDict, FrozenList, FrozenTuple
from dryml.core.repo import Repo


def test_is_canonical_key_accepts_exact_str_and_int_only():
    assert is_canonical_key("x")
    assert is_canonical_key("")
    assert is_canonical_key(0)
    assert is_canonical_key(123)

    assert not is_canonical_key(True)
    assert not is_canonical_key(False)
    assert not is_canonical_key(3.14)
    assert not is_canonical_key(b"x")
    assert not is_canonical_key((1, 2))


def test_iter_value_children_accepts_valid_dict_keys():
    x = {"a": 1, 2: 3}

    children = list(iter_value_children(x))
    assert children == [("a", 1), (2, 3)]


def test_iter_value_children_rejects_bool_key():
    x = {True: 1}

    with pytest.raises(TypeError, match="Only str and int keys are allowed"):
        list(iter_value_children(x))


def test_iter_value_children_rejects_float_key():
    x = {3.14: 1}

    with pytest.raises(TypeError, match="Only str and int keys are allowed"):
        list(iter_value_children(x))


def test_to_canonical_preserves_valid_keys_and_canonicalizes_values():
    x = {
        "a": [1, 2],
        3: ("x", "y"),
    }

    out = to_canonical(x)

    assert isinstance(out, FrozenDict)
    assert set(out.keys()) == {"a", 3}

    assert isinstance(out["a"], FrozenList)
    assert list(out["a"]) == [1, 2]

    assert isinstance(out[3], FrozenTuple)
    assert tuple(out[3]) == ("x", "y")


def test_to_canonical_rejects_bool_key():
    x = {True: [1, 2, 3]}

    with pytest.raises(TypeError, match="Only str and int keys are allowed"):
        to_canonical(x)


def test_to_canonical_rejects_float_key():
    x = {3.14: [1, 2, 3]}

    with pytest.raises(TypeError, match="Only str and int keys are allowed"):
        to_canonical(x)


def test_thaw_value_preserves_valid_keys():
    x = FrozenDict({
        "a": FrozenList([1, 2]),
        5: FrozenTuple(("x", "y")),
    })

    out = thaw_value(x)

    assert type(out) is dict
    assert set(out.keys()) == {"a", 5}
    assert out["a"] == [1, 2]
    assert out[5] == ("x", "y")


def test_thaw_value_rejects_bool_key():
    x = FrozenDict({
        True: FrozenList([1, 2]),
    })

    with pytest.raises(TypeError, match="Only str and int keys are allowed"):
        thaw_value(x)


def test_thaw_value_rejects_float_key():
    x = FrozenDict({
        2.5: FrozenList([1, 2]),
    })

    with pytest.raises(TypeError, match="Only str and int keys are allowed"):
        thaw_value(x)


def test_from_canonical_preserves_keys_and_transforms_only_values():
    repo = Repo()

    x = FrozenDict({
        "a": FrozenList([1, 2]),
        7: FrozenTuple(("x", "y")),
    })

    out = from_canonical(x, repo=repo)

    assert type(out) is dict
    assert set(out.keys()) == {"a", 7}
    assert out["a"] == [1, 2]
    assert out[7] == ("x", "y")


def test_from_canonical_does_not_transform_int_key():
    repo = Repo()

    x = FrozenDict({
        1: FrozenList([10]),
    })

    out = from_canonical(x, repo=repo)

    assert 1 in out
    assert 2 not in out
    assert out[1] == [10]


def test_from_canonical_rejects_bool_key():
    repo = Repo()

    x = FrozenDict({
        True: FrozenList([10]),
    })

    with pytest.raises(TypeError, match="Only str and int keys are allowed"):
        from_canonical(x, repo=repo)


def test_from_canonical_rejects_float_key():
    repo = Repo()

    x = FrozenDict({
        1.5: FrozenList([10]),
    })

    with pytest.raises(TypeError, match="Only str and int keys are allowed"):
        from_canonical(x, repo=repo)


def test_transform_container_preserves_dict_keys_and_transforms_only_values_same():
    x = {"a": 1, 2: 3}

    out = transform_container(
        x,
        lambda p, v: v * 10,
        target="same",
    )

    assert out == {"a": 10, 2: 30}


def test_transform_container_preserves_dict_keys_and_transforms_only_values_canonical():
    x = {"a": [1, 2], 2: (3, 4)}

    out = transform_container(
        x,
        lambda p, v: to_canonical(v),
        target="canonical",
    )

    assert isinstance(out, FrozenDict)
    assert set(out.keys()) == {"a", 2}
    assert isinstance(out["a"], FrozenList)
    assert isinstance(out[2], FrozenTuple)


def test_transform_container_rejects_invalid_dict_key():
    x = {True: 1}

    with pytest.raises(TypeError, match="Only str and int keys are allowed"):
        transform_container(
            x,
            lambda p, v: v,
            target="same",
        )
