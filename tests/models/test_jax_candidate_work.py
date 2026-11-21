"""Backend-free work and failure boundaries for JAX candidate placement."""

from types import SimpleNamespace

import pytest

from dryml.models.jax.base import Training


_PLACEMENT = SimpleNamespace(device_set=frozenset({"cpu:0"}))


class PlacementArray:
    """Array double with observable placement metadata reads and stable sharding."""

    def __init__(self, committed=False):
        self.committed = committed
        self.sharding = _PLACEMENT


def placement_backend(*, fail=False):
    """Return a flat-argument JAX double recording placement and native calls."""
    puts, calls = [], []

    def device_put(value, sharding):
        puts.append((value, sharding))
        if fail:
            raise RuntimeError("placement failed")
        result = PlacementArray(committed=True)
        result.sharding = sharding
        return result

    def jit(transition):
        def native(*args):
            calls.append(args)
            return args
        return native

    backend = SimpleNamespace(
        device_put=device_put, jit=jit,
        tree_util=SimpleNamespace(
            tree_flatten=lambda values: (values, None),
            tree_unflatten=lambda tree, values: tuple(values),
        ),
    )
    return backend, puts, calls


def test_native_placement_is_local_preserves_aliases_and_skips_committed_arrays():
    """Only uncommitted references are placed, once per distinct leaf per call."""
    jax, puts, calls = placement_backend()
    update = Training._native_update(jax, None, None, None, None)
    assert puts == calls == []  # Construction must not read native owners.
    shared, ready = PlacementArray(), PlacementArray(committed=True)
    result = update(shared, shared, ready, ready, ready, ready)
    assert puts == [(shared, shared.sharding)]
    assert result[0] is result[1] and result[0] is not shared
    assert all(value is ready for value in result[2:])
    assert not shared.committed
    assert update(*result) == result and len(puts) == 1
    # A later caller may supply a new uncommitted leaf, even with identical shape.
    replacement = PlacementArray()
    update(replacement, replacement, *result[2:])
    assert puts[-1] == (replacement, replacement.sharding) and len(puts) == 2
    assert len(calls) == 3


def test_native_placement_failure_precedes_pure_computation():
    """Placement failure propagates without executing or installing a candidate."""
    jax, puts, calls = placement_backend(fail=True)
    update = Training._native_update(jax, None, None, None, None)
    with pytest.raises(RuntimeError, match="placement failed"):
        update(*(PlacementArray() for _ in range(5)), PlacementArray(committed=True))
    assert len(puts) == 1 and calls == []


@pytest.mark.parametrize("evidence", ("uncommitted", "no_sharding"))
def test_native_placement_requires_committed_anchor(evidence):
    """Without a committed anchor and placement, pass original arguments to JIT."""
    jax, puts, calls = placement_backend(fail=True)
    update = Training._native_update(jax, None, None, None, None)
    shared = PlacementArray(committed=evidence == "no_sharding")
    if evidence == "no_sharding":
        shared.sharding = None
    args = (shared,) * 6
    result = update(*args)
    assert all(actual is original for actual, original in zip(result, args))
    assert puts == [] and calls == [args]


@pytest.mark.parametrize("devices", (("cpu:1",), ("cpu:0", "cpu:1")))
def test_native_placement_keeps_mixed_and_multidevice_inference(devices):
    """Explicit placement must not constrain ordinary heterogeneous JAX inference."""
    jax, puts, calls = placement_backend()
    update = Training._native_update(jax, None, None, None, None)
    first, other = PlacementArray(committed=True), PlacementArray()
    other.sharding = SimpleNamespace(device_set=frozenset(devices))
    if len(devices) > 1:
        first.sharding = other.sharding
    args = (first, other, first, other, first, other)
    assert update(*args) == args
    assert puts == [] and calls == [args]
