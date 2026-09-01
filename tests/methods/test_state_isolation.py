"""Contract tests for process-local Method preparation state isolation."""

import gc
import os
import weakref
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from dryml.core.object import Pickleable
from dryml.core.repo import Repo
from dryml.core.store.dir import DirStore
from dryml.methods import Method, traits


class Stateful(Method):
    """Minimal alternative-backed fixture for weak side-state tests."""

    @traits(backend="numpy")
    def numpy(self, value):
        """Return the selected NumPy value unchanged."""

        return value


class StructurallyEqualStateful(Stateful):
    """Fixture that intentionally compares equal across distinct live instances."""

    def __eq__(self, other):
        """Treat all instances of this fixture as structurally equal."""

        return isinstance(other, StructurallyEqualStateful)

    def __hash__(self):
        """Use one structural hash to exercise non-equality-keyed state storage."""

        return 1


class PersistentStateful(Method, Pickleable):
    """Pickleable Method fixture proving preparation state stays external."""

    def __init__(self, marker="value"):
        """Store one ordinary heavy-state value."""

        self.marker = marker

    @traits(backend="numpy")
    def numpy(self, value):
        """Return the selected value unchanged."""

        return value


def test_structurally_equal_instances_have_independent_defaults_and_cached_state():
    """Identity-keyed side state never follows Object structural equality."""

    first = StructurallyEqualStateful()
    second = StructurallyEqualStateful()
    assert first == second
    first.default_batched = True
    first.learn()
    first(np.ones((2,), dtype=np.float32))

    assert first.call_mode == "cached"
    assert first.default_batched is True
    assert second.call_mode == "eager"
    assert second.default_batched is None


def test_preparation_never_changes_definition_hash_or_instance_storage():
    """Defaults and caches remain outside logical identity and Object payload state."""

    method = Stateful()
    definition = method.definition
    stable_hash = definition.stable_hash()
    object_keys = set(method.__dict__)

    method.default_batched = True
    method.learn()
    method(np.ones((2,), dtype=np.float32))

    assert method.definition == definition
    assert method.definition.stable_hash() == stable_hash
    assert set(method.__dict__) == object_keys


def test_pickleable_save_load_resets_fresh_state_and_preserves_live_reuse(tmp_path):
    """Saving excludes preparation while explicit reuse retains only the same live object."""

    repo = Repo(DirStore(tmp_path / "store"))
    method = PersistentStateful(repo=repo)
    method.default_batched = True
    method.learn()
    method(np.ones((2,), dtype=np.float32))

    state = method.save(repo=repo)
    assert method.call_mode == "cached"
    assert method.default_batched is True

    reused = repo.load_state_ref(state, reuse_live="matching")
    assert reused is method
    assert reused.call_mode == "cached"
    assert reused.default_batched is True

    fresh = repo.load_state_ref(state, reuse_live="never")
    assert fresh is not method
    assert fresh.marker == "value"
    assert fresh.call_mode == "eager"
    assert fresh.default_batched is None


def test_cached_side_state_does_not_keep_method_instances_alive():
    """The weak table and unbound cached invocation release dead Method keys."""

    method = Stateful()
    method.learn()
    method(np.ones((2,), dtype=np.float32))
    reference = weakref.ref(method)
    del method
    gc.collect()

    assert reference() is None


def test_concurrent_cached_reads_and_unrelated_transitions_remain_isolated():
    """Cached reads do not mutate state while other Method identities transition."""

    cached = Stateful()
    cached.learn()
    cached(np.ones((2,), dtype=np.float32))
    signature = cached.cached_signature

    def read_cached(index):
        value = np.full((2,), index, dtype=np.float32)
        assert cached(value) is value
        return cached.cached_signature

    def transition_unrelated(index):
        method = Stateful()
        method.default_batched = bool(index % 2)
        method.learn()
        method(np.ones((2,), dtype=np.float32))
        method.eager()
        return method.call_mode, method.default_batched

    with ThreadPoolExecutor(max_workers=8) as executor:
        read_results = tuple(executor.map(read_cached, range(32)))
        transition_results = tuple(executor.map(transition_unrelated, range(32)))

    assert all(result == signature for result in read_results)
    assert cached.call_mode == "cached"
    assert cached.cached_signature == signature
    assert all(mode == "eager" and default is bool(index % 2)
               for index, (mode, default) in enumerate(transition_results))


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork support")
def test_forked_child_replaces_inherited_state_even_while_parent_lock_is_held():
    """The at-fork child callback resets state without acquiring a parent-owned lock."""

    from dryml.methods import method as method_module

    method = Stateful()
    method.default_batched = True
    method.learn()
    method(np.ones((2,), dtype=np.float32))
    read_fd, write_fd = os.pipe()
    method_module._STATE_LOCK.acquire()
    try:
        child = os.fork()
        if child == 0:
            try:
                value = f"{method.call_mode}:{method.default_batched}".encode()
                os.write(write_fd, value)
            finally:
                os._exit(0)
    finally:
        method_module._STATE_LOCK.release()
        os.close(write_fd)
    try:
        observed = os.read(read_fd, 32).decode()
        _, status = os.waitpid(child, 0)
    finally:
        os.close(read_fd)

    assert status == 0
    assert observed == "eager:None"
    assert method.call_mode == "cached"
    assert method.default_batched is True
