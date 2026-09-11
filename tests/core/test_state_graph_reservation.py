from pathlib import Path
import os
from threading import Event, Thread

import pytest

from dryml.core import Object, Repo, Serializable
from dryml.core.repo import RepoSaveError
from dryml.core.store.dir import DirStore


class ReservedState(Serializable):
    def __init__(self, value):
        self.value = value

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        Path(dest_dir, "value").write_text(str(self.value), encoding="ascii")

    def restore_state_from_dir_imp(self, src_dir, *, codec):
        self.value = int(Path(src_dir, "value").read_text(encoding="ascii"))


class StatelessRoot(Object):
    def __init__(self, child):
        self.child = child


class CountingReservedState(ReservedState):
    captures = 0

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        type(self).captures += 1
        super().save_state_to_dir_imp(dest_dir, codec=codec)


def test_graph_reservation_covers_stateful_descendants_and_rejects_nested_owner(tmp_path):
    repo = Repo(DirStore(tmp_path / "store"))
    child = ReservedState(1, repo=repo)
    root = StatelessRoot(child, repo=repo)

    with repo.reserve_state_graph(root) as reservation:
        assert reservation.active
        assert reservation.object_ref == root.object_ref
        assert reservation.object_ids == (child.object_id,)
        with pytest.raises(RepoSaveError, match="reserved"):
            repo.reserve_state_graph(root)

    assert not reservation.active
    with repo.reserve_state_graph(root):
        pass

    with pytest.raises(RepoSaveError, match="inactive"):
        repo.save_object(root, reservation=reservation)


def test_graph_reservation_allows_its_exact_save_reuse(tmp_path):
    repo = Repo(DirStore(tmp_path / "store"))
    obj = ReservedState(3, repo=repo)

    with repo.reserve_state_graph(obj) as reservation:
        state = repo.save_object(obj, reservation=reservation)

    assert state.object == obj.object_ref


def test_failed_save_on_closed_repo_releases_owned_graph_reservation(tmp_path):
    """A failed context admission does not retain the save's graph reservation."""
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    obj = ReservedState(3, repo=repo)
    repo.close(flush=False)

    with pytest.raises(RuntimeError, match="Cannot retain a save context"):
        repo.save_object(obj)

    with Repo(store).reserve_state_graph(obj):
        pass


def test_graph_reservation_reuses_its_route_neutral_evidence_for_save(tmp_path, monkeypatch):
    """An admitted save does not rebuild graph bindings after reservation."""
    repo = Repo(DirStore(tmp_path / "store"))
    obj = ReservedState(3, repo=repo)
    calls = 0
    original = repo._state_graph_evidence

    def count_evidence(value):
        nonlocal calls
        calls += 1
        return original(value)

    monkeypatch.setattr(repo, "_state_graph_evidence", count_evidence)
    with repo.reserve_state_graph(obj) as reservation:
        repo.save_object(obj, reservation=reservation)

    assert calls == 1


def test_graph_reservation_revalidates_invalidated_retained_nodes_before_save_hooks(tmp_path):
    """Reservation reuse cannot authorize a graph invalidated after admission."""
    repo = Repo(DirStore(tmp_path / "store"))
    child = CountingReservedState(3, repo=repo)
    root = StatelessRoot(child, repo=repo)
    CountingReservedState.captures = 0

    with repo.reserve_state_graph(root) as reservation:
        root._restore_failed = True
        with pytest.raises(RepoSaveError, match="invalidated"):
            repo.save_object(root, reservation=reservation, deep_capture=True)

    assert CountingReservedState.captures == 0


def test_graph_reservation_rejects_a_token_for_a_different_live_graph(tmp_path):
    repo = Repo(DirStore(tmp_path / "store"))
    first = ReservedState(1, repo=repo)
    second = ReservedState(2, repo=repo)

    with repo.reserve_state_graph(first) as reservation:
        with pytest.raises(RepoSaveError, match="does not cover"):
            repo.save_object(second, reservation=reservation)


def test_disjoint_graph_save_can_proceed_while_another_graph_is_reserved(tmp_path):
    repo = Repo(DirStore(tmp_path / "store"))
    first = ReservedState(1, repo=repo)
    second = ReservedState(2, repo=repo)

    with repo.reserve_state_graph(first):
        state = repo.save_object(second)

    assert state.object == second.object_ref


def test_graph_reservation_rejects_a_sibling_thread_before_hooks(tmp_path):
    repo = Repo(DirStore(tmp_path / "store"))
    obj = ReservedState(3, repo=repo)
    acquired = Event()
    release = Event()

    def hold_reservation():
        with repo.reserve_state_graph(obj):
            acquired.set()
            assert release.wait(timeout=5)

    thread = Thread(target=hold_reservation)
    thread.start()
    assert acquired.wait(timeout=5)
    try:
        with pytest.raises(RepoSaveError, match="already reserved"):
            repo.reserve_state_graph(obj)
    finally:
        release.set()
        thread.join(timeout=5)
    assert not thread.is_alive()


def test_graph_reservation_orders_object_ids_by_canonical_identity(tmp_path):
    """Reservation order is a full canonical ObjectId order, not display truncation."""

    repo = Repo(DirStore(tmp_path / "store"))
    root = StatelessRoot(ReservedState(3, repo=repo), repo=repo)

    with repo.reserve_state_graph(root) as reservation:
        assert reservation.object_ids == tuple(
            sorted(reservation.object_ids, key=lambda object_id: object_id.__stable_leaf_bytes__())
        )


@pytest.mark.parametrize("reuse_live", ["matching", "greedy"])
def test_reserved_logical_object_is_not_reused_by_a_sibling_thread(tmp_path, reuse_live):
    repo = Repo(DirStore(tmp_path / "store"))
    obj = ReservedState(3, repo=repo)
    state = repo.save_object(obj)
    acquired = Event()
    release = Event()

    def hold_reservation():
        with repo.reserve_state_graph(obj):
            acquired.set()
            assert release.wait(timeout=5)

    thread = Thread(target=hold_reservation)
    thread.start()
    assert acquired.wait(timeout=5)
    try:
        loaded = repo.load_state_ref(state, reuse_live=reuse_live)
    finally:
        release.set()
        thread.join(timeout=5)

    assert loaded is not obj
    assert loaded.object_id == obj.object_id


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
def test_forked_graph_reservation_is_not_inherited(tmp_path):
    repo = Repo(DirStore(tmp_path / "store"))
    obj = ReservedState(3, repo=repo)
    read_fd, write_fd = os.pipe()
    with repo.reserve_state_graph(obj) as reservation:
        child = os.fork()
        if child == 0:
            try:
                assert not reservation.active
                with repo.reserve_state_graph(obj):
                    pass
                os.write(write_fd, b"1")
            finally:
                os._exit(0)
        os.close(write_fd)
        assert os.read(read_fd, 1) == b"1"
        _, status = os.waitpid(child, 0)
        assert os.waitstatus_to_exitcode(status) == 0
    os.close(read_fd)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
def test_fork_resets_registry_when_another_thread_held_the_parent_lock(tmp_path):
    from dryml.core import state as state_module

    repo = Repo(DirStore(tmp_path / "store"))
    obj = ReservedState(3, repo=repo)
    entered = Event()
    release = Event()

    def hold_registry_lock():
        with state_module._REGISTRY_LOCK:
            entered.set()
            assert release.wait(timeout=5)

    thread = Thread(target=hold_registry_lock)
    thread.start()
    assert entered.wait(timeout=5)
    read_fd, write_fd = os.pipe()
    try:
        child = os.fork()
        if child == 0:
            try:
                with repo.reserve_state_graph(obj):
                    pass
                os.write(write_fd, b"1")
            finally:
                os._exit(0)
        os.close(write_fd)
        release.set()
        assert os.read(read_fd, 1) == b"1"
        _, status = os.waitpid(child, 0)
        assert os.waitstatus_to_exitcode(status) == 0
    finally:
        release.set()
        thread.join(timeout=5)
        os.close(read_fd)
        if 'child' in locals():
            try:
                os.waitpid(child, os.WNOHANG)
            except ChildProcessError:
                pass
