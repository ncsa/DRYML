"""U14 detached admission, independent locks and fenced authority witnesses."""

import asyncio
from contextlib import contextmanager
from dataclasses import replace
import json
import multiprocessing
import os
from threading import Event, Thread
from uuid import uuid4

import pytest

from dryml.core import Definition, Object, Repo, Serializable
from dryml.core.reference_values import ObjectId
from dryml.core.store.dir import DirStore
from dryml.managed import managed_operation
from dryml.managed import identity as identities
from dryml.managed.control import ControlSnapshot, ManagedControlStore
from dryml.managed.errors import ManagedConflictError, ManagedControlError
from dryml.managed.results import ResultReference


class Dataset(Serializable):
    def __init__(self, value=1):
        self.value = value


class Experiment(Object):
    def __init__(self, dataset):
        self.dataset = dataset

    @managed_operation(resumable=True)
    def run(self, epochs=1, *, managed):
        raise AssertionError("foundation must never execute workload")


def facts(receiver, epochs=1):
    return identities.AdmissionIdentity.from_call(receiver, Experiment.run, (), {"epochs": epochs})


def receiver(dataset=None):
    dataset = dataset or Definition(Dataset).concretize().with_identity(ObjectId())
    return Definition(Experiment, dataset).concretize()


def adapter(tmp_path, *, index="none"):
    store = DirStore(tmp_path / "control", query_index=index)
    return ManagedControlStore(store, Repo(store))


def running(identity, admission, *, generation=1):
    return ControlSnapshot(
        receiver=identity.receiver, member=identity.member,
        argument_digest=identity.argument_digest, run_id=admission.run_id,
        invocation_id=admission.invocation_id, generation=generation, state="running",
    )


def test_detached_identity_never_constructs_or_opens_payload(tmp_path, monkeypatch):
    node = receiver()
    def forbidden(*args, **kwargs):
        pytest.fail("metadata admission attempted realization or private IO")
    monkeypatch.setattr(Experiment, "__init__", forbidden)
    monkeypatch.setattr(Dataset, "__init__", forbidden)
    monkeypatch.setattr(DirStore, "read_state", forbidden)
    identity = facts(node)
    control = adapter(tmp_path)
    with control.admission(identity) as admitted:
        assert admitted.current is None
        assert admitted.active
        assert admitted.identity.operation_id == identities.operation_digest(node, "run")
        assert facts(node, 2).operation_id == identity.operation_id
        assert facts(node, 2).argument_digest != identity.argument_digest
        assert admitted.realize(lambda metadata: metadata) is node
    assert not admitted.active


def test_duplicate_immediate_and_distinct_readonly_live_dataset_allowed(tmp_path):
    control = adapter(tmp_path)
    repo = control.state_repo
    shared = Dataset(repo=repo)
    first, second = Experiment(shared, repo=repo), Experiment(shared, repo=repo)
    # The operation roots must differ even with an otherwise identical graph.
    first_node = Definition(Experiment, shared.definition.with_identity(shared.object_id)).concretize().with_identity(ObjectId())
    second_node = first_node.with_realization(identities={"$": ObjectId()})
    with control.admission(facts(first_node)):
        with pytest.raises(ManagedConflictError, match="already_running"):
            with control.admission(facts(first_node)):
                pytest.fail("duplicate admitted")
        with control.admission(facts(second_node)):
            assert first.dataset is second.dataset is shared
        assert not __import__("dryml.core.state", fromlist=["is_reserved"]).is_reserved(shared)


def test_tasks_and_threads_cannot_release_foreign_operation_owner(tmp_path):
    control, identity = adapter(tmp_path), facts(receiver())
    with control.admission(identity) as admitted:
        outcomes = []
        def foreign():
            outcomes.append(admitted.release())
            with pytest.raises(ManagedConflictError):
                admitted.require_owner()
            with pytest.raises(ManagedConflictError, match="already_running"):
                with control.admission(identity):
                    pass
        thread = Thread(target=foreign)
        thread.start(); thread.join(10)
        assert not thread.is_alive() and outcomes == [False]
        async def other_task():
            assert not admitted.release()
        asyncio.run(other_task())
        assert admitted.active


def test_run_stable_invocation_fresh_even_metadata_replay(tmp_path):
    control, identity = adapter(tmp_path), facts(receiver())
    with control.admission(identity) as first:
        initial = control.create_initial(running(identity, first), owner=first)
    with control.admission(identity) as resumed:
        assert resumed.run_id == first.run_id
        assert resumed.invocation_id != first.invocation_id
        assert resumed.current == initial
    with control.admission(identity) as replay:
        assert replay.invocation_id not in {first.invocation_id, resumed.invocation_id}
        assert replay.run_id == first.run_id
    with pytest.raises(ManagedConflictError, match="argument_mismatch"):
        with control.admission(facts(identity.receiver, 2)):
            pass


def test_stale_generation_and_ids_fence_all_owner_mutations(tmp_path):
    control, identity = adapter(tmp_path), facts(receiver())
    with control.admission(identity) as owner:
        initial = control.create_initial(running(identity, owner), owner=owner)
        for run_id, invocation_id, generation in (
            (uuid4().hex, owner.invocation_id, 1),
            (owner.run_id, uuid4().hex, 1),
            (owner.run_id, owner.invocation_id, 2),
        ):
            with pytest.raises(ManagedConflictError):
                control.transition_running_owner(
                    identity.operation_id, owner=owner, run_id=run_id,
                    invocation_id=invocation_id, expected_generation=generation,
                    build=lambda current: pytest.fail("stale authority reached owner builder"),
                )
        assert control.inspect(identity.operation_id) == initial


def test_short_control_status_interrupt_responsive_and_no_inverse_locks(tmp_path, monkeypatch):
    control, identity = adapter(tmp_path), facts(receiver())
    held = False
    original_control = control._control_lock
    original_writer = control.control_store.writer_lock
    @contextmanager
    def short_control():
        nonlocal held
        with original_control():
            held = True
            try: yield
            finally: held = False
    @contextmanager
    def writer():
        assert not held, "Store fence acquired under control"
        with original_writer(): yield
    monkeypatch.setattr(control, "_control_lock", short_control)
    monkeypatch.setattr(control.control_store, "writer_lock", writer)
    original_probe = control.probe_operation_owner
    def probe(operation_id):
        assert not held, "operation probe acquired under short control"
        return original_probe(operation_id)
    monkeypatch.setattr(control, "probe_operation_owner", probe)
    from dryml.core.node_prepare import NodePreparation
    original_prepare = NodePreparation.__init__
    def prepare(self, *args, **kwargs):
        assert not held, "graph preparation ran under short control"
        return original_prepare(self, *args, **kwargs)
    monkeypatch.setattr(NodePreparation, "__init__", prepare)
    with control.admission(identity) as owner:
        initial = control.create_initial(running(identity, owner), owner=owner)
        done, errors = Event(), []
        def inspect_and_interrupt():
            try:
                assert control.status(identity.operation_id).state == "running"
                outcome, updated = control.request_interrupt_if_running(
                    identity.operation_id, expected_run_id=owner.run_id,
                    expected_invocation_id=owner.invocation_id, expected_generation=1)
                assert outcome == "requested" and updated.generation == 2
            except BaseException as error:
                errors.append(error)
            finally: done.set()
        thread = Thread(target=inspect_and_interrupt)
        thread.start()
        assert done.wait(10), "short control waited for lifetime operation owner"
        thread.join(10)
        assert errors == []
        with pytest.raises(ManagedConflictError):
            control.transition_running_owner(identity.operation_id, owner=owner,
                run_id=owner.run_id, invocation_id=owner.invocation_id,
                expected_generation=initial.generation,
                build=lambda current: replace(current, generation=2))


def test_fresh_checkpoint_and_completion_cas_preserve_producer(tmp_path):
    from tests.managed.test_control_codec import saved_receiver
    control = adapter(tmp_path)
    node = saved_receiver(control)
    identity = facts(node)
    with control.admission(identity) as owner:
        control.create_initial(running(identity, owner), owner=owner)
        checkpoint = control.transition_running_owner(identity.operation_id, owner=owner,
            run_id=owner.run_id, invocation_id=owner.invocation_id, expected_generation=1,
            build=lambda current: replace(current, generation=2, checkpoint=node))
        complete = control.transition_running_owner(identity.operation_id, owner=owner,
            run_id=owner.run_id, invocation_id=owner.invocation_id, expected_generation=checkpoint.generation,
            build=lambda current: replace(current, generation=3, state="completed", final=node, result_reference=ResultReference("f" * 64),
                producer_invocation_id=owner.invocation_id))
        assert control.status(identity.operation_id).producer_invocation_id == owner.invocation_id
        assert complete.checkpoint.graph_equal(node)


def test_owner_loss_probe_before_control_and_unchanged_fence(tmp_path, monkeypatch):
    control, identity = adapter(tmp_path), facts(receiver())
    with control.admission(identity) as owner:
        initial = control.create_initial(running(identity, owner), owner=owner)
        assert control.observe_running_owner(identity.operation_id) == (initial, False, True)
    assert control.observe_running_owner(identity.operation_id) == (initial, True, True)
    original_probe = control.probe_operation_owner
    def changed_probe(operation_id):
        free = original_probe(operation_id)
        replacement = replace(initial, generation=2, invocation_id=uuid4().hex)
        control._replace_file(control._current_path(control._operation_path(operation_id)), replacement.to_bytes())
        return free
    monkeypatch.setattr(control, "probe_operation_owner", changed_probe)
    observed, lost, stable = control.observe_running_owner(identity.operation_id)
    assert observed.generation == 2 and not lost and not stable


@pytest.mark.parametrize("target,version", [("area", 1), ("current", 2), ("pending", 1)])
def test_incompatible_versions_preserve_source_bytes(tmp_path, target, version):
    control, identity = adapter(tmp_path), facts(receiver())
    with control.admission(identity) as owner:
        initial = control.create_initial(running(identity, owner), owner=owner)
    operation = control._operation_path(identity.operation_id)
    path = {"area": os.path.join(control.root, "format.json"),
            "current": control._current_path(operation), "pending": control._pending_path(operation)}[target]
    if target == "pending":
        payload = json.dumps({"schema": "dryml-managed-pending", "version": version}).encode()
    else:
        with open(path, "rb") as stream: data = json.load(stream)
        data["version"] = version
        payload = json.dumps(data).encode()
    with open(path, "wb") as stream: stream.write(payload)
    with pytest.raises(ManagedControlError): control.inspect(identity.operation_id)
    with open(path, "rb") as stream: assert stream.read() == payload


def _spawn_owner(path, identity, ready, release):
    store = DirStore(path, query_index="none")
    control = ManagedControlStore(store, Repo(store))
    with control.admission(identity):
        ready.set()
        assert release.wait(15)
        os._exit(0)


def test_spawn_duplicate_distinct_and_owner_exit(tmp_path):
    control, identity = adapter(tmp_path), facts(receiver())
    context = multiprocessing.get_context("spawn")
    ready, release = context.Event(), context.Event()
    process = context.Process(target=_spawn_owner, args=(control.control_store.base_dir, identity, ready, release))
    process.start()
    try:
        assert ready.wait(15)
        with pytest.raises(ManagedConflictError, match="already_running"):
            with control.admission(identity): pass
        with control.admission(facts(identity.receiver.with_identity(ObjectId()))): pass
        release.set(); process.join(15)
        assert process.exitcode == 0
        with control.admission(identity): pass
    finally:
        release.set()
        if process.is_alive(): process.terminate(); process.join(5)


def test_control_journal_invalidation_precedes_cas_and_projection_follows(tmp_path, monkeypatch):
    control, identity = adapter(tmp_path, index="sqlite"), facts(receiver())
    events = []
    original = control._replace_file
    def publish(path, payload):
        if path.endswith("current.json"):
            from pathlib import Path
            from dryml.core.store.records import AuthorityRecord
            cut = AuthorityRecord.from_bytes(Path(control.control_store.base_dir, "authority.record").read_bytes())
            assert cut.pending
            events.append("cas")
        return original(path, payload)
    monkeypatch.setattr(control, "_replace_file", publish)
    with control.admission(identity) as owner:
        control.create_initial(running(identity, owner), owner=owner)
    assert events == ["cas"]
    assert control.control_store.authority_cut().pending == ()
    index = control.control_store.open_query_index()
    assert any(identity.operation_id in row[0] for row in index.projection_rows("record"))


def test_distinct_tasks_share_dataset_without_inherited_owner_authority(tmp_path):
    control = adapter(tmp_path)
    first = facts(receiver())
    second = facts(first.receiver.with_identity(ObjectId()))
    async def run():
        held, release = asyncio.Event(), asyncio.Event()
        owners = []
        async def a():
            with control.admission(first) as owner:
                owners.append(owner)
                held.set()
                await release.wait()
        async def b():
            await held.wait()
            assert not owners[0].release()
            with pytest.raises(ManagedConflictError, match="already_running"):
                control.admission(first)
            with control.admission(second) as owner:
                assert owner.active
            release.set()
        await asyncio.gather(a(), b())
    asyncio.run(run())


def test_query_coverage_refuses_gap_and_interrupt_can_win_before_projection_ack(tmp_path, monkeypatch):
    from dryml.core.query.model import QueryIndexDirty
    control, identity = adapter(tmp_path, index="sqlite"), facts(receiver())
    index = control.control_store.open_query_index()
    committed, release, outcomes = Event(), Event(), []
    original = control._finish_projection
    paused = []
    def pause_once(*args):
        if not paused:
            paused.append(True)
            committed.set()
            assert release.wait(10)
        return original(*args)
    monkeypatch.setattr(control, "_finish_projection", pause_once)
    def publish():
        try:
            with control.admission(identity) as owner:
                control.create_initial(running(identity, owner), owner=owner)
                outcomes.append("published")
        except BaseException as error: outcomes.append(error)
    worker = Thread(target=publish); worker.start()
    try:
        assert committed.wait(10)
        with pytest.raises(QueryIndexDirty): index.projection_rows("record")
        assert control.status(identity.operation_id).state == "running"
        outcome, interrupted = control.request_interrupt_if_running(identity.operation_id)
        assert outcome == "requested" and interrupted.generation == 2
    finally:
        release.set(); worker.join(10)
    assert not worker.is_alive() and outcomes == ["published"]
    assert not control.control_store.authority_cut().pending
