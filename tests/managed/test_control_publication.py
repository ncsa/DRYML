"""Current-format control publication, exact reconciliation and owner fencing.

Retired descendant-lock/StateRef closure fixtures are replaced by detached Node
authority. Checkpoint/final/result publication qualification is owned by U16/U17;
the still-unported owning scenarios remain in test_failure_windows.py.
"""

from contextlib import contextmanager
from dataclasses import replace
import multiprocessing
import os
from pathlib import Path
from threading import Event, Thread
from uuid import uuid4

import pytest

from dryml.core import Repo
from dryml.core.store.dir import DirStore
from dryml.core.store.authority import StoreMutationError
from dryml.managed.control import ControlHistory, ControlIntent, ManagedControlStore, StartPreparation, _PendingIntent
from dryml.managed.errors import ManagedConflictError, ManagedControlError
from tests.managed.test_u14_authority import adapter, facts, receiver, running
from tests.managed.test_control_codec import force_fresh_handle_order, unordered_saved_receiver


def transition(control, owner, old, proposed):
    return control.transition(old.operation_id, proposed, owner=owner,
        expected_generation=old.generation, expected_run_id=old.run_id,
        expected_invocation_id=old.invocation_id)


def test_publisher_never_waits_for_short_control_while_holding_store_writer(tmp_path, monkeypatch):
    """A contended short CAS refuses; its cancelled cut is recovered after fences end."""
    from dryml.core.store.retention import _writers
    control, identity = adapter(tmp_path), facts(receiver())
    registered, held, release = Event(), Event(), Event()
    store = control.control_store
    with control.admission(identity) as owner:
        initial = control.create_initial(running(identity, owner), owner=owner)
        original = store._atomic_write
        injected = []
        def write(path, payload):
            result = original(path, payload)
            if str(path).endswith("authority.record") and not injected:
                injected.append(True)
                registered.set()
                assert held.wait(10)
            return result
        def block_control():
            assert registered.wait(10)
            with control._control_lock():
                held.set()
                assert release.wait(15)
        worker = Thread(target=block_control)
        worker.start()
        monkeypatch.setattr(store, "_atomic_write", write)
        try:
            with pytest.raises(StoreMutationError) as raised:
                transition(control, owner, initial, replace(initial, generation=2))
            assert isinstance(raised.value.__cause__, ManagedConflictError)
            assert raised.value.__cause__.reason == "control_busy"
            assert not _writers() and not release.is_set()
            # The caller has already released its Store fence while the other
            # caller still holds control; no timed delay establishes this proof.
            with store.writer_lock():
                assert _writers()
        finally:
            release.set(); worker.join(15)
        assert not worker.is_alive()
        monkeypatch.setattr(store, "_atomic_write", original)
        assert control.reconcile(identity.operation_id, owner=owner).to_bytes() == initial.to_bytes()
        assert not store.pending_mutations()


@pytest.mark.parametrize("after", [False, True])
def test_cancelled_replacement_preserves_cut_and_recovers_fresh_current(tmp_path, monkeypatch, after):
    control, identity = adapter(tmp_path), facts(receiver())
    with control.admission(identity) as owner:
        initial = control.create_initial(running(identity, owner), owner=owner)
        proposed = replace(initial, generation=2)
        original = control._replace_file
        def cancel(path, payload):
            if path.endswith("current.json"):
                if after:
                    original(path, payload)
                raise KeyboardInterrupt()
            return original(path, payload)
        monkeypatch.setattr(control, "_replace_file", cancel)
        with pytest.raises(KeyboardInterrupt) as raised:
            transition(control, owner, initial, proposed)
        assert raised.value.mutation_id and raised.value.cut.pending
        monkeypatch.setattr(control, "_replace_file", original)
        expected = proposed if after else initial
        assert control.reconcile(identity.operation_id, owner=owner).to_bytes() == expected.to_bytes()
        assert not control.control_store.pending_mutations()


def test_overridden_current_reader_is_fresh_and_original_errors_are_not_bypassed(tmp_path, monkeypatch):
    control, identity = adapter(tmp_path), facts(receiver())
    with control.admission(identity) as owner:
        current = control.create_initial(running(identity, owner), owner=owner)
        original = control._read_optional
        error, calls = ValueError("overridden current reader"), []
        def read(path):
            if path.endswith("current.json"):
                calls.append(path)
                raise error
            return original(path)
        monkeypatch.setattr(control, "_read_optional", read)
        for _ in range(2):
            with pytest.raises(ValueError) as raised:
                control.inspect(identity.operation_id)
            assert raised.value is error
        assert len(calls) == 2
        monkeypatch.setattr(control, "_read_optional", original)
        assert control.inspect(identity.operation_id).to_bytes() == current.to_bytes()


def test_initial_operation_and_expected_transition_are_atomic(tmp_path):
    control, identity = adapter(tmp_path), facts(receiver())
    with control.admission(identity) as owner:
        initial = control.create_initial(running(identity, owner), owner=owner)
        proposed = replace(initial, generation=2)
        assert transition(control, owner, initial, proposed) == proposed
        with pytest.raises(ManagedConflictError):
            transition(control, owner, initial, proposed)
        assert control.inspect(identity.operation_id) == proposed


def test_successful_control_publication_carries_native_persistence_confirmation(tmp_path, monkeypatch):
    """Completed native writes need no unsupported post-hoc directory barrier."""
    from dryml.core.store.records import MutationIntentRecord

    control, identity = adapter(tmp_path), facts(receiver())
    store = control.control_store
    monkeypatch.setattr(store, "_persist_authority_projection",
                        lambda record: pytest.fail("Successful native publication was treated as ambiguous."))
    with control.admission(identity) as owner:
        initial = control.create_initial(running(identity, owner), owner=owner)
        proposed = replace(initial, generation=2)
        assert transition(control, owner, initial, proposed) == proposed
    records = [MutationIntentRecord.from_bytes(path.read_bytes()).intent
               for path in (Path(store.base_dir) / "mutations").glob("*.record")]
    current = [record for record in records if "managed-current:" + identity.operation_id in record["scopes"]]
    assert len(current) == 2
    assert all(record["persistence_confirmed"] and record["status"] == "acknowledged" for record in current)
    assert not store.authority_cut().pending


def test_absent_operation_hierarchy_is_not_started_without_bootstrap(tmp_path):
    control = adapter(tmp_path)
    assert control.inspect(facts(receiver()).operation_id) is None
    assert not os.path.exists(control.root)


@pytest.mark.parametrize("after", [False, True])
def test_replace_failure_reconciles_only_exact_old_or_new_current(tmp_path, monkeypatch, after):
    """Ambiguous native failure retains an unconfirmed exact mutation for recovery."""
    from dryml.core.store.records import MutationIntentRecord

    control, identity = adapter(tmp_path), facts(receiver())
    with control.admission(identity) as owner:
        initial = control.create_initial(running(identity, owner), owner=owner)
        proposed = replace(initial, generation=2)
        original = control._replace_file
        def fail(path, payload):
            if path.endswith("current.json"):
                if after:
                    original(path, payload)
                raise OSError("replace acknowledgement failed")
            return original(path, payload)
        monkeypatch.setattr(control, "_replace_file", fail)
        with pytest.raises(StoreMutationError) as raised:
            transition(control, owner, initial, proposed)
        record = MutationIntentRecord.from_bytes(
            (Path(control.control_store.base_dir) / "mutations" / (raised.value.mutation_id + ".record")).read_bytes())
        assert record.intent["persistence_confirmed"] is False
        with pytest.raises(ManagedControlError, match="pending"):
            control.inspect(identity.operation_id)
        monkeypatch.setattr(control, "_replace_file", original)
        assert control.reconcile(identity.operation_id, owner=owner) == (proposed if after else initial)


def test_pending_neither_current_fails_closed(tmp_path):
    control, identity = adapter(tmp_path), facts(receiver())
    with control.admission(identity) as owner:
        control.create_initial(running(identity, owner), owner=owner)
        path = control._pending_path(control._operation_path(identity.operation_id))
        intent = _PendingIntent(identity.operation_id, 1, "d" * 64, 2, "e" * 64, uuid4().hex)
        control._replace_file(path, intent.to_bytes())
        with pytest.raises(ManagedControlError, match="does not identify"):
            control.reconcile(identity.operation_id, owner=owner)
        assert control._read_optional(path) == intent.to_bytes()


def test_projection_failure_preserves_committed_current_for_recovery_only_admission(tmp_path, monkeypatch):
    control, identity = adapter(tmp_path), facts(receiver())
    with control.admission(identity) as owner:
        initial = control.create_initial(running(identity, owner), owner=owner)
        original = control._finish_projection
        monkeypatch.setattr(control, "_finish_projection", lambda *args: (_ for _ in ()).throw(OSError("index phase")))
        proposed = replace(initial, generation=2)
        with pytest.raises(OSError, match="index phase"):
            transition(control, owner, initial, proposed)
        assert control.inspect(identity.operation_id) == proposed
    monkeypatch.setattr(control, "_finish_projection", original)
    with control.admission(identity) as recovered:
        assert recovered.reconciled and recovered.current == proposed
        with pytest.raises(ManagedConflictError, match="reconciliation_only"):
            recovered.realize(lambda _: pytest.fail("recovered admission executed"))


def test_initial_fsync_failure_does_not_publish_current(tmp_path, monkeypatch):
    control, identity = adapter(tmp_path), facts(receiver())
    with control.admission(identity) as owner:
        original = os.fsync
        monkeypatch.setattr(os, "fsync", lambda fd: (_ for _ in ()).throw(OSError("sync failed")))
        with pytest.raises(OSError):
            control.create_initial(running(identity, owner), owner=owner)
        monkeypatch.setattr(os, "fsync", original)
        assert control.inspect(identity.operation_id) is None


def test_builder_and_graph_preparation_run_outside_short_control_and_writer(tmp_path, monkeypatch):
    control, identity = adapter(tmp_path), facts(receiver())
    held = []
    original = control._control_lock
    @contextmanager
    def short():
        with original():
            held.append("control")
            try:
                yield
            finally:
                held.pop()
    monkeypatch.setattr(control, "_control_lock", short)
    with control.admission(identity) as owner:
        control.create_initial(running(identity, owner), owner=owner)
        def build(current):
            from dryml.core.store.retention import _writers
            assert not held and not _writers()
            return replace(current, generation=2)
        assert control.transition_running_owner(identity.operation_id, owner=owner,
            run_id=owner.run_id, invocation_id=owner.invocation_id,
            expected_generation=1, build=build).generation == 2


def test_interrupt_and_stale_owner_fence_builder_before_effects(tmp_path):
    control, identity = adapter(tmp_path), facts(receiver())
    with control.admission(identity) as owner:
        current = control.create_initial(running(identity, owner), owner=owner)
        outcome, interrupted = control.request_interrupt_if_running(identity.operation_id)
        assert outcome == "requested" and interrupted.generation == 2
        with pytest.raises(ManagedConflictError):
            control.transition_running_owner(identity.operation_id, owner=owner,
                run_id=current.run_id, invocation_id=current.invocation_id,
                expected_generation=1, build=lambda _: pytest.fail("stale builder"))


def test_unordered_status_checkpoint_interrupt_keep_exact_cas(tmp_path, monkeypatch):
    flip = force_fresh_handle_order(monkeypatch)
    node = unordered_saved_receiver()
    control, identity = adapter(tmp_path), facts(node)
    with control.admission(identity) as owner:
        root = ControlIntent(uuid4().hex, owner.run_id, owner.invocation_id, 1,
                             "adopted", kind="initial", selection=node)
        current = control.create_initial(replace(running(identity, owner), intents=(root,)), owner=owner)
        for _ in range(2):
            flip()
            first, second = control.inspect(identity.operation_id), control.inspect(identity.operation_id)
            assert first == second == current
            assert first.to_bytes() == second.to_bytes()
            assert control.status(identity.operation_id).state == "running"
            current = transition(control, owner, second, replace(second,
                generation=second.generation + 1, checkpoint=node,
                history=(ControlHistory(second.run_id, second.invocation_id, second.generation, "running", node),)))
        outcome, interrupted = control.request_interrupt_if_running(identity.operation_id)
        assert outcome == "requested" and interrupted.generation == current.generation + 1
        assert control.request_interrupt_if_running(identity.operation_id)[0] == "already_requested"
        assert control.status(identity.operation_id).interruption_requested
        with pytest.raises(ManagedConflictError):
            transition(control, owner, current, replace(current, generation=current.generation + 1))
        interrupted = control.inspect(identity.operation_id)
        final = transition(control, owner, interrupted, replace(interrupted,
            generation=interrupted.generation + 1, state="interrupted", interrupt_request=None))
        assert final.checkpoint.graph_equal(node) and control.inspect(identity.operation_id) == final


@pytest.mark.parametrize("seam", ["current", "start_write", "start_admit", "start_previous"])
def test_exact_source_witness_rejects_real_raw_byte_change(tmp_path, monkeypatch, seam):
    """Equal logical metadata cannot replace an exact fresh persisted byte check."""
    control, identity = adapter(tmp_path), facts(receiver())
    with control.admission(identity) as owner:
        initial = control.create_initial(running(identity, owner), owner=owner)
        current_path = Path(control._current_path(control._operation_path(identity.operation_id)))
        previous = control.inspect(identity.operation_id)
        if seam == "current":
            # Mutation after capture must be detected at the actual writer CAS.
            original = control.control_store.preflight_publication
            def changed(*args, **kwargs):
                current_path.write_bytes(previous.to_bytes() + b"\n")
                return original(*args, **kwargs)
            monkeypatch.setattr(control.control_store, "preflight_publication", changed)
            with pytest.raises(ManagedConflictError, match="exact current changed"):
                transition(control, owner, initial, replace(initial, generation=2))
            assert current_path.read_bytes() == previous.to_bytes() + b"\n"
        else:
            proposed = replace(previous, generation=2)
            prepared = control.write_preparation(StartPreparation(uuid4().hex, previous, proposed),
                                                 owner=owner, expected=None)
            ready = control.write_preparation(replace(prepared, phase="ready"), owner=owner, expected=prepared)
            observed = control.read_preparation(identity.operation_id)
            path = Path(control._preparation_path(identity.operation_id))
            if seam == "start_previous":
                current_path.write_bytes(previous.to_bytes() + b"\n")
                assert control.inspect(identity.operation_id) == previous
                with pytest.raises(ManagedConflictError, match="authority or interrupt fence changed") as raised:
                    control.admit_prepared(observed, owner=owner)
                assert raised.value.reason == "generation_conflict"
                assert current_path.read_bytes() == previous.to_bytes() + b"\n"
            else:
                path.write_bytes(observed.to_bytes() + b"\n")
                assert control.read_preparation(identity.operation_id) == ready
                with pytest.raises(ManagedConflictError, match="preparation_conflict"):
                    if seam == "start_write":
                        control.write_preparation(replace(observed, phase="admitted"), owner=owner, expected=observed)
                    else:
                        control.admit_prepared(observed, owner=owner)
                assert path.read_bytes() == observed.to_bytes() + b"\n"
            assert not control.control_store.pending_mutations()


@pytest.mark.parametrize("admitted", [False, True])
def test_start_reconciliation_rejects_byte_changed_logical_match(tmp_path, admitted):
    from dryml.managed import runtime
    from dryml.managed.errors import ManagedRecoveryError

    control, identity = adapter(tmp_path), facts(receiver())
    with control.admission(identity) as owner:
        initial = control.create_initial(running(identity, owner), owner=owner)
        previous = control.inspect(identity.operation_id)
        prepared = control.write_preparation(StartPreparation(uuid4().hex, previous,
            replace(previous, generation=2)), owner=owner, expected=None)
        ready = control.write_preparation(replace(prepared, phase="ready"), owner=owner, expected=prepared)
        current = control.admit_prepared(ready, owner=owner) if admitted else initial
        path = Path(control._current_path(control._operation_path(identity.operation_id)))
        changed = path.read_bytes() + b"\n"
        path.write_bytes(changed)
        assert control.inspect(identity.operation_id) == current
        with pytest.raises(ManagedRecoveryError, match="ambiguous_admission"):
            runtime._settle_preparation(control, owner, control.read_preparation(identity.operation_id))
        assert path.read_bytes() == changed
        assert not control.read_preparation(identity.operation_id).settled


def _read_unordered_control(root, operation_id, queue):
    with pytest.MonkeyPatch.context() as patch:
        flip = force_fresh_handle_order(patch)
        flip()
        store = DirStore(root, query_index="none")
        control = ManagedControlStore(store, Repo(store))
        current = control.inspect(operation_id)
        preparation = control.read_preparation(operation_id)
        raw = Path(control._current_path(control._operation_path(operation_id))).read_bytes()
        start_raw = Path(control._preparation_path(operation_id)).read_bytes()
        queue.put((current.to_bytes() == raw, preparation.to_bytes() == start_raw,
                   preparation.previous.to_bytes() == raw, current == preparation.previous))


def test_unordered_preparation_fresh_process_and_redecoded_phase_cas(tmp_path, monkeypatch):
    flip = force_fresh_handle_order(monkeypatch)
    control, identity = adapter(tmp_path), facts(unordered_saved_receiver())
    with control.admission(identity) as owner:
        current = control.create_initial(running(identity, owner), owner=owner)
        previous_bytes = current.to_bytes()
        flip()
        previous = control.inspect(identity.operation_id)
        proposed = replace(previous, generation=2)
        prepared = control.write_preparation(StartPreparation(uuid4().hex, previous, proposed), owner=owner, expected=None)
        ctx = multiprocessing.get_context("spawn")
        queue = ctx.Queue()
        process = ctx.Process(target=_read_unordered_control,
            args=(control.control_store.base_dir, identity.operation_id, queue))
        try:
            process.start()
            assert queue.get(timeout=30) == (True, True, True, True)
            process.join(15)
            assert process.exitcode == 0
        finally:
            if process.is_alive():
                process.terminate()
                process.join(5)
        flip()
        observed = control.read_preparation(identity.operation_id)
        assert observed == prepared and observed.previous.to_bytes() == previous_bytes
        ready = control.write_preparation(replace(observed, phase="ready"), owner=owner, expected=observed)
        flip()
        ready = control.read_preparation(identity.operation_id)
        admitted = control.admit_prepared(ready, owner=owner)
        assert admitted.to_bytes() == ready.proposed.to_bytes()
        observed = control.read_preparation(identity.operation_id)
        settled = control.write_preparation(replace(observed, phase="admitted", settled=True), owner=owner, expected=observed)
        assert control.read_preparation(identity.operation_id) == settled
        assert control.inspect(identity.operation_id) == admitted


def _bootstrap(root, ready, start, queue):
    store = DirStore(root, query_index="none")
    control = ManagedControlStore(store, Repo(store))
    ready.set()
    assert start.wait(15)
    control.initialize()
    queue.put("ok")


def test_spawned_bootstrap_uses_writer_lock_and_one_final_gate(tmp_path):
    ctx = multiprocessing.get_context("spawn")
    ready = [ctx.Event(), ctx.Event()]
    start, queue = ctx.Event(), ctx.Queue()
    root = os.fspath(tmp_path / "store")
    # The Store itself exists before racing only managed namespace bootstrap.
    store = DirStore(root, query_index="none")
    processes = [ctx.Process(target=_bootstrap, args=(root, event, start, queue)) for event in ready]
    try:
        for process in processes:
            process.start()
        assert all(event.wait(15) for event in ready)
        start.set()
        for process in processes:
            process.join(15)
        assert all(process.exitcode == 0 for process in processes)
        assert [queue.get(timeout=2), queue.get(timeout=2)] == ["ok", "ok"]
        assert ManagedControlStore(store, Repo(store)).inspect("a" * 64) is None
    finally:
        start.set()
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(5)
