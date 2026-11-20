"""Independent managed workers share SQLite-backed Store publication safely."""

import multiprocessing
import traceback

import pytest

from dryml.core import Repo
from dryml.core.store.dir import DirStore


def _checkpoint_body(obj, context):
    for _ in range(3):
        obj.value += 1
        context.checkpoint()
    return obj.value


def _coordinate_finalization(value, ready, pending, confirmed):
    """Expose a real unrelated control CAS after the first worker's grant ack."""
    from dryml.core.store.retention import RetentionAuthority
    from dryml.managed import runtime
    from dryml.managed.control import ManagedControlStore

    if value == 32:
        original_confirm = runtime._confirm_final
        original_adopt = RetentionAuthority.adopt
        confirming = False

        def adopt(self, *args, **kwargs):
            ledger = original_adopt(self, *args, **kwargs)
            if confirming:
                ready.set()
                assert pending.wait(20), "other operation did not publish its control CAS"
                records = self.store.pending_mutations()
                assert records and all(
                    "managed-current:" + operation_id not in record.intent["scopes"]
                    for record in records)
            return ledger

        def confirm(control, current):
            nonlocal confirming, operation_id
            operation_id = current.operation_id
            confirming = True
            try:
                return original_confirm(control, current)
            finally:
                confirming = False
                confirmed.set()

        operation_id = None
        RetentionAuthority.adopt = adopt
        runtime._confirm_final = confirm
    else:
        original_publish = ManagedControlStore._publish
        original_finish = ManagedControlStore._finish_projection
        publishing = False

        def publish(self, current, proposed):
            nonlocal publishing
            if current is not None and current.state == "running" and proposed.state == "finalizing":
                assert ready.wait(20), "first operation did not confirm its grants"
                publishing = True
            try:
                return original_publish(self, current, proposed)
            finally:
                publishing = False

        def finish(self, mutation_id=None):
            if publishing:
                assert mutation_id in self.control_store.authority_cut().pending
                pending.set()
                assert confirmed.wait(20), "first operation did not finish confirmation"
            return original_finish(self, mutation_id)

        ManagedControlStore._publish = publish
        ManagedControlStore._finish_projection = finish


def _worker(root, value, barrier, ready, pending, confirmed, result):
    from dryml.environments import introspection
    from dryml.managed import ManagedConfig
    from tests.conftest import _synthetic_environment_record
    from tests.managed.u16_fixtures import Value

    introspection.inspect_current = _synthetic_environment_record
    repo = Repo(DirStore.open_existing(root))
    try:
        _coordinate_finalization(value, ready, pending, confirmed)
        Value.action = _checkpoint_body
        live = Value(value, repo=repo)
        barrier.wait(timeout=30)
        outcome = live.run(managed=ManagedConfig(state_repo=repo))
        result.send((outcome.execution.operation_id, outcome.value, Value.bodies, Value.captures))
    except BaseException:
        result.send(traceback.format_exc())
    finally:
        repo.close(flush=False)
        result.close()


@pytest.mark.usefixtures("fixed_managed_snapshot_environment")
def test_final_confirmation_refuses_own_pending_control(tmp_path, monkeypatch):
    """An unrelated-work allowance must still reject real unacknowledged own CAS."""
    from dataclasses import replace

    from dryml.core.store.retention import RetentionAuthority
    from dryml.managed import ManagedConfig, ManagedRecoveryError, runtime
    from tests.managed.u16_fixtures import Value, setup

    control, live = setup(tmp_path, index="sqlite")
    store = control.control_store
    Value.action = lambda obj, context: obj.value
    try:
        result = live.run(managed=ManagedConfig(state_repo=control.state_repo))
        current = control.inspect(result.execution.operation_id)
        original_adopt = RetentionAuthority.adopt

        def fail_ack(*args):
            raise OSError("own control acknowledgement interrupted")

        def adopt(self, *args, **kwargs):
            ledger = original_adopt(self, *args, **kwargs)
            with monkeypatch.context() as patch:
                patch.setattr(control, "_finish_projection", fail_ack)
                with pytest.raises(OSError, match="own control acknowledgement interrupted"):
                    control._publish(current, replace(current, generation=current.generation + 1))
            assert store.authority_cut().pending
            return ledger

        with monkeypatch.context() as patch:
            patch.setattr(RetentionAuthority, "adopt", adopt)
            with pytest.raises(ManagedRecoveryError, match="pending_finalization"):
                runtime._confirm_final(control, current)
        assert control._pending_mutations(current.operation_id)
        control._finish_projection()
        assert not store.authority_cut().pending
    finally:
        control.state_repo.close(flush=False)


def test_two_processes_checkpoint_and_complete_distinct_operations_in_one_store(tmp_path):
    """Unrelated pending control work must not fence exact final acknowledgements."""
    from dryml.managed.control import ManagedControlStore

    repo = Repo(DirStore(tmp_path / "store"))
    root = repo.default_store.base_dir
    ctx = multiprocessing.get_context("spawn")
    barrier = ctx.Barrier(2)
    ready, pending, confirmed = (ctx.Event() for _ in range(3))
    pipes = [ctx.Pipe(duplex=False) for _ in range(2)]
    processes = [ctx.Process(target=_worker, args=(root, value, barrier, ready, pending, confirmed, pipes[i][1]))
                 for i, value in enumerate((32, 64))]
    try:
        for process in processes:
            process.start()
        outcomes = []
        for reader, writer in pipes:
            writer.close()
            assert reader.poll(90), "managed worker did not finish"
            outcomes.append(reader.recv())
        for process in processes:
            process.join(15)
            assert process.exitcode == 0
        control = ManagedControlStore(repo.default_store, repo)
        for initial, outcome in zip((32, 64), outcomes):
            assert isinstance(outcome, tuple), outcome
            operation, value, bodies, captures = outcome
            assert (value, bodies, captures) == (initial + 3, 1, 4)
            current = control.inspect(operation)
            assert current.state == "completed"
            assert repo.load_state_ref(current.final, reuse_live="never").value == initial + 3
            assert repo.load_state_ref(current.checkpoint, reuse_live="never").value == initial + 3
            current.result_ack.confirm((repo.default_store,))
            final, = (root for root in current.intents if root.kind == "final")
            assert final.disposition == "adopted" and final.selection.graph_equal(current.final)
        assert len({outcome[0] for outcome in outcomes}) == 2
        assert not repo.default_store.authority_cut().pending
        assert repo.default_store.open_query_index().status().state == "ready"
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            if process.pid is not None:
                process.join(10)
        for reader, writer in pipes:
            reader.close()
            writer.close()
        repo.close(flush=False)
