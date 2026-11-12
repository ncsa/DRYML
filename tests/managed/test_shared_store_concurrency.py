"""Independent managed workers share SQLite-backed Store publication safely."""

import multiprocessing
import traceback

from dryml.core import Repo
from dryml.core.store.dir import DirStore


def _checkpoint_body(obj, context):
    for _ in range(3):
        obj.value += 1
        context.checkpoint()
    return obj.value


def _worker(root, value, barrier, result):
    from dryml.environments import introspection
    from dryml.managed import ManagedConfig
    from tests.conftest import _synthetic_environment_record
    from tests.managed.u16_fixtures import Value

    introspection.inspect_current = _synthetic_environment_record
    repo = Repo(DirStore.open_existing(root))
    try:
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


def test_two_processes_checkpoint_and_complete_distinct_operations_in_one_store(tmp_path):
    from dryml.managed.control import ManagedControlStore

    repo = Repo(DirStore(tmp_path / "store"))
    root = repo.default_store.base_dir
    ctx = multiprocessing.get_context("spawn")
    barrier = ctx.Barrier(2)
    pipes = [ctx.Pipe(duplex=False) for _ in range(2)]
    processes = [ctx.Process(target=_worker, args=(root, value, barrier, pipes[i][1]))
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
