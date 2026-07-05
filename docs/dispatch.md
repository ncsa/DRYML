# DRYML Dispatch

`dryml.dispatch` now contains both the canonical metadata plane and the first real execution backend: a one-operation local subprocess worker. `dryml.execute` remains available as the legacy pickled-callable compatibility path; deletion or rerouting of `dryml.execute` is intentionally deferred.

## Metadata

`DispatchSpec` is request intent: operation ID plus policies and overrides that affect dispatch identity. `ExecutionRecipe` is resolved plan metadata: backend, environment/runtime choices, store strategy, input/output plan, and log plan. Both are canonical JSON specs, not DRYML Objects.

```python
import dryml.operations as ops
from dryml.dispatch import attach_dispatch_id, attach_recipe_id, make_dispatch_spec, make_execution_recipe

operation = ops.attach_operation_id(ops.make_function_call_spec("my_pkg.train:run", args=[1, 2]))
dispatch = attach_dispatch_id(make_dispatch_spec(operation_id=operation["id"], operation=operation))
recipe = attach_recipe_id(
    make_execution_recipe(
        dispatch_id=dispatch["id"],
        operation_id=operation["id"],
        backend={"name": "dryml.local_subprocess", "kind": "local_subprocess"},
    )
)
```

An `ExecutionEnvelope` is launch-time worker protocol data. It may include absolute same-host `DirStore` paths, work directories, and pickle file paths. Those fields are deliberately excluded from `DispatchSpec` and `ExecutionRecipe` identity.

## Local Subprocess

The high-level API plans and runs one function or method operation in a clean child process:

```python
from dryml.dispatch import Dispatcher, LocalSubprocessBackend

dispatcher = Dispatcher(backend=LocalSubprocessBackend(), store=repo.default_store)
result = dispatcher.run(operation, record_policy="descriptive")
```

Convenience wrapper:

```python
result = dryml.dispatch.run(operation, backend="local_subprocess", store=repo.default_store)
```

Function-call dispatch uses an import path and imports the target only inside the worker after runtime activation:

```python
operation = ops.attach_operation_id(ops.make_function_call_spec("my_pkg.math:add", args=[1, 2]))
result = dispatcher.run(operation)
```

Method-call dispatch materializes the subject CDef from the shared/output store in the worker:

```python
operation = ops.attach_operation_id(ops.make_method_call_spec("cdef-v4-...", "train", kwargs={"epochs": 1}))
result = dispatcher.run(operation)
```

`FunctionRef`/import-path function calls and method calls are the preferred portable path. `PickledCallable` exists only as an explicit same-Python convenience and is marked non-portable in the recipe constraints.

## Local Worlds

`Dispatcher.run_world(...)` is the explicit Sprint 10 entrypoint for coordinated same-host multi-worker dispatch. `Dispatcher.run(...)` remains the single-worker local subprocess path for compatibility.

```python
from dryml.dispatch import Dispatcher

result = Dispatcher(store=repo.default_store).run_world(
    operation,
    world={
        "trainer": {"replicas": 1, "process": {"resources": {"cpus": 2}}},
        "data": {"replicas": 1, "process": {"resources": {"cpus": 1}}},
    },
)
```

The first implementation runs the same `OperationSpec` in every allocated role/replica. User code can branch on the active runtime allocation:

```python
import dryml.runtime as rt

def worker_main():
    alloc = rt.require_workload_allocation("run role-aware operation")
    return {"role": alloc.role, "replica": alloc.replica, "rank": alloc.rank}
```

The worker process environment also includes `DRYML_WORLD_ID`, `DRYML_WORLD_ALLOCATION_ID`, `DRYML_WORLD_ROLE`, `DRYML_WORLD_REPLICA`, `DRYML_WORLD_RANK`, `DRYML_WORLD_LOCAL_RANK`, `DRYML_WORLD_SIZE`, and `DRYML_WORLD_ROLE_SIZE`.

`WorldDispatchResult` contains aggregate `status`, `dispatch_id`, `recipe_id`, `world_id`, `world_allocation_id`, `primary`, `workers`, `execution_record_ids`, diagnostics, error, and cancellation fields. Worker results are keyed by `WorldWorkerKey(role, replica, rank, local_rank)`. The primary result is `main` replica 0 if present, else `worker` replica 0, else the first sorted worker. Aggregate status is conservative: any timeout, failure, cancellation, or unsupported worker prevents an `ok` aggregate.

The local-world backend creates one group work directory, launches one subprocess per role/replica, waits for all handshakes, then writes a start marker. If a required worker fails, times out, reports unsupported, or misses protocol files, siblings are cancelled. `LocalWorldFuture.cancel(...)`, `result(timeout=...)`, and `KeyboardInterrupt` cancellation all target the whole worker group.

Pre-start control-plane failures keep their status shape: malformed or mismatched handshakes report `failed`, unsupported handshakes report `unsupported`, and handshake timeouts report `timeout`; they are not collapsed into user cancellation. `result(timeout=...)` applies to post-handshake execution waiting after the configured handshake timeout phase.

Local-world dispatch is local-only: all workers run on the same host and share the same `DirStore` path. Distributed rendezvous, collectives, Ray/Slurm/cloud launch, containers, SSH, role-specific runtime-spec selection, and heterogeneous role-specific Python executables are deferred. Role `process.env` values and DRYML role/rank facts are applied per worker; the software `EnvironmentSpec` and `RuntimeContextSpec` are currently homogeneous across the world.

## Worker Protocol

The local backend launches `python -m dryml.dispatch.worker` with JSON request, handshake, and response files in a per-dispatch work directory. Child stdout/stderr are captured to separate files from process start, so user output cannot corrupt protocol JSON.

The worker handshake reports protocol version, Python/platform, pid, supported operation kinds, call transports, store kinds, record schemas, runtime modes, environment kind, process-group support, and store accessibility. The parent waits for this phase before trusting a worker result, and an `ok` worker response is accepted only after an observed `ok` handshake. Missing features, protocol mismatch, inaccessible store paths, or inconsistent envelope IDs return structured failed/unsupported responses.

## Store Marshalling

Local subprocess dispatch prefers same-host `DirStore` handoff:

```text
parent writes/has objects and sidecars in DirStore
worker opens the same absolute DirStore path through WorkerStoreRef
worker materializes CDef args and writes outputs/records/products back
parent reads compact result refs and records
```

`WorkerStoreRef` roles are `input`, `work`, `output`, and `shared`; modes are `read`, `write`, and `readwrite`. Request/response JSON carries CDef IDs and record IDs, not object state bytes.

## Runtime And Environment

The worker enters `RuntimeMode.WORKER` with a real CPU-only `RuntimeAllocationView` by default and assigned device visibility before target import or object materialization. Supported launch specs are current Python, explicit Python executable, and Conda command construction/direct prefix launch where available.

Python path policies are:

| Policy | Behavior |
|---|---|
| `none` | Do not alter `PYTHONPATH` beyond the child environment. |
| `inherit` | Use parent `PYTHONPATH`. |
| `explicit` | Use only `extra_pythonpath`. |
| `dryml-source` | Add the current DRYML source root, then `extra_pythonpath`. |

## Results, Logs, And Records

`DispatchResult` returns compact fields: status, operation/dispatch/recipe IDs, execution record ID, canonical literal or CDef result refs, operation-produced record IDs, stdout/stderr refs, diagnostics, error, and cancellation. `WorkerResponse` enforces the same basic status context as execution records: ok responses do not carry errors, failed/timeout/unsupported responses include error details or diagnostics, and cancelled responses include cancellation facts. `execution_record_id` is provenance and is kept separate from `produced_record_ids`, which are reserved for records produced by the operation itself.

When provenance is enabled, operation, dispatch, and execution-recipe specs are written beside the execution record so provenance refs are store-resolvable. `ExecutionRecord` sidecars are emitted for success, user-code failure, timeout, cancellation, and parent-side protocol failures when metadata permits. stdout/stderr products use self product refs such as `products/<execution-record-id>/stdout.txt` and `stderr.txt`.

In local-world mode, `plan_world(...)` writes the requested `WorldSpec`, operation spec, dispatch spec, execution recipe, and actual `WorldAllocation` spec when provenance is enabled, before worker execution records reference them. Per-worker execution records are the Sprint 10 provenance authority; each includes `world_id`, `world_allocation_id`, worker key payload data, and role/replica/rank/local-rank metadata. Per-worker stdout/stderr are captured independently and copied into each worker execution record's product directory.

## Cancellation

`LocalSubprocessFuture.cancel()` starts POSIX process-group cancellation with SIGINT, escalates to SIGTERM, then SIGKILL when needed. `result(timeout=...)` cancels and records timeout provenance. `KeyboardInterrupt` while waiting cancels the worker and re-raises.
