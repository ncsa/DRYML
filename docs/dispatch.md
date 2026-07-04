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

## Worker Protocol

The local backend launches `python -m dryml.dispatch.worker` with JSON request, handshake, and response files in a per-dispatch work directory. Child stdout/stderr are captured to separate files from process start, so user output cannot corrupt protocol JSON.

The worker handshake reports protocol version, Python/platform, pid, supported operation kinds, call transports, store kinds, record schemas, runtime modes, environment kind, process-group support, and store accessibility. The parent waits for this phase before trusting a worker result. Missing features, protocol mismatch, or inaccessible store paths return structured unsupported responses.

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

`DispatchResult` returns compact fields: status, operation/dispatch/recipe IDs, execution record ID, canonical literal or CDef result refs, operation-produced record IDs, stdout/stderr refs, diagnostics, error, and cancellation. `execution_record_id` is provenance and is kept separate from `produced_record_ids`, which are reserved for records produced by the operation itself.

When provenance is enabled, operation, dispatch, and execution-recipe specs are written beside the execution record so provenance refs are store-resolvable. `ExecutionRecord` sidecars are emitted for success, user-code failure, timeout, cancellation, and parent-side protocol failures when metadata permits. stdout/stderr products use self product refs such as `products/<execution-record-id>/stdout.txt` and `stderr.txt`.

## Cancellation

`LocalSubprocessFuture.cancel()` starts POSIX process-group cancellation with SIGINT, escalates to SIGTERM, then SIGKILL when needed. `result(timeout=...)` cancels and records timeout provenance. `KeyboardInterrupt` while waiting cancels the worker and re-raises.
