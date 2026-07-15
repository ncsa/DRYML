# DRYML Dispatch

`dryml.dispatch` contains both the canonical metadata plane and the reference local execution backend: a one-operation local subprocess worker.

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

## Requirement-Aware Planning

`plan`, `submit`, and `run` normalize an operation once, collect its static
annotation facts, resolve requirements/defaults, select candidates, and check
the selected candidates before launch. Explicit `environment=`, `world=`, and
`runtime=` values choose candidates but do not bypass hard requirements.

Candidate precedence is deterministic: explicit, annotation-default, and
context-current candidates remain authoritative. Only when those slots are
absent, environment planning can perform a bounded explicit-registry search and
world planning can synthesize a minimal local world for a hard requirement.
Planning never searches or synthesizes after an incompatible higher-precedence
candidate.

`requirement_policy` accepts `"strict"`, `"warn"`, or `"ignore"`. When it is
omitted, active `RuntimeEnforcement.STRICT`, `.WARN`, and `.OFF` select strict,
warn, and ignore respectively. Warn and ignore relax only requirement checks;
they cannot bypass invalid operation structure, worker/allocation safety, or the
same-environment restriction of `PickledCallable`/`pickle_small` transport.

Use `explain(...)` to inspect the same pipeline without launching a worker,
allocating workload resources, or creating execution records:

```python
explanation = dryml.dispatch.explain(
    train_fn,
    store=repo.default_store,
    environment=my_environment,
)
print(explanation.launchable)
print(explanation.resolution.environment_check.to_data())
```

Explanation may perform bounded static analysis, code/environment probes,
explicit-registry resolution, and read-only local inventory/synthesis when
needed. It does not launch workloads, activate an allocation, persist records,
or solve/install packages. It does run the same
non-allocating local capacity validation as planning, so an explanation is not
launchable when the selected one-worker world cannot fit the supplied or
discovered inventory. `plan(...)` and `plan_world(...)` then validate the actual
backend allocation against hard world requirements before constructing workers
when requirement policy is `strict`; `warn` and `ignore` retain their established
compatibility semantics while never bypassing allocation feasibility.

### Explicit current-process dynamic trace

Dynamic tracing is off by default. It is requested only with the closed
`analysis_policy` mapping; a `CodeAnalysisContext`, including one with
`allow_dynamic_execution=True`, remains a non-tracing compatibility form.

```python
plan = dispatcher.plan(
    orchestration_function,
    store=repo.default_store,
    args=(stored_cdef,),
    analysis_policy={"dynamic_trace": True},
)
```

The only mapping members are `context`, a positive finite `probe_timeout_s`,
and `dynamic_trace`. The trace member is exactly `True` (the default
`DynamicTracePolicy`) or an already validated `DynamicTracePolicy`; falsey or
truthy substitutes, mappings, and unknown keys are rejected before operation
normalization or pickle transport creation.

Only a live, exact, synchronous Python function is traceable.  Explicit
`PickledCallable` is traceable only after its preliminary candidate is confirmed
to use the current Python.  OperationSpec-only, source-only, CDef/object method,
bound-method, callable-instance, class, builtin, coroutine, generator, and
unsupported-container targets fail before invocation.  Trace-aware calls require
an exact outer `tuple` of arguments and exact `dict[str, value]` kwargs.
Dispatch performs that exact-function gate before generic callable inspection,
importability metadata reads, or pickle creation, so unsupported callable
instances, classes, wrappers, and descriptors are not inspected merely to decide
trace eligibility.

Dispatch first canonicalizes the ordinary worker payload, then derives the
*effective worker invocation* with `resolve_call_arguments()`: raw CDef IDs are
read structurally from the planning Store without building objects;
`ref(cdef-v...)` remains an ID string; `{"$literal": value}` unwraps once; and
nested lists/mappings retain canonical shape. Caller-supplied live CDefs are
checked structurally against the Store before their private trace proxy is used.
For `pickle_small`, the canonical identity marker remains in the operation, but
the validated marker suffix is stripped at `identity_arg_count` before tracing.
Malformed marker/count data, missing/mismatched CDefs, aliases, and inputs that
cannot be reconstructed without building fail before the facade or target.

For an eligible input, dispatch calls `dryml.code.trace(...)` exactly once in the
caller process and resolves direct fragments followed by accepted trace calls
and serialized method facts through `dryml.annotations`. Identical fragments are
first-occurrence deduplicated; later observations remain provenance only.
`explain` performs that same explicitly requested trace, but remains
non-launching and non-persisting. Requested trace failures, incomplete results,
malformed/rejected evidence, and provenance-limit failures block planning under
`strict`, `warn`, and `ignore`; partial facts are never treated as an empty
requirement set.

This executes trusted user code in the current process. It is not a sandbox,
does not have a hard timeout, and is not sent to a probe, selected environment,
or worker. Dispatch forces the private facade context to collect post-start
trace failures even when the caller context uses `diagnostics_policy="raise"`.
Per-run bounded trace evidence is carried in dispatch/recipe/envelope planning
metadata and explanations, not in immutable operation metadata. The versioned
projection allows at most 256 calls, 1,024 accepted/duplicate observations, 256
diagnostics, 4,096 characters per scalar, depth 32, and 1 MiB JSON; overflow is
reported as `provenance_limit_exceeded`, never truncated. Its validated complete
or incomplete 9B summary is retained while calls and observations are empty, so
an over-limit trace is not mistaken for an empty trace. The policy restored from
the carrier has the exact 9B bounds (`max_calls` 1 through 10,000), and only the
four normalized transport tokens `import_path`, `pickle_small`,
`operation_spec`, and `method_call` are accepted; an unknown token is a schema
error rather than being serialized or substituted.

A null `trace_input_id` is allowed only when the effective invocation itself
could not be constructed. Dispatch admits a no-summary `pre_execution_failed`
carrier only for the exact 9B diagnostic-only set (`dynamic_trace_disabled`,
`dynamic_trace_invalid_context`, `dynamic_trace_unsupported_target`,
`dynamic_trace_unsupported_argument`, `dynamic_trace_argument_limit_exceeded`,
and `dynamic_trace_receiver_resolution_failed`), with no facts and a target
known not to have started. Stale target/envelope evidence, malformed summaries,
unknown summary outcomes, and other malformed or mixed evidence are
`evidence_rejected`. Dispatch independently validates a summary and calls before
that rejection: evidence proving execution started retains nonempty input/run
IDs and `execution_started=true`; genuinely unknown start is represented by
`null`. Only independently validated bounded summary/call wires can be retained
for diagnostics, never for requirement resolution or publication. Each carrier
diagnostic uses the fixed machine schema `{"code": str, "severity":
"info"|"warning"|"error", "data": {"trace_diagnostic_codes": [str,
...]}}`. The bounded `trace_diagnostic_codes` array preserves safely available
underlying 9B code identifiers without promoting their messages or arbitrary
data into the projection. Carriers contain no exception messages, tracebacks,
locals, source, environment values, streams, live objects, or arbitrary repr.
If accepted trace facts change a `pickle_small` final candidate to a different
Python environment, planning cleans the temporary pickle and blocks launch while
returning that completed diagnostic trace carrier.

## Unsupported Graph Prototype Package

`dryml.graph` is not a supported DRYML package or public export. Clean source
exports, wheels, and source distributions exclude it; applications must not
import it. This contract concerns tracked distribution contents only and does not
delete or otherwise modify any untracked local prototype directory.

## Local Worlds

`Dispatcher.run_world(...)` is the explicit Sprint 8 entrypoint for coordinated same-host multi-worker dispatch. `Dispatcher.run(...)` remains the single-worker local subprocess path for compatibility.

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
# Automatic Local Planning

When no explicit, annotation-default, or context-current environment/world is
selected, dispatch can resolve an injected environment registry and synthesize a
local world for hard requirements. Higher-precedence candidates are never
silently replaced. `dispatch.explain(...)` uses the same bounded discovery path
but does not launch work, activate an allocation, or write execution records.

A synthesized single-worker local world can run through local subprocess
dispatch. Multi-worker worlds require `plan_world(...)` or `run_world(...)`.

Environment candidates are ordered as caller candidates, name-sorted entries in
an explicitly supplied `EnvironmentRegistry`, then the current environment.
Registry hints only avoid definite mismatches; a probe record remains the
compatibility authority. Resolver input, probes, trace metadata, and probe
output are bounded, including resolver probe durations and aggregate inventory
metadata. Candidate discovery never replaces an incompatible explicit,
annotation-default, or context-current environment/world.

When candidates are retained on a `Dispatcher` for repeated notebook calls,
pass a re-iterable collection such as a tuple or list. One-shot iterators are
rejected at construction so an earlier `explain(...)` cannot consume candidates
that a later `plan(...)` would need. If bounded candidate enumeration reaches its
deadline, resolution is reported as incomplete and does not fall through to a
lower-precedence registry or current-environment candidate.

`inventory=` injects one `LocalResourceInventory` for synthesis and allocation.
With no injection, `inventory_policy="lightweight"` is framework-free. To use
`"external"`, call `worlds.local_inventory(policy="external",
command_runner=...)` yourself and pass that inventory to dispatch; dispatcher
methods do not accept a command runner. The runner timeout is cooperative for
custom in-process callbacks, which must enforce any hard deadline themselves.
Actual allocation feasibility, backend support, target importability, and topology
the backend cannot enforce remain blocking even under
`requirement_policy="warn"` or `"ignore"`.

`plan_world(..., oversubscribe=True)` is an explicit advanced local-world
allocator policy. Automatic synthesis remains disjoint; the resulting planning
metadata records either `disjoint_local` or `oversubscribed_local` under
`dryml.world_allocation`.

For notebooks, ordinary context APIs are sufficient:

```python
registry = dryml.environments.EnvironmentRegistry()
dryml.environments.set_current(dryml.environments.CurrentEnvironmentSpec())
world = dryml.worlds.synthesize(None, inventory=dryml.worlds.local_inventory()).require_world()
with dryml.worlds.use(world):
    print(dryml.dispatch.explain(train, environment_registry=registry))
```

`explain(...)` may read local inventory and perform bounded probes, but never
launches workloads, activates an allocation, writes Store records, or mutates
the registry. Cross-plan probe and inventory caching is intentionally deferred.
