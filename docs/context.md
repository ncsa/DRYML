# Runtime, Worlds, And Dispatch

Status: current.

DRYML separates execution concerns into five layers:

- `dryml.session` is the persistent common facade. Fresh sessions intentionally
  use unchecked Python; managed and orchestrator checks are explicit opt-ins.
- `dryml.environments` describes software compatibility, such as Python versions and package requirements.
- `dryml.worlds` describes resource and topology requirements and requested launch shapes.
- `dryml.runtime` describes process-local mode, active allocation, device visibility, framework bootstrap, and guards.
- `dryml.dispatch` runs operation specs in local subprocess workers and records execution metadata.

## Declaring Requirements

```python
import dryml


@dryml.env.req(packages={"torch": ">=2.4"})
@dryml.world.req(cpus={"min": 2}, accelerators={"gpu": {"min": 1}})
@dryml.world.default(cpus=4, memory="16GiB", accelerators={"gpu": 1})
@dryml.runtime.default(mode="worker", device_visibility={"policy": "assigned"})
def train(model, data):
    allocation = dryml.runtime.require_worker_allocation("train() uses workload resources")
    return {"role": allocation.role, "cpus": allocation.cpus}
```

Decorators attach metadata only. They do not allocate resources, enter runtime, import frameworks, or spawn workers.

## Common Session Setup

```python
import dryml

dryml.session.manage(cpus=2)       # Current process, CPU-only and checked.
dryml.session.request_world(cpus=2, gpus=1)  # Later worker intent.
```

The two calls intentionally describe different processes. `manage()` affects
direct annotated calls and framework visibility in this process;
`request_world()` supplies a lower-precedence default for future dispatch. See
[Sessions](session.md) for atomic `configure(...)`, snapshots, status reporting,
safe reset, and the post-framework-import restart boundary.

## Running Work

```python
import dryml

operation = dryml.operations.attach_operation_id(
    dryml.operations.make_function_call_spec("my_project.training:train", args=["cdef-v4-model", "cdef-v4-data"])
)
result = dryml.dispatch.run(operation, backend="local_subprocess")
```

Normal callers can instead pass an importable function or stored CDef plus method
name to `dryml.dispatch.run(...)`; [dispatch](dispatch.md) shows the
Python-shaped APIs. Explicit operation specs remain the advanced portable IR.
`PickledCallable` is only a same-Python convenience path.

## Runtime Guards

Use `dryml.runtime.require_allocation(...)` or `require_worker_allocation(...)` in workload code. Use `assert_no_workload_allocation(...)` in orchestrator/probe code.

Missing allocation errors point to the runtime layer because active allocation is process-local state. Unsatisfied CPU/GPU/memory constraints belong to `dryml.world.req(...)` and world compatibility checks. Missing package/software constraints belong to `dryml.env.req(...)` and environment compatibility checks.

## Related Docs

- [Annotations](annotations.md)
- [Dispatch](dispatch.md)
- [World/Runtime Split](world_runtime.md)
- [Code Analysis](architecture/code_analysis.md)
