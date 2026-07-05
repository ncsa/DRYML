# Legacy Runtime API Removal

DRYML removed the old public context and process-execution packages. New code should use environments, worlds, runtime guards, annotations, operation specs, and dispatch.

## Migration Map

| Old concept | New API |
|---|---|
| Context resource dictionaries | `dryml.world.req(...)`, `dryml.world.default(...)`, `dryml.worlds.WorldRequirement`, `dryml.worlds.WorldSpec` |
| Active process resource checks | `dryml.runtime.require_allocation(...)`, `require_worker_allocation(...)`, `assert_no_workload_allocation(...)` |
| Software/package requirements | `dryml.env.req(...)`, `dryml.environments.EnvironmentRequirement` |
| Process/local execution helpers | `dryml.dispatch.run(...)`, `Dispatcher.run(...)`, `Dispatcher.submit(...)` |
| Pickled callable process calls | Canonical `dryml.operations.make_function_call_spec(...)` or `make_method_call_spec(...)` |

## New Shape

```python
import dryml


@dryml.env.req(packages={"numpy": ">=1.26"})
@dryml.world.req(cpus={"min": 1})
@dryml.world.default(cpus=1)
@dryml.runtime.default(mode="worker")
def add(x, y):
    dryml.runtime.require_worker_allocation("add() runs in a worker")
    return x + y


operation = dryml.operations.attach_operation_id(
    dryml.operations.make_function_call_spec("my_project.tasks:add", args=[2, 3])
)
result = dryml.dispatch.run(operation, backend="local_subprocess")
```

World requirements are planning metadata. Runtime allocation is process-local state. Dispatch activates worker runtime before importing targets or materializing objects.
