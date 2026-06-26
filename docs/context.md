# Contexts

Status: draft.

DRYML contexts describe runtime execution constraints. They help code declare and check requirements such as backend compatibility, CPU/GPU resources, memory, and process bootstrap settings.

## Core Functions

The public context API includes:

- `active_context()`
- `use_context(...)`
- `set_context(...)`
- `add_context(...)`
- `clear_context()`
- `check_context(...)`

Typical use:

```python
from dryml.context import use_context, check_context

with use_context({"plain": {"num_cpus": 1}}):
    check_context({"plain": {"num_cpus": 1}})
```

## Resource Specs

`ResourceSpec` normalizes resource requests. Requests can include CPUs, GPUs, specific device resources, and memory.

The resource pool tracks available resources and allocates them to active contexts. Failed allocation raises `InsufficientResourcesError`.

## Context Containers

The active context is represented as a context container. It can hold one or more backend-specific compute contexts, such as plain Python, TensorFlow, PyTorch, or JAX contexts.

Contexts can be added, replaced, and cleared. `use_context(...)` is the normal scoped interface.

## Backend Contexts

DRYML includes backend context classes for:

- plain execution
- TensorFlow
- PyTorch
- JAX

Backend contexts can validate the current runtime and apply best-effort runtime effects. For example, a backend context may check whether the runtime is compatible before executing backend-specific code.

## Context Checks

`check_context(...)` verifies that objects or object graphs can run in the active context. This is intended to catch incompatible execution environments before work starts.

Objects can contribute context requirements. The checker combines requirements and compares them with the active context.

## Exceptions

Important exceptions:

- `ContextError`: base context exception.
- `NoContextError`: no active context was available.
- `WrongContextError`: active context does not match required context.
- `InsufficientResourcesError`: requested resources could not be allocated.
- `ContextAlreadyActiveError`: an already-active context was activated again.
- `ContextIncompatibilityError`: runtime/backend incompatibility.
- `ContextBootstrapError`: failure while preparing process bootstrap state.

## Example Pattern

```python
from dryml.context import use_context

requirements = {
    "plain": {
        "num_cpus": 2,
        "num_gpus": 0,
    }
}

with use_context(requirements):
    # Build, load, train, or evaluate objects here.
    pass
```

## Bootstrap Notes

Contexts can provide environment information for workers. This matters for execution systems where child processes need environment variables, resource assignments, or backend-specific initialization.

## Common Pitfalls

- Do not assume a GPU exists just because a backend library is installed.
- Keep context scopes short and explicit.
- Release contexts with context managers instead of manually managing global state when possible.
- Treat backend-specific runtime effects as process-local.

## Related Docs

- [Tensor Specs](tensor_specs.md)
- [Models API](models.md)
