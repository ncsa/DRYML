# Legacy Context and Execute Migration

DRYML removed the old public context and process-execution packages. New code
uses `dryml.session`, environments, worlds, runtime, annotations, dispatch, and—only for advanced
control—the canonical `OperationSpec` IR.

## Migration Map

| From | To | Required distinction |
|---|---|---|
| Legacy context resource dictionaries | [`dryml.world.req(...)`, `dryml.world.default(...)`, `WorldRequirement`, and `WorldSpec`](../world_runtime.md#boundaries) | A requested world and this process's actual allocation are distinct. |
| Common notebook runtime setup | [`dryml.session`](../session.md) | Use `manage()` for the current process and `worker_world_request()` to set the default world for later workers; no role or context-manager lifetime is needed. |
| Legacy execute/process helpers | [`dryml.dispatch.run(function, ...)` or `run(cdef, "method", ...)`](../dispatch.md#python-shaped-dispatch) | Normal users pass functions or CDef plus method name. |
| Manual OperationSpec-first examples | [Python-shaped dispatch](../dispatch.md#python-shaped-dispatch) | Explicit [`OperationSpec`](../operations.md) remains supported advanced IR. |
| `dryml.code` method-model imports | [`dryml.core.methods`](../architecture/code_analysis.md#relationship-to-method-and-method-handles) | Compatibility aliases remain warning-free; no deprecation is issued. |
| Removed `dryml.graph` prototypes | [`dryml.code` analyzers and `trace(...)`](../architecture/code_analysis.md#dynamic-trace-contract) | [No compatibility package exists](../dispatch.md#unsupported-graph-prototype-package); static possibilities and dynamic observations differ. |
| Notebook process assumptions | [Session, requested defaults, and active runtime allocation](../world_runtime.md#session-requested-defaults-allocation-and-plain-mode) | Setting a worker request does not replace the current-process allowance. |
| Local execution with checks | [`runtime.plain()`](../world_runtime.md#session-requested-defaults-allocation-and-plain-mode) | Plain mode is inline enforcement-off execution, not dispatched isolation. |
| Non-importable callable assumptions | [Explicit `allow_pickle=True`](../dispatch.md#python-shaped-dispatch) | Pickle transport remains same-Python-only. |

## Current Shape

Create an importable module:

```python
# my_package/tasks.py
def add(x, y):
    return x + y
```

Then dispatch it from another module or notebook:

```python
import dryml
from dryml.core.store.dir import DirStore


from my_package.tasks import add


store = DirStore("work/store", query_index="none")
result = dryml.dispatch.run(add, store=store, args=(2, 3))
```

`add` is module-level and importable, so this default portable path does not use
pickle transport. A function defined in a running script or notebook has module
`__main__` and must instead pass `allow_pickle=True`; that explicit transport is
same-Python-only.

Dispatch activates worker runtime before importing targets or materializing CDefs.
`runtime.plain()` is the advanced deliberate alternative for trusted inline work.
Fresh session behavior is intentionally ordinary unchecked Python; see the
[session-default migration](session_runtime_default.md), [dispatch](../dispatch.md), [world/runtime](../world_runtime.md), and the
[runnable examples](../../examples/dispatch/python_shaped_dispatch.py).
