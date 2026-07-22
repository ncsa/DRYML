# Legacy Context and Execute Migration

DRYML removed the old public context and process-execution packages. New code
uses environments, worlds, runtime, annotations, dispatch, and—only for advanced
control—the canonical `OperationSpec` IR.

## Migration Map

| From | To | Required distinction |
|---|---|---|
| Legacy context resource dictionaries | [`dryml.world.req(...)`, `dryml.world.default(...)`, `WorldRequirement`, and `WorldSpec`](../world_runtime.md#boundaries) | A requested world and this process's actual allocation are distinct. |
| Legacy execute/process helpers | [`dryml.dispatch.run(function, ...)` or `run(cdef, "method", ...)`](../dispatch.md#python-shaped-dispatch) | Normal users pass functions or CDef plus method name. |
| Manual OperationSpec-first examples | [Python-shaped dispatch](../dispatch.md#python-shaped-dispatch) | Explicit [`OperationSpec`](../operations.md) remains supported advanced IR. |
| `dryml.code` method-model imports | [`dryml.core2.methods`](../architecture/code_analysis.md#relationship-to-method-and-method-handles) | Compatibility aliases remain warning-free; no deprecation is issued. |
| Removed `dryml.graph` prototypes | [`dryml.code` analyzers and `trace(...)`](../architecture/code_analysis.md#dynamic-trace-contract) | [No compatibility package exists](../dispatch.md#unsupported-graph-prototype-package); static possibilities and dynamic observations differ. |
| Notebook process assumptions | [Current environment/world defaults and active runtime allocation](../world_runtime.md#requested-defaults-allocation-and-plain-mode) | Setting defaults does not allocate resources. |
| Local execution with checks | [`runtime.plain()`](../world_runtime.md#requested-defaults-allocation-and-plain-mode) | Plain mode is inline enforcement-off execution, not dispatched isolation. |
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
from dryml.core2.store.dir import DirStore


from my_package.tasks import add


store = DirStore("work/store", query_index="none")
result = dryml.dispatch.run(add, store=store, args=(2, 3))
```

`add` is module-level and importable, so this default portable path does not use
pickle transport. A function defined in a running script or notebook has module
`__main__` and must instead pass `allow_pickle=True`; that explicit transport is
same-Python-only.

Dispatch activates worker runtime before importing targets or materializing CDefs.
`runtime.plain()` is the deliberate alternative for trusted inline work. See
[dispatch](../dispatch.md), [world/runtime](../world_runtime.md), and the
[runnable examples](../../examples/dispatch/python_shaped_dispatch.py).
