# Legacy Context and Execute Migration

DRYML removed the old public context and process-execution packages. New code
uses environments, worlds, runtime, annotations, dispatch, and—only for advanced
control—the canonical `OperationSpec` IR.

## Migration Map

| From | To | Required distinction |
|---|---|---|
| Legacy context resource dictionaries | `dryml.world.req(...)`, `dryml.world.default(...)`, `WorldRequirement`, and `WorldSpec` | A requested world and this process's actual allocation are distinct. |
| Legacy execute/process helpers | `dryml.dispatch.run(function, ...)` or `run(cdef, "method", ...)` | Normal users pass functions or CDef plus method name. |
| Manual OperationSpec-first examples | Python-shaped dispatch | Explicit `OperationSpec` remains supported advanced IR. |
| `dryml.code` method-model imports | `dryml.core2.methods` | Compatibility aliases remain warning-free; Sprint 10 adds no deprecation. |
| Removed `dryml.graph` prototypes | `dryml.code` analyzers and `trace(...)` | No compatibility package exists; static possibilities and dynamic observations differ. |
| Notebook process assumptions | Current environment/world defaults and active runtime allocation | Setting defaults does not allocate resources. |
| Local execution with checks | `runtime.plain()` | Plain mode is inline enforcement-off execution, not dispatched isolation. |
| Non-importable callable assumptions | Explicit `allow_pickle=True` | Pickle transport remains same-Python-only. |

## Current Shape

```python
import dryml
from dryml.core2.store.dir import DirStore


@dryml.env.req(packages={"numpy": ">=1"})
@dryml.world.req(cpus={"min": 1})
def add(x, y):
    return x + y


store = DirStore("work/store", query_index="none")
result = dryml.dispatch.run(add, store=store, args=(2, 3))
```

Dispatch activates worker runtime before importing targets or materializing CDefs.
`runtime.plain()` is the deliberate alternative for trusted inline work. See
[dispatch](../dispatch.md), [world/runtime](../world_runtime.md), and the
[runnable examples](../../examples/dispatch/python_shaped_dispatch.py).
