# Frozen Definitions

Frozen definitions let an object record an exact definition reference without making that reference part of the object's materialization graph.

Use this for metrics, reports, provenance records, cache entries, and artifact-like objects that should remember heavy inputs without loading them when the owner loads.

## API

```python
frozen_cdef = obj.definition.freeze()
assert frozen_cdef.thaw() == obj.definition

frozen_selector = Definition(MyModel, family="small").freeze()
selector = frozen_selector.thaw()
```

`dryml.freeze(value)` accepts an `Object`, `ConcreteDefinition`, `Definition`, or already-frozen wrapper.

## Argument Roles

Class authors can declare non-materializing constructor arguments with annotations:

```python
class Accuracy(dryml.core2.Object):
    def __init__(self, data: dryml.FrozenCDef, model: dryml.FrozenCDef, value=None):
        super().__init__()
        self.data = data
        self.model = model
        self.value = value
```

Users still pass ordinary objects:

```python
acc = Accuracy(data=test_data, model=model, value=0.92)
```

The canonical definition stores `FrozenConcreteDefinition` values, but the constructor receives `ConcreteDefinition` values. Loading `acc` later does not load `test_data` or `model`.

Classes that cannot use annotations can declare the same roles with `__dryml_arg_roles__`:

```python
class Accuracy(dryml.core2.Object):
    __dryml_arg_roles__ = {"data": "frozen_cdef", "model": "frozen_cdef"}
```

Selector-valued arguments use `FrozenDef`:

```python
class Plot(dryml.core2.Object):
    def __init__(self, models: dryml.FrozenDef):
        super().__init__()
        self.models = models
```

The canonical definition stores `FrozenDefinition`; the constructor receives an independent mutable `Definition` copy.

## Edge Semantics

Raw `ConcreteDefinition` values remain materialization edges. `FrozenConcreteDefinition` values create frozen edges.

Materialization, save, collect, and default nested traversal follow materialization edges only. Query indexes store edge kind, and frozen-reference selectors match frozen edges without matching materialization edges.

SQLite query indexes rebuild because the edge schema version changed to include `edge_kind`.
