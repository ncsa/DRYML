# Frozen Definitions

Frozen definitions let an object record an exact definition reference without making that reference part of the object's materialization graph.

Use this for metrics, reports, provenance records, cache entries, and artifact-like objects that should remember heavy inputs without loading them when the owner loads.

## API

`ConcreteDefinition.freeze()` returns a `FrozenConcreteDefinition`, and `FrozenConcreteDefinition.thaw()` returns the target `ConcreteDefinition`.

`Definition.freeze()` returns a `FrozenDefinition` selector snapshot without calling `Definition.concretize()`. `FrozenDefinition.thaw()` returns a fresh mutable `Definition` copy.

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

Raw `ConcreteDefinition` values remain `materialize` edges. `FrozenConcreteDefinition` values create `frozen` edges.

Materialization, save, collect, and default nested traversal follow materialization edges only. Query indexes store edge kind, and frozen-reference selectors match frozen edges without matching materialization edges.

Structural frozen-reference search is available by querying with the same frozen role shape as the stored owner, for example `repo.query(Accuracy(data=data_obj, model=model_obj)).stored().defs()`. Public reverse-reference helpers such as `repo.query_references(...)` and public `nested(edge_kinds=...)` controls are deferred.

For import safety, automatic role resolution applies to live class selectors. Serialized selectors whose class is only an import reference should contain explicit frozen wrappers already, for example `Definition(AccuracyRef, model=model_cdef.freeze())`, so definition-only query planning does not import backend modules just to rediscover annotations.

SQLite query indexes rebuild because the edge schema version changed to include `edge_kind`.
