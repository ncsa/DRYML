# Ref Edges And Selector Values

Use `Ref(target)` for a non-materializing parent-slot edge. The target remains queryable graph metadata, but materialization does not load or build it as a dependency.

```python
metric = ConfusionMatrix(
    experiment.train.result,
    test_cache.compute.result,
    labels=("cat", "dog"),
)
```

`ManagedOutputRef` is a lightweight Object that applies this rule to one
producer/method/output slot. Its producer edge is non-materializing, and its
identity contains no Store, realization, record, or representation ID.

Constructor roles can request this behavior with `RefCDef` or `RefCDefArg`. Runtime constructors receive the referenced `ConcreteDefinition` target, not the parent-slot graph wrapper:

```python
class Evaluation(Object):
    def __init__(self, data: dryml.RefCDef, model: dryml.RefCDef):
        self.data = data
        self.model = model
```

Use `SelectorSpec(selector)` or `Definition.quote()` / `QuotedDef(defn)` when an expression is constructor data rather than an object edge. Constructors annotated with `SelectorArg` receive the storage wrapper (`SelectorSpec` or `QuotedDef`) so they can distinguish selector-as-data from a materialized runtime object.

```python
class Plot(Object):
    def __init__(self, models: dryml.SelectorArg):
        self.models = models

plot = Definition(Plot, Selector(Definition(Model, layers=dryml.Present())))
```

Raw nested `ConcreteDefinition` values remain `materialize` edges. `Ref(ConcreteDefinition)` values create `ref` edges. Materialization, save, collect, and default nested traversal follow materialization edges only.
