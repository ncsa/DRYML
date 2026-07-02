# Ref Edges And Selector Values

Use `Ref(target)` for a non-materializing parent-slot edge. The target remains queryable graph metadata, but materialization does not load or build it as a dependency.

```python
metric = Definition(Accuracy, data=Ref(test_data.definition), model=Ref(model.definition))
```

Constructor roles can request this behavior with `RefCDef` or `RefCDefArg`. Runtime constructors receive the referenced `ConcreteDefinition` target, not the parent-slot graph wrapper:

```python
class Accuracy(Object):
    def __init__(self, data: dryml.RefCDef, model: dryml.RefCDef, value=None):
        self.data = data
        self.model = model
        self.value = value
```

Use `SelectorSpec(selector)` or `Definition.quote()` / `QuotedDef(defn)` when an expression is constructor data rather than an object edge. Constructors annotated with `SelectorArg` receive the storage wrapper (`SelectorSpec` or `QuotedDef`) so they can distinguish selector-as-data from a materialized runtime object.

```python
class Plot(Object):
    def __init__(self, models: dryml.SelectorArg):
        self.models = models

plot = Definition(Plot, Selector(Definition(Model, layers=dryml.Present())))
```

Raw nested `ConcreteDefinition` values remain `materialize` edges. `Ref(ConcreteDefinition)` values create `ref` edges. Materialization, save, collect, and default nested traversal follow materialization edges only.
