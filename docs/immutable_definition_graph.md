# Immutable Definition Graph

DRYML now separates graph expression syntax from storage identity, query interpretation, generation, and runtime execution.

`Definition` is the immutable structural expression value. It deeply freezes user containers at construction and update boundaries, uses structural equality, and is hashable. Use `with_args`, `with_arg`, `with_kwargs`, `with_kwarg`, `without_kwarg`, and `at(path).set(value)` for copy-on-write updates.

`ConcreteDefinition` is the exact canonical materializable identity used by repos, stores, materialization, graph extraction, and stable hashes. It is separate from `Definition`.

`Ref(target)` marks a parent slot as a non-materializing graph edge. `Mat(target)` marks an explicit materializing edge. Raw nested `Definition` or `ConcreteDefinition` values are materializing children. At concrete identity boundaries, materializing links collapse to the raw child `ConcreteDefinition`, so there is only one canonical materializing representation. Materialization follows materializing edges only; constructors annotated with `RefCDef` receive the referenced `ConcreteDefinition` target, not graph edge state.

`Selector(root)` is the query interpretation of a `Definition`. `repo.query(defn)` lifts definitions to selectors, while `repo.query(selector)` uses the selector directly. `Definition.__eq__` is never selector matching; use `Selector(defn).matches(target)` for semantic matching.

`QuotedDef(defn)` and `SelectorSpec(selector)` store expressions as local constructor data. They do not emit object graph edges, which is the distinction from `Ref(selector_or_def)`.

`Par` placeholders carry a matcher and optionally a generator. Matchers such as `Present`, `Missing`, `AnyValue`, `Exact`, `Choice`, `IntRange`, `SubclassOf`, and `Satisfies` participate in selector verification. Unresolved `Par` values cannot be concretized.

`Ref(Selector(...))` and `Mat(Selector(...))` are query-only link patterns and cannot be concretized as stored object identity. Store selector expressions as data with `SelectorSpec` or `QuotedDef` instead.

Importable function values are symbolized at `Definition` boundaries. Anonymous functions/lambdas are rejected as raw values; use `Satisfies(predicate, name="stable-name")` for scan-only selector predicates. Anonymous `Satisfies` predicates can match in query-only selectors but are not stable-hashable, so selector-valued concrete data must use named or symbolically identifiable predicates. A `Satisfies` name is a semantic identity, not a display label; do not reuse one name for predicates with different behavior.

`SearchSpace` is the generative interpretation of a `Definition` containing generator-backed `Par` values such as `UniformIntRange` and `UniformFromSet`. It can sample definitions, enumerate finite grids, and produce a support selector.

```python
from dryml.core2 import Definition, Ref, Selector, UniformFromSet, UniformIntRange

model = Definition(Model, layers=UniformIntRange(2, 4), width=UniformFromSet([128, 256]))
space = model.as_space()
sample = space.sample()
support = space.support_selector()

artifact = Definition(Accuracy, model=sample.ref(), data=Ref(dataset_cdef))
selector = Selector(Definition(Accuracy, model=support.root.ref()))
```
