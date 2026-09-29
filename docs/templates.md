# Definition Templates

`Template` is an inert, reusable Definition recipe. Construct it class-first:
`Template(Model, ...)` captures the supplied call without resolving or calling
`Model`. Use `Template.from_value(value)` for an existing Definition or a generic
list/tuple/container root, and `Definition.as_template()` for an existing soft
Definition. `Template(Definition(...))` is not a constructor overload.

```python
import random
from dryml.core import F, Par, Template, TemplateGenerator, UniformFromSet

class Model:
    def __init__(self, layers, width):
        self.layers = layers
        self.width = width

group = [F("builtins:tuple", Par("width")), F("builtins:tuple")]
model = Template(Model, group * Par("depth"), width=Par("width"))
bound = model.sub(width=64, depth=2)
definition = bound.to_definition()

generator = TemplateGenerator(
    model,
    width=UniformFromSet((32, 64)),
    depth=UniformFromSet((1, 2)),
)
sample = generator.sample(random.Random(7))
grid = generator.grid()
exact_selector = generator.support_selector()
loose_selector = model.as_selector()
```

`sub` accepts static supported values only. It never samples, invokes a factory,
constructs an Object, computes an Artifact, or saves state. Bindings are one pass:
parameters already present in the receiving template are replaced once, while
parameters introduced by a replacement Template require a later `.sub(...)`.
`TemplateGenerator` captures all static/domain bindings without sampling and its
`sample()` and `grid()` return complete `Definition` values. Missing roots or a
result that remains active raise `TemplateError` or `UnresolvedTemplateError`.

## Names And Expressions

`Par("encoder/width")` has the qualified binding root `encoder/width`.
`Par("this.model")` uses `this` as its root and `model` as its relative graph
path. Use `sub_dict` for qualified names that are not Python keywords, and use
`namespace` to qualify keyword bindings. Equal qualified roots deliberately share
one binding; composition does not create an implicit namespace.

```python
scaled = Template(Model, [], width=Par("encoder/width") * Par("scale"))
scaled = scaled.sub(sub_dict={"encoder/width": 64}, scale=2)
prefixed = scaled.remap(prefix="experiment")
restored = prefixed.remap(strip="experiment")
```

`*`, `/`, and `//` retain Python arithmetic semantics after every operand is
bound. They reject unsupported operands and arithmetic failures rather than
coercing or rounding values. List and tuple groups may be repeated with a static
or symbolic count. Expansion is bounded; invalid or excessive counts raise
`TemplateLimitError` rather than producing a completed definition.

```python
from dryml.core import Shared, repeat

independent = Template.from_value(repeat([Par("width")], Par("depth")))
shared = Template.from_value([Par("width")] * Shared(Par("depth")))
```

Ordinary repetition freshens DRYML construction-node identity for every copy while
retaining parameter names, so copies have linked configuration but independent
nodes. `Shared(count)` retains corresponding DRYML graph nodes. This is distinct
from two structurally equal nodes, which are never deduplicated automatically.
Neither policy requires TensorFlow or PyTorch factory consumers to reuse a backend
layer/module instance: backend weight tying remains author-owned.

## Selection, References, And Persistence

`Template.as_selector()` is a loose ordinary `Selector`: it retains known concrete
structure and fixed sequence positions but drops unknown roots, arithmetic
relationships, and sharing topology. `TemplateGenerator.support_selector()` is
the exact alternative: it checks captured domains, linked names, arithmetic,
ordering, and graph topology. Bounds from custom domains can reject candidates but
never prove membership; unavailable exact verification raises
`UnsupportedTemplateVerificationError`.

An unresolved recipe can be carried only by a constructor slot declared
`Ref[Template]`. That recipe is opaque to outer `sub` and `remap` calls by default;
pass `traverse_refs=True` to enter pre-existing carried recipes. Ref opacity does
not make undeclared slots valid and does not cause recursive same-call binding.

`TemplateBundle` carries an ordered named collection of inert recipes for an
artifact owner. It accepts one `Template`, a list/tuple (named `artifact_0`,
`artifact_1`, and so on), or a string-keyed mapping whose insertion order and
punctuation-bearing nonempty names are retained. Bundles are accepted only through
`Ref[TemplateBundle]`, encode all recipes in one bounded closed payload, and have
the same default Ref traversal barrier as `Template`.

The same boundary applies to recursive `object_projection()`: ordinary Ref-held
`StateRef` values are weakened to their `ObjectRef` associations, while a
Ref-held Template stays unchanged unless `traverse_refs=True`. Opting in retains
the recipe's `Par` expressions and only weakens supported exact references; it
does not resolve or construct the recipe.

```python
from dryml.core import Definition, Object, Ref

class RecipeConsumer(Object):
    def __init__(self, recipe: Ref[Template]):
        self.recipe = recipe

consumer = Definition(RecipeConsumer, recipe=model).concretize()
```

The unresolved `model` recipe remains data while `RecipeConsumer` is
materialized, saved, restored, and encoded. Its target is not constructed, and
the consumer can later call `.sub(...)` on the received recipe explicitly.
Template and built-in-domain selector data use closed portable codecs. Runtime
providers and `TemplateGenerator` itself are not portable.

Migration is direct: replace `Definition(...).as_space()` with a class-first
`Template(...)` or `.as_template()`, then capture bindings in `TemplateGenerator`.
`SearchSpace`, `space`, `space_mode`, old predicate-bearing `Par`, and the old
generator classes have no aliases. Predicate helpers such as `Exact` and
`IntRange` now return independent `Match` leaves for ordinary selectors; use
`Par(name)` for template bindings and `UniformIntRange`/`UniformFromSet` domains
with a generator.

Templates operate on trusted DRYML code and values and are not a sandbox or safe
deserialization boundary. Provider diagnostic disclosure and pre-parse payload
ingestion hardening remain deferred security work; this documentation does not
claim those limitations are fixed.
