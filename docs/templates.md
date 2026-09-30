# Symbolic Definitions And Generation

`Definition` is DRYML's immutable construction-description value. It can contain
`Par` and supported arithmetic or repetition expressions without resolving a
target, applying defaults, calling a factory, materializing an Object, or saving
anything. `Definition.names` reports active roots in authored first-occurrence
order. `Definition.is_resolved` means only that no active expression remains; it
does not prove that a constructor call is complete or valid.

Use `Definition.sub(...)` for static values. Substitution is immutable and one
pass: it evaluates expressions made closed by the supplied values, but never
samples a provider. A `Distribution` is invalid anywhere in Definition-owned
structure, including nested containers.

```python
from dryml.core import Definition, Generator, Par, UniformFromSet

class Model:
    def __init__(self, width, depth):
        self.width = width
        self.depth = depth

authored = Definition(Model, width=Par("width"), depth=Par("depth"))
fixed_depth = authored.sub(depth=2)
assert fixed_depth.names == ("width",)

generator = Generator(
    fixed_depth,
    {"width": UniformFromSet((32, 64))},
)
sample = generator.sample()
grid = generator.grid()
support = generator.support_selector()
```

`Generator(definition, distributions)` accepts exactly one Distribution mapping
that covers every active selected root. Static values, missing roots, unknown
roots, and sampled results that leave expressions active fail. Bind static
values with `Definition.sub(...)` before creating a Generator. Providers are
stored by sorted fully-qualified root name, preserving deterministic RNG draw
and grid order. Generator construction does not sample, resolve targets,
construct Objects, or persist providers. `sample()` and `grid()` return only
resolved Definitions and are all-or-error at their provider boundary.

## Receiving Roles

`Mat[Definition]`, `Ref[Definition]`, and bare `Template` are receiving roles,
not alternate construction-value types. `Mat[Definition]` admits resolved
materializing structure and delivers an Object. `Ref[Definition]` requires a
symbolically resolved Definition and delivers a Definition without
materialization. Bare `Template` is shorthand for `Template[Definition]`; it
accepts a symbolic or resolved Definition as inert quotation data and delivers
a Definition. `Template` cannot be instantiated and has no other subscription
target.

Role admission is delayed until the annotated boundary is activated. This is
why an authored Definition may remain structurally unresolved when an active
expression is safely carried by a nested Template slot, while a `Ref[Definition]`
boundary rejects the same authored active expression.

`Ref(definition)` records a non-materializing assertion and does not bypass the
destination role's admission rule. Use `Definition.quote()` or `QuotedDef` when
a Definition must be explicit local expression data rather than a graph edge.
Direct `QuotedDef` values remain symbolic traversal barriers by default.

`Experiment.artifacts` uses the narrow `Mapping[str, Template] | None` form.
Each nonempty string key names one Definition recipe; singleton recipes,
sequences, nested mappings, and runtime Template values are invalid. At the
activated Experiment boundary entries are independently quoted and ordered by
`canonical_key_bytes`; that order controls evaluation and checkpoint recovery.
Definition and selector authoring retain the mapping inertly until that boundary
is concretized, so recipe targets are not resolved during authoring.

## Exact Support And Persistence

`Generator.support_selector()` returns `GeneratorSelector`, which proves exact
finite support including linked roots, arithmetic, ordering, and construction
graph topology. Conservative provider bounds may reject a candidate but never
prove membership. An unavailable finite proof raises
`UnsupportedGeneratorVerificationError`; bounded grid and proof operations
raise `ParameterizationLimitError` rather than returning partial results.

Only immutable built-in distributions are portable selector data. A
`GeneratorSelector` keeps the established `dryml-template` v1
`template-selector` payload tag and canonical bytes for compatible definitions;
it decodes directly to Definition-based state. A Generator and arbitrary runtime
providers are intentionally nonportable. GeneratorSelector works with Repo
queries and save routing while retaining exact witness verification.

Templates and providers operate on trusted DRYML code and values. They are not a
sandbox or safe-deserialization boundary.
