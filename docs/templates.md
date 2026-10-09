# Symbolic Definitions And Generation

`Definition` is DRYML's one immutable construction-description value, for both
resolved and symbolic authoring. It can contain `Par` and supported arithmetic or
repetition expressions without resolving a target, applying defaults, calling a
factory, materializing an Object, or saving anything. `Definition.names` reports
active roots in authored first-occurrence order. `Definition.is_resolved` means
only that no active expression remains; it does not prove that a constructor call
is complete or valid.

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
not alternate construction-value types.

| Receiving annotation | Accepts | Delivers | Active-expression rule |
| --- | --- | --- | --- |
| `Mat[Definition]` | materializing Definition structure | materialized Object | After activated roles establish quotation barriers, materializing structure must be resolved before CDef, Repo, target, cache, or Store effects. |
| `Ref[Definition]` | selected Definition | Definition | The selected authored Definition must be resolved; constructor completeness is not tested. |
| `Template` or `Template[Definition]` | symbolic or resolved Definition, including compatible `Ref(definition)` | Definition | Symbolic content is inert quotation data. |

Bare `Template` is exactly shorthand for `Template[Definition]`. It cannot be
instantiated, has no other subscription target, and is not a runtime recipe
value. Role admission is delayed until the annotated boundary is activated. An
authored outer Definition can therefore remain structurally unresolved because a
nested Template slot carries an active expression, while materialization of that
outer Definition succeeds because the nested recipe is quotation data. The same
outer Definition supplied to `Ref[Definition]` is rejected without inspecting the
outer constructor roles.

`Ref(definition)` records a lossless non-materializing assertion and does not
bypass the destination role's admission rule. `Definition.quote()` and
`QuotedDef` instead make a Definition explicit local expression data, not an
Object graph edge. Direct QuotedDef values are traversal barriers by default;
they are delivered as `QuotedDef` only through an explicit `Ref[QuotedDef]` role.

## Experiment Artifact Mappings

`Experiment.artifacts` accepts only `Mapping[str, Template] | None`, with
`Mapping[str, Template[Definition]]` as the equivalent long spelling. Each value
is admitted separately as a Template-role Definition. Keys must be nonempty
strings; lists, tuples, singleton recipes, nested mappings, non-Definition
values, and non-string keys fail before training effects. At the activated
Experiment boundary, entries are ordered by `canonical_key_bytes`; that canonical
order controls evaluation and checkpoint recovery. Definition and selector
authoring retain an artifact mapping inertly until later concretization.

## Exact Support And Persistence

`Definition.loose_selector()` replaces active symbolic expressions with local
wildcards while retaining known construction structure. It is not a class-only
selector: fixed nested `ConcreteDefinition` values remain exact definition
anchors, and fixed `ObjectRef`/`StateRef` values retain their identities. A
different dataset fork or a changed model initializer can therefore fall outside
the original template's loose selector. Use a partial `Definition` to constrain
only the experiment fields of interest, or
`Definition(Experiment, SKIP_ARGS)` to select all Experiment definitions.

Query counts depend on the selected identity domain. `.cdefs().count()` counts
distinct definitions, `.state_refs().count()` counts saved snapshots, and
`.state_refs().object_projection().count()` counts distinct realized object
graphs across those snapshots. Repeated runs of one definition can create
multiple object graphs; checkpoints can create multiple states of one object.

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

Symbolic Definitions and providers operate on trusted DRYML code and values.
They are not a sandbox or safe-deserialization boundary.

## Beta Removals

There is no runtime `Template` carrier and no `TemplateBundle`,
`TemplateGenerator`, or `TemplateSelector` API. There are no aliases, source or
pickle compatibility shims, or persisted-value migration for those beta values,
including previously persisted symbolic `Ref[Definition]` values. Re-author a
recipe as a Definition, bind fixed values with `Definition.sub(...)`, and use
`Generator` only for an explicit Distribution mapping. Compatible
`GeneratorSelector` payloads are the sole retained beta-format compatibility
commitment.
