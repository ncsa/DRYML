# Symbolic Definitions And Generation

`Definition(...)` authors an immutable `Node` construction-description graph, for
both resolved and symbolic authoring. It can contain `Par` and supported arithmetic or
repetition expressions without resolving a target, applying defaults, calling a
factory, materializing an Object, or saving anything. `Definition.names` reports
active roots in authored first-occurrence order. `Definition.is_resolved` means
only that no active expression remains; it does not prove that a constructor call
is complete or valid.

Use `Definition.sub(...)` for static values. Substitution is immutable and one
pass: it evaluates expressions made closed by the supplied values, but never
samples a provider. A `Distribution` is invalid anywhere in Definition-owned
structure, including nested containers.
Dataset or model Definition/CDef values supplied through bindings retain their
construction-value types rather than becoming mappings. Exact ObjectRef/StateRef
bindings retain their identities, while live Object bindings lower to ObjectRefs
without materializing or saving their graphs.
Substitution, remapping, and generation retain fixed node-local ObjectId and
LocalState selections, sharing, and Ref/quotation boundaries. Independent equal
Nodes remain independent. Changing a recipe never silently discards a selected
state: `LocalState.validate_for(...)` rejects incompatible configuration. Explicit
`Node.build()` now preflights selected local state before realization (U11).
The query semantic compiler is available independently; format-4 query source
inventory and selective lowering still await U13.

Node handle `==` is occurrence/view identity. Use `graph_equal(...)` to compare
exact graphs, including facets and alias partitions, or `recipe_equal(...)` for
recipe-only comparison.

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
construct Objects, or persist providers. `sample()` and `grid()` return resolved
Node descriptions and are all-or-error at their provider boundary. They do not
fill omitted defaults, allocate identities, or capture state. A ConcreteNode can
be supplied as a fixed zero-root generator and is returned unchanged. Explicit
`concretize()` binds construction defaults without allocating ObjectIds; the
original query/generation description retains its omissions.

## Authoring Helpers

`dryml.core.authoring_helper` (also `dryml.authoring_helper`) shares the call
binding and authoring/concrete branching for graph-producing helpers. A helper
still supplies its own explicit graph builder; the decorator does not analyze or
lift an arbitrary Python function body into a Definition.

```python
from dryml.core import Par, authoring_helper
from dryml.data import Scale

def scale_definition(arguments):
    return Scale.defn(arguments["factor"])

@authoring_helper(author_definition=scale_definition)
def scale_method(factor=1):
    return Scale(factor)

recipe = scale_method(Par("factor"))
bound_recipe = recipe.sub(factor=2)
```

Every call first binds the original Python signature and includes its defaults.
Policies receive a read-only named argument mapping, with `*args` and `**kwargs`
represented by their named tuple/dictionary entries. Missing, duplicate, or
unexpected arguments fail before policies run. These are the supported controls:

| Control | Responsibility |
| --- | --- |
| `author_definition` | Required callback returning an inert Node (Definition or ConcreteNode) from the bound mapping. The exact returned handle and its facets are retained. It must not construct or materialize the described graph. |
| `should_author` | Optional exact-bool predicate on the mapping. The default detects soft Definitions and Expr values through supported containers, factory arguments, and Mat links, while preserving Ref/quotation barriers. Exact references alone do not select authoring. |
| `validate_known_arguments` | Optional authoring-only validator called before the builder. Reject known literal errors; defer expression-dependent checks to graph binding. |
| `normalize_concrete` | Exact bool, default false. Opt into core `function` argument/return normalization on the concrete path only. |

The symbolic branch bypasses both the concrete target and its optional signature
normalization boundary. Concrete calls retain the original call spelling and
target behavior. The decorated helper preserves its name, documentation, and
Python signature; workload keywords are never consumed as decorator controls.
Invalid predicate/builder results fail explicitly, and policy or target failures
never fall back to the other branch. The mapping is shallowly read-only, not a
copy or security boundary for its values. All callbacks are trusted and own
their side effects; the wrapper keeps no per-call mutable state shared between
invocations.

`as_supervised` uses a source-specific predicate that includes CDef and saved
reference authority, so those sources return a Map Definition while live Datasets
return live Maps. Metric helpers use the default detector and enable concrete
signature normalization, preserving uncomputed Fold returns for concrete-only
calls. Their domain-specific builders and validation remain in data/metrics;
core does not import those consumer packages.

## Receiving Roles

`Mat[T]`, `Ref[Node]`, and bare `Template` are receiving roles,
not alternate construction-value types.

| Receiving annotation | Accepts | Delivers | Active-expression rule |
| --- | --- | --- | --- |
| `Mat[T]`, with T an Object class | compatible live Object or construction description | live T | After activated roles establish quotation barriers, active construction must be resolved before materialization effects. Legacy `Mat[Definition]`/`Mat[ConcreteDefinition]` remain available. |
| `Ref[Node]` | selected Node metadata | Node | Symbolically resolved; no defaults or constructor-completeness check. |
| `Ref[ConcreteNode]` | normalized Node metadata | same ConcreteNode | Requires normalization already present; no implicit concretization. |
| `Ref[SavedNode]` | normalized, saved-complete Node metadata | same ConcreteNode handle | Uses `require_saved`; neither capture nor availability is implied. |
| `Template` or `Template[Node]` | symbolic or resolved Node, including compatible `Ref(node)` | same Node | Future recipe is inert quotation data. |

Bare `Template` is shorthand for `Template[Node]`; `Template[Definition]` is
retained as an equivalent authoring spelling. It cannot be instantiated, has no
other subscription target, and is not a runtime recipe
value. Role admission is delayed until the annotated boundary is activated. An
authored outer Definition can therefore remain structurally unresolved because a
nested Template slot carries an active expression, while materialization of that
outer Definition succeeds because the nested recipe is quotation data. The same
outer Definition supplied to `Ref[Node]` (or retained `Ref[Definition]`) is rejected without inspecting the
outer constructor roles.

`Ref(definition)` records a lossless non-materializing assertion and does not
bypass the destination role's admission rule. `Definition.quote()` and
`QuotedDef` instead make a Definition explicit local expression data, not an
Object graph edge. Direct QuotedDef values are traversal barriers by default;
they are delivered as `QuotedDef` only through an explicit `Ref[QuotedDef]` role.

## Experiment Artifact Mappings

`Experiment.artifacts` accepts only `Mapping[str, Template] | None`, with
`Mapping[str, Template[Node]]` as the equivalent long spelling; the Definition
spelling is also retained. Each value is admitted separately as a Template-role
Node. Keys must be nonempty
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

The retained V3 query contract counts by the selected identity domain. `.cdefs().count()` counts
distinct definitions, `.state_refs().count()` counts saved snapshots, and
`.state_refs().object_projection().count()` counts distinct realized object
graphs across those snapshots. Repeated runs of one definition can create
multiple object graphs; checkpoints can create multiple states of one object.

`Generator.support_selector()` returns `GeneratorSelector`, which proves exact
finite support including linked roots, arithmetic, ordering, fixed identity/state
facets, and construction graph topology. Graph-valued choices use bounded finite
witness enumeration rather than process-local handle membership. Conservative provider bounds may reject a candidate but never
prove membership. An unavailable finite proof raises
`UnsupportedGeneratorVerificationError`; bounded grid and proof operations
raise `ParameterizationLimitError` rather than returning partial results.

`QueryPredicate.support(generator, max_assignments=...)` captures this same
exact proof as an explicit predicate. Compose it with Node patterns, `&`, `|`,
`~`, and `QueryPredicate.absent("state" | "identity", path=...)`. Ordinary Node
patterns need no mandatory Selector wrapper. Matching retains supplied facets,
alias partitions, Mat/Ref roles and exact quoted configuration; it never binds
omitted defaults. A loose range/prefilter false positive must still pass the
correlated assignment witness. Unsupported runtime-provider proofs and exhausted
caps remain visible errors, including inside boolean composition.

Explicit predicate persistence uses the new `dryml-node-query-predicate` v1
family, embedding the existing authenticated Node codec rather than an opaque
provider payload. Built-in range and ordered-choice domains remain typed;
arbitrary runtime providers are refused. The full predicate and its graph-valued
choices share a Node table. These semantic APIs work against detached metadata;
they do not claim format-4 Repo/Store query-source integration.

Only immutable built-in distributions are portable selector data. A
`GeneratorSelector` keeps the established `dryml-template` v1
`template-selector` payload tag and canonical bytes for compatible definitions.
Descriptions requiring Node-only evidence explicitly select the separate
`dryml-node-template` v1 family, retaining that kind and embedding one authenticated
`dryml-node-graph/v1` packet. The description and domain choices share one table,
so cross-choice aliases, facets, roles, and retained construction evidence survive
decoding. Node expression quotations use this same family when needed; compatible
legacy expressions retain `dryml-definition-expression` v1. Both families have
bounded, closed, canonical decoding. Incompatible versions and malformed packets
fail without rewriting the input; legacy readers do not reinterpret new facets.
A Generator and arbitrary runtime
providers are intentionally nonportable. The legacy GeneratorSelector adapter
retains exact witness verification; its historical Repo query/save-routing
consumers do not establish format-4 query support.

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
