# Signatures

DRYML has one explicit signature-normalization boundary. Import `Ref`, `Mat`,
`AutoRef`, `normalize_args`, `normalize_return`, `signature_context`, `function`,
and `SignatureError` from `dryml` or `dryml.core`. Advanced
`SignaturePlan`, `BoundaryPlan`, and `compile_signature` are available only from
`dryml.core.signatures` for integrations that own a call boundary.

Annotations are inert on ordinary Python calls. Use `@function`, a direct helper,
or an integrating DRYML boundary such as construction, `Method`, or a managed
operation to activate them.

## Conversion

`Ref[T]` selects and delivers reference or quotation data. `Mat[T]` selects the
same authority but asks the caller-provided Repo to realize it. `Ref(value)` and
`Mat(value)` are assertions, not conversion requests: an opposing assertion fails
before selection. An unannotated slot is `Mat` by default. Scalars and a directly
supplied live `Object` continue to pass through naturally, while an unannotated
`Definition`, `ConcreteDefinition`, `ObjectRef`, or `StateRef` is materialized.

Supported exact targets are `Definition`, `ConcreteDefinition`, `ObjectRef`,
`StateRef`, `Object`, `QuotedDef`, `Selector`, and `SelectorSpec` where applicable.
Exact requests deliver that exact authority: a `StateRef` supplied to
`Ref[ObjectRef]` delivers its `ObjectRef`, for example. `Ref[AutoRef]` retains an
incoming reference kind; for a live Object it selects CDef for an entirely
stateless materializing graph, its known last `StateRef` for a stateful graph, or
its `ObjectRef` otherwise. Automatic selection never saves or searches for a new
receipt. `AutoRef` is an annotation-only marker class; use `Ref[AutoRef]`, not
`Ref[AutoRef()]`, on every supported Python version.

Same-role unions such as `Ref[ConcreteDefinition | ObjectRef | StateRef]` select
the strictest reachable authority independent of member order. Ambiguous
strengthening fails instead of falling back. Nullable forms are flat:
`Ref[ObjectRef | None]` and `Ref[ObjectRef] | None` accept a plain `None`;
distributed forms such as `Ref[ConcreteDefinition] | Ref[ObjectRef] | None` are
equivalent. Only plain `None` may appear outside role branches, and opposing value
wrappers still conflict before the nullable branch is considered. `Ref[None]` does
not compile. Ref/Mat forms are top-level only. Nested roles such as
`list[Ref[ObjectRef]]`, container targets, mixed Ref/Mat unions, incomparable data
unions, and Object-subclass annotations fail with `SignatureError`.

`QuotedDef` and `SelectorSpec` are expression data. Use `Ref[QuotedDef]`,
`Ref[Selector]`, or `Ref[SelectorSpec]` to wrap, unwrap, or deliver them without
constructing a target. `Mat` rejects quoted input until the caller explicitly
unwraps it. Constructor CDefs retain quotation data inertly, while a constructor
declaring `Ref[Definition]` or `Ref[Selector]` receives the exact unwrapped type at
runtime. Persisted reconstruction does not rerun preparation, binding, or signature
interpretation. Query indexing and matching treat these constructor markers as their
quotation payloads rather than as graph boundaries.

A fresh constructor with an explicit Ref/Mat role validates finalized structural
links against that role before canonicalization can erase their edge authority.
Opposing links fail, while compatible links normalize to the declared recipient
type. Unannotated constructor slots continue to accept finalized canonical links as
intentional structure, and persisted or already-bound replay remains inert.

A role on `*items` or `**extra` applies independently to every expanded occurrence;
the packed tuple/mapping in `BoundaryPlan.authority` and ordinary Python call
projection are preserved. Selection controls name ordinary slots with a string,
the return with `"return"`, positional variadic occurrences with `("items", 0)`,
and keyword variadic occurrences with `("extra", "key")`. Parameter-level or
unbound variadic paths, malformed selection values, and unknown paths raise before
authority reads or realization.

## Authority And Effects

Definition-to-CDef reference strengthening requires an already fully bound
expression; it cannot fill defaults or run a search. CDef-to-ObjectRef selection
uses matching authoritative declarations even when their claims are `available`,
`claimed`, or `completed`. That declaration eligibility does not grant construction:
`Mat` separately requires an eligible live object, saved state, or active claim.
When `ReferenceSelection` pins a declaration Store for a Mat slot, delivery requires
that exact handle to remain connected with matching declaration and ClaimRecord
authority. Repo does not substitute another replica; conflicting Store domains for
the same selected identity fail before realization effects. Duplicate connected
handles in one authority-fence domain are treated as the same Store domain.

`Mat[StateRef]` submits the selected exact snapshot to Repo. Repo owns claim
admission, reservations, caching, restoration, and cleanup. A direct live Object
is preserved for ordinary `Mat[Object]` delivery. Under Repo's existing
`reuse_live="matching"` or `"greedy"` policy, a checkpoint-marked live object
mutated after that checkpoint can be reused without restoration; use
`reuse_live="never"` when a fresh load is required. This accepted limitation does
not permit selecting another identity or snapshot.

Aggregate ObjectRef admission classifies nested dependencies before claims are
acquired. A nested identity already selected as live or from saved state is reused
during parent construction rather than reclaimed or reconstructed. Unresolved
nested declarations use their preflighted claim and any explicitly pinned
declaration Store.

Selection, normalization, and `function` never save, open a Repo, allocate an
identity, or capture payloads. They can read supplied Repo authority for reference
evidence. Mat realization can construct, restore, reserve, or claim through Repo.
Return normalization happens after the target body, so it sees a receipt saved by
the body; a return error does not roll back body effects or automatic saving by a
higher-level owner.

## Calling Boundaries

`@function` binds and normalizes arguments, invokes a synchronous function,
method, or callable instance, then normalizes its return. `normalize_args` and
`normalize_return` provide the same behavior for a uniquely discoverable immediate
caller. Discovery ambiguity, async/generator targets, unsupported annotations, and
unavailable authority raise `SignatureError`; helpers never guess a target.

Use `signature_context(repo=..., cache=..., reuse_live=..., selections=...)` to
borrow controls without consuming target keywords. Contexts are limited to the
originating thread and task, invalidate on exit, and do not open or close the
caller-owned Repo. Each boundary is one-shot and belongs to its creating
thread/task; concurrent callers must provide independent valid controls. Context
and boundary selection mappings are immutable snapshots. A prepared boundary
retains the borrowed lease and cannot realize after context exit or in another
task/thread.

Advanced `SignaturePlan.prepare_args`, `prepare_bound`, and `prepare_return` use
only controls passed explicitly to that call; they never fall back to ambient
authority. The official `function`, `normalize_args`, and `normalize_return`
helpers deliberately borrow an active `signature_context` for each boundary.
Consequently an integration can pass explicit controls in a child task even when
an inherited ambient context is invalid, provided the explicitly borrowed
Repo/Store lifetime remains valid.

Constructors normalize once after `__prepare_args__`; persisted and already-bound
records do not replay preparation. `Method` selects from raw tensor facts first,
then normalizes the selected implementation's arguments and return. Method adapters
deliberately borrow the active `signature_context` through the same core-owned path
as `function`, including its task/thread and lease checks. Managed
operations exclude their receiver and injected `managed` control, digest selected
annotated authority before realization, retain unannotated argument identity tags,
and normalize returns after the terminal-interruption guard and before final
publication. An unannotated managed structural return now materializes; annotate a
top-level structural result with `Ref[...]` when it must remain data.

Core Execute reconstructs a new worker-local plan rather than transporting a
`SignaturePlan` or `BoundaryPlan`. Its function, Method, and managed integrations
use their owners' narrow raw-result seam: the owner delivers arguments and invokes
once, Execute may publish a live result, and that same owner then performs its one
declared return normalization. These seams do not make contexts, prepared plans,
or caller-owned Repo/Store handles portable.

Core Execute's call description structurally lowers live Objects to their selected
`StateRef` or CDef authority and uses a single receiving Repo boundary for Mat
delivery. It recursively walks closure globals, defaults, annotations, instance
attributes, and `__slots__`; live Repo/Store resources are rejected rather than
being carried by a pickle fallback. Selected declaration Stores are represented by
the frozen execution Store table, never filesystem paths. This transport remains
internal and bounded; it is not a general pickle or RPC format.

The callable graph includes globals, defaults, annotations, closure cells, bound
receivers, instance attributes, and `__slots__` in one pass, preserving aliases
with the delivered arguments. Live Repo/Store captures fail rather than falling
back to pickling. A `ReferenceSelection` keeps its declaration Store pinned by
the frozen worker Store table, so recovery cannot choose a path-equivalent or
different replica. Execute does not auto-save inputs: only its core result owner
may publish a returned live Object or explicitly delivered update root after the
owner's raw-return seam.

## Migration

| Retired spelling | Replacement or rationale |
| --- | --- |
| `RefCDef` | Use `Ref[ConcreteDefinition]`; use `Ref[ConcreteDefinition | None]` for a nullable slot. |
| `RefCDefArg` | Use the same `Ref[...]` annotation. |
| `SelectorArg` | Use explicit `Ref[QuotedDef]`, `Ref[Selector]`, or `Ref[SelectorSpec]`. |
| `ArgRole`, `MaterializeArg`, `ValueArg` | Removed. The default is Mat and ordinary scalar values remain ordinary values. |
| `apply_arg_roles`, `apply_bound_arg_roles`, `apply_definition_arg_roles` | Removed. Constructors and bound records delegate to signature normalization. |
| `resolve_arg_roles`, `normalize_role`, `role_from_annotation` | Removed internal policy entry points; no replacement is supported. |
| `__dryml_arg_roles__` | Removed. Declare a flat `Ref[...]` or `Mat[...]` annotation instead. |

There are no aliases or forwarding modules for these names. In particular,
`Ref(value)` in an unannotated constructor or function slot now fails because that
slot defaults to Mat. New live constructor arguments can therefore form a selected
reference edge or materializing value according to their signature rather than the
retired role policy. Existing CDef, ObjectRef, and StateRef encodings and Store
contents are not rewritten or migrated.

## Errors

`SignatureError` has a bounded reason and optional slot path for invalid grammar,
binding, discovery, context ownership, wrapper conflicts, and unavailable or
ambiguous authority. Repo and lifecycle errors retain their own types and cleanup
semantics during Mat realization.
