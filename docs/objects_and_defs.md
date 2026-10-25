# Objects and Definitions

`Definition(...)` authors an immutable `Node` for construction, selection, and
search. `concretize()` returns a `ConcreteNode`, using the existing fully bound
`ConcreteDefinition` (CDef) record representation. Concreteness does not imply
identity or saved-state completeness. CDef
exposes import-free `.cls` and `.parameters`; `.args` and `.kwargs` project the
current class signature and may import the class. `Arg` and `Kwarg` represent soft
Definition spelling only. CDef paths use `Parameter` segments.

Node equality and `hash()` identify an immutable occurrence/view handle. Two
independently authored equal recipes remain two set members; repeating one handle
remains one. Use `recipe_equal(other)` for recipe comparison and
`graph_equal(other)` for exact roles, facets and alias topology. The latter also
matches unordered incidence without using `repr` as identity. Python pickle
round trips preserve the alias partition and facets, with fresh process-local
handles rather than handle equality with the source.

Constructor values retain distinct list, tuple, set, frozenset and mapping tags.
`Repo.materialize_boundary` also preserves runtime container families, including
built-in `frozenset` versus tagged `FrozenSet`, while retaining reference aliases
across containers and roots.
Mappings admit exact string/integer keys (not bool-as-int keys). Mutable arrays
are snapshotted with dtype, shape, byte order and logical contents retained,
including across Python pickle; object-dtype arrays are rejected.

The existing plain `ConcreteDefinition` construction-view API retains structural
equality for existing consumers. Legacy `stable_hash()` and `graph_hash()` remain
construction-view keys, not new facet-aware Node keys or saved-graph encodings.
The versioned CDef construction codec likewise remains a construction-view codec.
`Node.build(repo=..., identities=..., states=...)` composes optional constraints,
concretizes once, and enters `Repo.build(ConcreteNode)`. Build allocates only missing
ObjectIds, initializes unspecified stateful nodes, and restores selected compatible
LocalStates once. Ref and quotation metadata are not materialized implicitly.

## Node Constraints And Navigation

Identity and state controls are Node methods, never special constructor keywords.
For example, `Definition(Target, state=value)` still supplies an ordinary
constructor parameter named `state`; `node.state` inspects the independent facet.

```python
from dryml.core import Definition, ObjectId

child = Definition(Target, width=32)
recipe = Definition(Parent, left=child, right=child)
identified = recipe.bind(identities={"$.left": ObjectId()})
assert identified.at("$.left") is identified.at("$.right")
assert child.object_id is None
```

`bind(identities=..., states=...)` accepts rooted maps or extracted `IdentitySpec`
and `StateSpec` mappings. State assignments must be exact `LocalState` values,
not alternatives, alias selectors, bare hashes, or bare StateKeys. Repeated equal
assignments are idempotent; complementary identity/state assignments compose.
Different exact assignments to a shared occurrence raise `ValueError`, even when
supplied through different paths. Independently authored equal nodes remain
independent. Embedding complementary immutable views of one occurrence composes
them at authoring; contradictory embedded views fail instead of taking precedence.

`with_identity(id)` and `with_state(local_state)` constrain the root. Deliberate
`replace_bindings(identities=..., states=...)` replaces existing facets and accepts
`None` to remove them. `without_identity()` and `without_state()` remove the root
facet. Bulk replacement rejects contradictions within the edit itself.
`replace(path, replacement_node)` replaces every alias of the selected occurrence
in the returned graph, leaving independent equal nodes and the original unchanged.
A soft replacement weakens affected concrete ancestors to general Node views.

The retained recipe editors `with_cls`, `with_args`, `with_arg`, `with_kwargs`,
`with_kwarg`, and `without_kwarg` preserve the root occurrence and its exact
ObjectId/LocalState facets. Nested facets, aliases, and quoted values survive;
newly embedded complementary aliases compose and contradictory views fail.
An actual recipe change invalidates the edited root's retained construction
evidence; an unchanged recipe preserves that evidence. These metadata edits do
not check state availability or compatibility. For example, changing width on a
state-bound model retains that selection, and an incompatible build rejects it
before constructors or restore hooks rather than initializing fresh state.
Remove selections deliberately with the binding-removal APIs above.

`node.at(path, traverse_refs=False, traverse_quoted=False)` returns **Node endpoints
only**, for soft and concrete metadata. Ordinary materializing structure is
traversable by default. A path can terminate at a Ref-held or quoted Node without
either opt-in; further descent needs the corresponding independent flag. Neither
flag changes the link's role or ownership. Scalar/container endpoints and blocked
descent raise `GraphPathError`. Typed `Arg`/`Kwarg` address soft call spelling;
`Parameter` addresses a concrete named record (or safely named soft parameters).
Use `node.lens(path).set(value)` for the former structural-slot editing API,
including scalar endpoints. A lens edits one slot; `replace` edits an occurrence.

`identity_spec()` and `state_spec()` are read-only rooted map adapters over facets,
not independently stored authority. They choose one primary path per owned
occurrence, rebase at the selected root, and preserve sharing. References and
quotations do not enlarge materializing ownership. `object_view()` retains
identities and alias topology while removing graph-reference state facets;
quoted configuration stays exact.

`with_realization(identities=..., states=...)` records supplied realization facets
as an immutable overlay. Its `construction_view` retains the prior immutable
construction selections: recording a child capture S1 does not overwrite an
explicit construction pin S0. This records metadata only; it does not capture,
restore, allocate, read payloads, or publish. Navigation, binding, replacement and
extraction likewise perform none of those effects or target resolution.
Realization overlays do not enter quotations: when an owned Node is also held as
quoted future-recipe data, only its unquoted aliases receive allocated identities,
captured states, partial state application, or removed state pins. The quotation
retains its exact original view and internal sharing, including when it holds a
different facet view of that same occurrence. Explicit authoring edits through
`bind` and `replace_bindings` retain their occurrence-wide alias semantics.
Graph identity/declaration collection likewise stops at quotations. When a
metadata transformation exactly removes an overlay, it reuses the retained
original view to preserve any original alias to quoted data; distinct facet views
are not merged merely because they share an occurrence.

`require_saved(concrete_node)` validates that every owned stateful occurrence has
an exact ObjectId and LocalState, with no independent identity collisions. It
returns the **same handle**, typed as the named `SavedNode` completeness
refinement. It adds no graph wrapper and asserts neither payload availability nor
current restore compatibility. Ephemeral nodes need no own private-state facet.
Mixed owned legacy ObjectRef/StateRef leaves are rejected by this new completeness
validator, rather than treated as an empty owned closure. Existing reference
consumers retain their separate APIs. `copy_graph()` preserves facets and the
alias partition while creating fresh private Node handles, without ObjectId
allocation or encoding.

### Allocating Identities Before Dispatch

`concrete_node.allocate_identities()` fills in every missing ObjectId in the
owned stateful graph without constructing Objects or registering anything in a
Store. Concretize first so ownership and stateful roles are already known:

```python
from dryml.core import object_namespace

cdef = recipe.concretize()
with object_namespace("jobs", "data"):
    identified = cdef.allocate_identities()

assert identified.allocate_identities() is identified
```

The returned ConcreteNode has one new UUID-based ObjectId per distinct owned
stateful occurrence. Shared children keep one identity; independent equal recipes
receive different identities. Stateless nodes receive none. Reference-only targets
are not allocated, while non-quoted references to occurrences also owned by this
graph remain aliases of those identified occurrences. Quoted future recipes stay
exact, including when they quote an owned occurrence.

Existing IDs and LocalState selections remain unchanged. A fresh state-free recipe
therefore becomes identity-complete without selecting or capturing any state; a
recipe containing a saved dataset/model retains that selection. New IDs are
realization metadata, not retroactive construction pins: retained construction
views and creation digests stay unchanged. The source graph is immutable. A fully
identified graph returns the same handle without allocating UUIDs. Soft inputs or
legacy owned reference leaves raise `TypeError`; independent owned nodes sharing
an ObjectId or stateless nodes carrying identity/state facets raise `ValueError`,
before any new IDs are allocated.

By default, `check_unique=False`: UUIDs are allocated without consulting a Repo.
To check new IDs against known Repo identities, use:

```python
identified = cdef.allocate_identities(check_unique=True, repo=repo)
```

Checked allocation uses a supplied Repo or an explicitly active default Repo;
without either it raises `ValueError`, rather than creating an empty Repo. It
checks full ObjectIds (namespace and nonce) against connected Store declaration
anchors, live caches and cached graph metadata, IDs already present in the supplied
graph, and IDs assigned earlier in this call. It preserves existing IDs, retrying
only newly generated collisions, with at most 32 candidates per missing ID before
`RuntimeError`. Store authority, capability and IO errors propagate. A fully
identified graph still returns itself without UUIDs or Store reads. Checked mode
does not publish or retain candidates. Its observations do not prevent another
worker from assigning/publishing the same ID after the check; this is not an atomic
reservation or a global uniqueness guarantee.

Allocate once per logical job, before sending it to workers, and reuse the returned
graph for retries. Separate calls on the original unbound recipe create independent
job identities. The helper performs no class resolution, payload IO, Store
publication, or persistent reservation. It replaces manual graph-wide identity
binding, not execution coordination: duplicate managed calls still require the
same control authority and follow the [managed operation ownership
contract](managed_operations.md#operation-ownership-and-shared-mutation).

## Creation, Compatibility And Operation Views

The metadata APIs distinguish three named transformations over the same immutable
Node graph. They use `NodePreparation`'s topology-aware exact verifier; a digest
or compatibility fingerprint is never sufficient equality evidence.

| API | Identity/state handling | Domain |
| --- | --- | --- |
| `node.creation_view()` / `creation_digest()` | Use retained normalized construction authority; exclude only the captured root's own identity/state. Preserve explicit dependency selections. | `dryml-creation-cdef/v1` |
| `node.compatibility_view()` / `compatibility_fingerprint()` | Remove realization identity/state recursively through graph links. Compare class, normalized configuration, sharing/topology and link roles. | `dryml-state-compatibility/v1` |
| `node.object_view()` / `object_projection_digest()` | Keep identities; recursively remove graph-reference state selections. | `dryml-object-projection/v2` |

All three preserve quoted constructor data exactly, including ordinary ObjectIds
and checkpoint pins in future-model recipes. Query traversal options do not alter
this rule. `object_projection_equal(original_candidate)` performs the necessary
key check **and** exact normalized graph comparison, including unordered aliases.
Creation and compatibility require concrete normalized roots. Object projection
also accepts soft Nodes without resolving their targets.

`dryml.core.node_views` exports `CREATION_VIEW`, `COMPATIBILITY_VIEW`, and
`OBJECT_PROJECTION_VIEW` for explicit operation-owned preparation, for example
`NodePreparation((node,), view=CREATION_VIEW)`. Requests use its transformed
`prepared.roots`; eager and deferred preparation implement the same key contract.
Ordinary named digest/equality calls also reuse bounded pure preparation owned by
the immutable source handle, so resident repeated calls add no graph discovery,
folds, or semantic hashes. Derived view tables and their reserved fact budgets
share the owner's entry/byte limits. Eviction or clearing preparation changes
work only; it never changes keys, quotation, roles, sharing, or facet semantics.
These owners are transient: fresh graph copies and portable transport do not
carry cache locks or external evidence, and unreachable source graphs can be
collected.

`local_state.validate_for(destination, codec="pkl")` validates the retained
normalized origin evidence against its full StateKey, then verifies exact
destination compatibility and the explicitly admitted codec. It returns the same
LocalState or raises `TypeError`/`ValueError` before any payload hook. It does not
resolve a class to guess codec support, invoke a custom compatibility callback,
open payloads, or prove their availability. Applying compatible foreign-origin
metadata preserves the donor's StateKey and creation evidence. A subsequent new
capture's address must derive from the destination's retained construction view.
Multiple LocalStates sharing immutable creation evidence reuse its preparation;
each validation still checks the selected StateKey and admitted codec and compares
exact destination evidence. Cached graph facts certify no alias, live Object,
Store contents, or payload availability. Build/load/restore reacquire and validate
their external evidence even after successful warm metadata validation.

A parent constructed with child S0 retains S0 in its creation view after a
realization overlay selects child S1. Automatically allocated identities are not
retroactive construction pins, and the root's own captured StateKey does not
recur in its creation digest. `receiver.adopt_saved_view(original_saved_node)` is
the metadata adoption step: it first matches the **original supplied graph's**
object projection, then returns supplied saved facets and construction pins over
the receiver's occurrence identities and sharing. Later creation inspection uses
that adopted construction authority; authenticated Node transport retains it.
The receiver and supplied graph are immutable and remain unchanged on failure.

The Node view APIs above are metadata-only. Repo build/load/apply/restore consume
these views with fresh state validation as described below. Managed-operation
integration remains a separate consumer boundary. The V2 `object_projection()`
query adapter and legacy call-surface
thaw/export refuse identity-3 facets or retained construction evidence rather
than silently dropping them. Use the Node APIs above for the new views.

## Live Objects And Exact State

An `Object` is one live realization. `Serializable` marks nodes that receive an
`ObjectId`; a plain Object is ephemeral even when it owns stateful descendants.
`Object.graph_at()` reads retained realization bindings without construction,
restoration, cache lookup, or imports. It returns a live Object for a materializing
path and the corresponding CDef, ObjectRef, or StateRef value unchanged for a
reference path.

```python
class Counter(Serializable):
    state_codec = "pkl"

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        ...  # Write one complete semantic checkpoint under dest_dir.

    def restore_state_from_dir_imp(self, src_dir, *, codec):
        ...  # Restore that checkpoint from the framework-provided directory.
```

Hooks receive only a framework-provided payload directory and validated opaque
codec. They must write and restore a complete semantic checkpoint; DRYML cannot
detect arbitrary mutation outside that contract. `Object.save()` delegates to the
Repo save contract and returns one immutable root `SavedNode`, or `(SavedNode,
StoreReport)` when `report_stores=True`. Its `store=` override selects one
whole-graph closure; routing otherwise determines per-object or closure placement.
See [Repos and Stores](repos.md) for mode precedence, replication, and partial
publication evidence.

`Object.last_state_ref` is a read-only historical SavedNode receipt, or `None`.
Complete protected save publication and successful top-level exact loading/full
restoration install receipts. An incomplete save preserves the previous receipt.
Ordinary mutation does not update it. It proves neither current live state nor
current payload availability and never authorizes skipping a required restore.

## Restore And References

`Repo.load(saved_node)` and `Repo.load_state_ref(saved_node)` require complete
SavedNode metadata. Every required stateful materializing occurrence needs its
identity and selected state. Missing, reclaimed, corrupt or incompatible selected
state fails explicitly before constructors/hooks where possible; no sibling state,
historical receipt or default initialization substitutes for it. Entirely stateless
concrete graphs are saved-complete without fabricated captures or identities.

`Repo.apply_state(obj, states)` accepts a rooted StateSpec or typed path/LocalState
map. It restores only selected destinations and leaves unspecified state untouched.
It neither assigns donor identities nor captures or publishes a complete checkpoint.
Compatible foreign-origin state retains its original full StateKey. A later fresh
capture uses the destination's retained construction view.

`Repo.restore(obj, saved_node)` (also `restore_state_ref_into`) checks the live
receiver's canonical object projection against the **original supplied graph**,
before adopting its construction pins. Successful restoration preserves Python
identities, ObjectIds and sharing. `Repo.describe(obj)` exposes adopted construction
and realization metadata. A parent constructed with child S0 and saved with child
S1 therefore retains S0 in its creation authority after load or full restore.

Selected sources are opened by full StateKey, grouped once per distinct key/Store
and once per creation bucket. Each destination independently validates creation
compatibility and codec admission. Native shared key leases span validation and
dependency-first constructors/hooks; hooks execute outside Store writer fences.
Every independent use revalidates availability, including live-cache reuse.
`reuse_live="matching"` and `"greedy"` retain unique-candidate identity rules;
eligible reused nodes still run their selected restore hook. `"never"` builds fresh.

For a handoff to another Core caller, `repo.prepare_restore(obj, saved_node)` returns
an owner-thread, one-shot `PreparedRestoration`. Use it as a context manager and call
`prepared.apply()` once, or close it without hooks. It retains the live reservation
and current source leases until consumption/close; neither a detached plan nor a
last-save receipt is a reusable authorization to skip restoration.

Validation failure before any hook leaves a previously usable live graph usable.
Failure after hook entry invalidates the affected retained live graph, evicts it
from framework caches, and rejects later state IO. There is no automatic rollback.
Recover with `repo.load_state_ref(saved_node, reuse_live="never")` from a valid
available saved graph; its ObjectIds are preserved.

Queries return immutable Nodes rather than live Objects. Select a concrete recipe
with `repo.query().sel(...).cdefs()` and pass the selected value to
`repo.build(concrete_recipe_node)` when construction is intended. Build allocates
missing identities and initializes unspecified state. For exact restoration,
select a complete SavedNode with `saved()` / `state_refs()` and use
`repo.load(saved_node)` or `repo.load_state_ref(saved_node)`; a stateful recipe
without the required identity/state facets is not loadable as a saved graph.

Default `Pickleable` restoration replaces ordinary payload fields rather than
merging them. It removes fields absent from the checkpoint while preserving graph
bindings and framework runtime metadata. A stateless Object root can own stateful
descendants; its checkpoint state comes from those descendants, not unsaved
root-only attributes.

ObjectRef/StateRef are legacy in-memory annotation/reference adapters. New public
materialization and capture use Node-local facets; persisted V2 snapshots are not
revived by format-4 loading. Exact state identity, sharing and reference-as-value
semantics remain independent of Store placement or replication.

Pre-V2 CDef records, raw tuple/dict records, missing identity versions, and mixed
graphs are rejected before construction. There is no migration, converter, or dual
reader. Durable layouts and validation limits are described in [Formats](formats.md).

Constructor and call-boundary roles are documented in [Signatures](signatures.md).
Newly formed live-argument definitions follow their declared Ref/Mat role; this
does not rewrite existing Store definitions or their identity algorithms.

## Construction And Factories

Ordinary Object construction binds a supplied call to the constructor signature:
the resulting CDef has named parameters and declared defaults. An authored
`Definition`, by contrast, preserves supplied call spelling and omission until it
is concretized. This difference is intentional: a partial Definition without
`seed` does not constrain `seed`, while a CDef created from that call can contain
the constructor's bound default.

`F(...)` and `FactorySpec(...)`, exported from both `dryml` and `dryml.core`, are
the same inert factory class for non-Object runtime values. They preserve only
the supplied target, positional arguments, and keywords. They neither inspect a
target signature nor insert omitted defaults, resolve a target, or construct an
object. Consequently, positional and keyword factory spellings can have distinct
identities, and a later target-default change can affect a build without changing
an unchanged factory declaration. `FactorySpec.build(namespace=...)` resolves and
invokes the target at that explicit consumer boundary; resolution can import
trusted target code.

## Symbolic Definitions

`Definition` is also the sole runtime value for symbolic construction. It may
contain `Par` roots and supported arithmetic or repetition expressions while
remaining immutable. `names` reports active roots in authored first-occurrence
order and `is_resolved` means only that no active expression remains. Neither
operation resolves a target, interprets a receiving constructor role, applies
defaults, or proves a constructor call is complete or valid.

Use `Definition.sub(...)` for static binding and `Definition.remap(...)` for
simultaneous root renaming; both return new Definitions and do not construct,
save, import, or sample. `Distribution` is not construction data and is rejected
anywhere in Definition-owned structure. To vary every remaining active root, pass
the partially bound Definition and one exact root-to-Distribution mapping to a
`Generator`. Its samples and grid members are resolved Definitions. See
[Symbolic Definitions And Generation](templates.md) for expression traversal,
role admission, and exact support selection.

## Categorical Selectors

`Definition.categorical(...)`, `ConcreteDefinition.categorical(...)`, and
`categorical_definition(...)` return a new `Definition` selector expression;
they never modify the source definition, its references, or saved state. Use
`drop=("seed",)` to remove semantic constructor constraints by name,
`drop_args=True` to remove every argument constraint, and `drop_class=True` to
remove the class constraint. Omission means unconstrained, not `None` or a
default. Both flags yield a classless Definition wildcard for definitions, not a
wildcard for arbitrary scalar values.

The definition and query methods default to projecting only the selected root;
the standalone `categorical_definition(...)` helper retains its
`recursive=True` default. Explicit `recursive=True` applies the
controls to nested definitions in that subtree, without inspecting FactorySpec
arguments or ordinary mapping keys. Each named drop must occur somewhere in the
original selected traversal; malformed controls, a wholly unmatched name, a
cycle, or a set transformation that would collapse members fails without a
partial result. Repeating an already removed named drop can therefore fail.
Parameters named `uid` and `metadata` are ordinary parameters and remain
constrained unless explicitly dropped.

Projection reads a CDef's stored named parameter record without resolving its
class. Naming positional or variadic constraints in an authored Definition can
resolve an ImportRef or execute trusted SourceSpec source to inspect a signature,
but never constructs the target or fills omitted defaults. The result is a
selection expression, not a promise that its retained fields are a valid
construction call. SourceSpec remains the recorded authority; signature
inspection does not replace it with a resolved class. DRYML provides no hot
reload, historical ImportRef, or live-Object migration guarantee.
