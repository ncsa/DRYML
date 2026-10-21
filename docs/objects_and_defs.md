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
This authoring API does not change Store formats, Generator policies, signatures,
or Repo realization of identity/state constraints.
`Node.build()` retains ordinary unbound-recipe construction and rejects new
identity/state-bearing Node graphs before entering the existing Repo boundary;
it does not silently discard those facets.

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
Repo save contract and returns one immutable root `StateRef`, or `(StateRef,
StoreReport)` when `report_stores=True`. Its `store=` override selects one
whole-graph closure; routing otherwise determines per-object or closure placement.
See [Repos and Stores](repos.md) for mode precedence, replication, and partial
publication evidence.

`Object.last_state_ref` is a read-only runtime receipt for the last complete
top-level StateRef publication, or `None`. It advances only after immutable root
authority is complete and before a derived index, main reference, alias, or
archive commit can fail. Thus a later failure can preserve a valid inspectable
receipt without reporting overall save success. A successful exact top-level load
also installs its requested receipt. Embedded descendants never receive projected
or synthetic receipts, and later ordinary mutation does not change the receipt.
For a managed declaration using `return_state_ref=True`, the call result is the
exact final StateRef captured for that invocation after association; callers must
not substitute a later mutable `last_state_ref` receipt. Interruption, warning
failure, or late publication/association failure returns no successful reference.

## Restore And References

`Repo.load(cdef)` and `load_object(cdef)` are structural operations; they do not
infer a snapshot. `Repo.load_state_ref(state_ref)` restores the requested exact
authority through connected Stores. `Repo.restore_state_ref_into()` restores only
the supplied exact live graph after preflight and never searches for a substitute
Object. A restore hook failure invalidates that supplied graph for later framework
state IO. Load a fresh graph with `reuse_live="never"` from the preserved
StateRef before continuing.

Query V3 returns immutable identity values rather than live Objects. Select a CDef
with `repo.query().sel(...).cdefs()` and pass the selected value to an explicit Repo
load method when construction is intended.

Default `Pickleable` restoration replaces ordinary payload fields rather than
merging them. It removes fields absent from the checkpoint while preserving graph
bindings and framework runtime metadata. A stateless Object root can own stateful
descendants; its checkpoint state comes from those descendants, not unsaved
root-only attributes.

ObjectRef identifies the exact Object graph and ObjectIds but not a snapshot.
StateRef identifies the corresponding immutable local-state hashes. Alias lookup
therefore resolves an Object alias to ObjectRef authority and a state selector to
StateRef authority; there is no generic object-returning alias load. Exact state
identity, graph sharing, and reference-as-value semantics remain independent of
Store placement or replication.

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
