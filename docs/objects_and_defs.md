# Objects and Definitions

`Definition` is a soft declaration for construction, selection, and search. A
finalized `ConcreteDefinition` (CDef) is a V2 fully bound semantic recipe. CDef
exposes import-free `.cls` and `.parameters`; `.args` and `.kwargs` project the
current class signature and may import the class. `Arg` and `Kwarg` represent soft
Definition spelling only. CDef paths use `Parameter` segments.

Two CDefs can compare structurally equal while describing different graph
topology. Equality, `hash()`, and `stable_hash()` compare class and parameters.
`graph_equal()` and `graph_hash()` additionally distinguish a shared child from
two independent equal children. Private CDef node tokens are neither persisted
nor public diagnostic identity.

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

## Restore And References

`Repo.load(cdef)` and `load_object(cdef)` are structural operations; they do not
infer a snapshot. `Repo.load_state_ref(state_ref)` restores the requested exact
authority through connected Stores. `Repo.restore_state_ref_into()` restores only
the supplied exact live graph after preflight and never searches for a substitute
Object. A restore hook failure invalidates that supplied graph for later framework
state IO. Load a fresh graph with `reuse_live="never"` from the preserved
StateRef before continuing.

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
