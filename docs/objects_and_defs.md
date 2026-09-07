# Objects and Definitions

`Definition` is a soft declaration for construction, selection, and search. A finalized `ConcreteDefinition` (CDef) is a V2 fully bound semantic recipe. It exposes import-free `.cls` and `.parameters`; `.args` and `.kwargs` project the current class signature and may import that class. `Arg` and `Kwarg` describe soft Definition spelling only. CDef paths use `Parameter` segments.

Two CDefs can be structurally equal while representing different graph topology. Ordinary equality, `hash()`, and `stable_hash()` compare only class and parameters. `graph_equal()` and `graph_hash()` additionally distinguish one shared child from two independent equal children. Private CDef node tokens are never public, persisted, or diagnostic identity.

An `Object` is one live realization. `Serializable` marks nodes that receive an `ObjectId`; a plain `Object` is ephemeral even when it owns stateful descendants. `Object.graph_at()` reads retained runtime bindings without construction, restoration, cache lookup, or imports. It returns live Objects for materializing paths and unchanged CDef/ObjectRef/StateRef values for `Ref` paths.

```python
class Counter(Serializable):
    state_codec = "pkl"

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        ...

    def restore_state_from_dir_imp(self, src_dir, *, codec):
        ...
```

Hooks receive only a framework-provided payload directory and opaque validated codec. They must write a complete semantic checkpoint; DRYML does not detect arbitrary Python mutation. `Object.save()` publishes a complete immutable `StateRef`; use `Repo.load_state_ref()` to restore a snapshot into a new realization.

`Object.last_state_ref` is a read-only runtime receipt for the last fully published top-level `StateRef`, or `None` before one completes. It is installed after immutable state authority is complete and before derived index, main-reference, or alias updates, so a later failure can propagate while the valid receipt remains available. A successful top-level exact load also installs its requested receipt. Embedded descendants never receive projected/synthetic receipts, and ordinary unsaved mutation does not change the receipt. The receipt is excluded from serialized payload state and does not promise that the live object remains unchanged since publication.

`Repo.restore_state_ref_into()` restores only the supplied exact live graph after
preflight; it never searches for or substitutes another object. If a restore hook
fails after hooks begin, that graph is invalid for further state IO. Load a fresh
exact graph from the preserved StateRef instead. Default `Pickleable` restoration
replaces ordinary payload fields rather than merging them, removing attributes
added after a checkpoint while preserving graph bindings and framework runtime
metadata. A stateless `Object` root may own stateful descendants; checkpoint state
comes from those descendants rather than unsaved root-only attributes.

Pre-V2 CDef records, raw tuples, missing identity versions, and mixed graphs are rejected before construction. There is no migration, converter, or dual reader.
