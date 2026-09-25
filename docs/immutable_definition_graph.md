# Immutable Definition Graph

CDef V2 represents a rooted construction graph. It has two identity projections: structural equality/hash over class and fully bound parameters, and graph equality/hash over the same data plus sharing topology. A private node token distinguishes realizations internally but is never serialized, compared publicly, or shown in diagnostics.

Every CDef constructor field is addressed by `Parameter(name)`. Typed container segments preserve deterministic traversal and primary paths. `Arg` and `Kwarg` are available only when working with soft `Definition` syntax; they are invalid CDef edges.

Raw nested CDefs and `Mat(...)` are materializing edges. `Ref(...)` is non-materializing and remains an unchanged lightweight target at runtime. CDef graph encoding uses deterministic graph-local labels, rejects duplicate or dangling declarations, and recreates private tokens on decode. Inspection, graph hashing, and reference projection do not resolve classes.

An `ObjectRef` expands owned materializing topology and records ObjectIds at its canonical primary paths. A `StateRef` adds local-state hashes at exactly those paths. Thus graph topology, durable lineage, and a checkpoint remain distinct values.

An Artifact Fold's source/model-facing `Ref[AutoRef]` edge is terminal data in this
graph. Saving or restoring a completed Fold state preserves that selected reference
but does not make its input an owned snapshot dependency. The Fold's result lives
in local Value state and can be restored independently; a future compute resolves
the retained reference under its managed Repo.

The signature boundary determines newly authored Ref/Mat edges; see
[Signatures](signatures.md) for supported annotations and delivery behavior.
