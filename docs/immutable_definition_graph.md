# Immutable Definition Graph

CDef V2 represents a rooted construction graph. It has two identity projections: structural equality/hash over class and fully bound parameters, and graph equality/hash over the same data plus sharing topology. A private node token distinguishes realizations internally but is never serialized, compared publicly, or shown in diagnostics.

Every CDef constructor field is addressed by `Parameter(name)`. Typed container segments preserve deterministic traversal and primary paths. `Arg` and `Kwarg` are available only when working with soft `Definition` syntax; they are invalid CDef edges.

Raw nested CDefs and `Mat(...)` are materializing edges. `Ref(...)` is non-materializing and remains an unchanged lightweight target at runtime. CDef graph encoding uses deterministic graph-local labels, rejects duplicate or dangling declarations, and recreates private tokens on decode. Inspection, graph hashing, and reference projection do not resolve classes.

An `ObjectRef` expands owned materializing topology and records ObjectIds at its canonical primary paths. A `StateRef` adds local-state hashes at exactly those paths. Thus graph topology, durable lineage, and a checkpoint remain distinct values.

An `ObjectSelector` is the query-only object projection of either reference. Its
projected CDef replaces encountered ObjectRef and StateRef leaves with nested
ObjectSelectors, preserving graph topology and ObjectIds without state hashes.
Raw references explicitly placed inside a selector remain exact. Selector-bearing
CDefs can be hashed and matched but are rejected by the authoritative CDef codec;
generated selectors retain their separate non-state ObjectRef association through
`ObjectSelector.reference` when persistence requires authority.

An Artifact Fold's source/model-facing `Ref[AutoRef]` edge is terminal data in this
graph. Saving or restoring a completed Fold state preserves that selected reference
but does not make its input an owned snapshot dependency. The Fold's result lives
in local Value state and can be restored independently; a future compute resolves
the retained reference under its managed Repo.

## Repo-Owned Token-Neutral Preparation

Public graph, reference and framed-record decoders always fully validate input.
An internal, explicitly passed Repo read-cut interface can reuse complete validated
record/graph preparation after fresh frame/dill decoding, exact source-byte matching
and lossless eligible-content matching. The existing codec separates canonical
validation from fresh node construction; there is no alternate graph validator.
Facts contain primitive labels, topology, hashes and role paths, never CDefs,
resolved classes, current annotations, Objects or payload-validity evidence.

Only closed exact-ImportRef graphs with primitive/tag/container/link values and
standard non-object numeric/bool ndarrays (including exact FrozenNDArray) qualify.
SourceSpec, quoted definitions, embedded references, arbitrary atoms/subclasses,
object/extension dtypes and unsupported contexts do not qualify. Retained arrays
use immutable owned bytes plus dtype/shape information. Realization rebuilds fresh
symbols, containers, arrays and node tokens, retaining graph-local sharing; making
a returned array/base writable or mutating nested returned role mappings cannot
alter later decodes. Class resolution, abstractness and role admission still use
the current class at the materialization boundary.

See [Repos and Stores](repos.md#bounded-snapshot-and-graph-content-reuse) for the
shared bounded resident/pending budget, cut success staging and lifecycle rules.

The signature boundary determines newly authored Ref/Mat edges; see
[Signatures](signatures.md) for supported annotations and delivery behavior.

Symbolic Definition repetition distinguishes construction topology from
configuration. Repeated groups receive fresh CDef nodes by default while
same-named active parameters retain one value; `Shared(...)` preserves node
identity at matching positions. This does not make FactorySpec consumers share
TensorFlow/Torch backend instances or weights. A Definition admitted through a
Template role is terminal `QuotedDef` recipe data in this graph and can remain
symbolic without becoming a materializing edge.
