# Immutable Definition Graph

## Node Exact Preparation (Identity 3)

`Definition` authors an immutable occurrence/view handle; concretization returns
`ConcreteNode`. Handle equality does not merge independently authored equal nodes.
Use `Node.graph_equal` for exact semantic comparison, `Node.exact_key` for the
`dryml-node-exact/v1` certificate, and `Node.exact_digest` for its domain-separated
Node graph digest. Identity/state facets participate in these exact views.
Recorded class statefulness and assertion-finalization flags are authenticated
metadata, rather than semantic identity. Mat/Ref roles and quotation are semantic.

An operation can own and reuse preparation explicitly:

```python
from dryml.core.node_prepare import NodePreparation
from dryml.core.node_codec import encode_node_graph, decode_node_graph

prepared = NodePreparation((root,), scheduler="eager")
key = root.exact_key(prepared=prepared)
packet = encode_node_graph(prepared, roots=prepared.nodes)
restored = decode_node_graph(packet, view=0)
assert prepared.exact_equal(root, NodePreparation((restored,)), restored)
```

`ConcreteDefinitionGraph.prepare()` adapts existing traversal roots to this same
preparation. Exact preparation includes reference and quoted configuration, even
when the traversal graph's ownership closure excludes them. It never fills
defaults, resolves construction/factory targets, creates Objects, reads payloads,
allocates ObjectIds, or validates external availability. Explicit enum and
registered extension-dtype leaves use lazy value-type codecs on decoding.

### One Key Contract, Two Schedules

Preparation discovers shared occurrence/view rows and typed incidence once, folds
local value evidence bottom-up, and derives shape-only closure-overlap facts.
Disjoint distinct-child closures compose their certificates with the repeated
direct-target alias partition. Cross-branch joins use exact root-relative
adjacency. Eager topological folding and deferred dependency requests execute the
same decisions and produce identical key bytes, independent of root order,
enclosing roots, cache state and private tokens.

Unordered graph-bearing members are labeled in the complete selected rooted view.
Exact partition refinement uses outgoing typed incidence; remaining ties are
searched with streaming, alias-preserving whole-root labelings, including aliases
from other branches into unordered members.
Ordering members independently, using `repr`, or using process order as a tie
breaker is not part of the contract. Cached search evidence retains canonical
labelings/partitions, not a descendant graph blob at every node.

The certificate contains domain-separated SHA-256 evidence. Digest/key equality
alone is **not** collision-proof semantic authority. `NodePreparation.exact_equal`
compares full normalized rooted evidence, including under injected digest ties.
The independent flat test oracle checks equality partitions, rather than copying
the hybrid certificate algorithm.

### Ownership, Edits And Work Counts

`NodePreparation(..., previous=old)` copies reusable immutable row/incidence facts
and unaffected certificates into the new owner, remaps local coordinates, and
recomputes changed value facts and dependent ancestor keys. Rebinding and bitmap
compaction work are counted separately. It retains no previous
preparation packet. New physical exports compact the selected current table and
its retained construction handles. Resident key/value/search facts have both an
entry bound and a conservative byte budget (`max_cached_keys`,
`max_cached_bytes`); digest-domain facts are bounded per entry. Eviction and
`clear()` change work, never semantic results. Key/digest requests and clear are
serialized within an owner; two owners do not share mutable digest maps.

Warm resident key/digest requests do zero new discovery, folds or hashes. Counters
separate new discovery, normalized edges, reused-edge rebinding, local folds,
semantic hashes, fallback rows, unordered labelings, partition rounds/rows,
frontier unions/bit spans, ancestor work, block hashes, exported rows and bytes.
Local value-step hashes are counted separately as `value_hashes`. Shared symbolic
operands are folded and exported as value DAGs rather than expanded expression
trees; incidental sharing of pure values does not change exact semantic identity.
These are deterministic work units, not latency or constant-cost byte claims.
Closure bitmaps and exact join/refinement/search work can be superlinear;
unordered symmetry can require combinatorial search. There is no universal linear
hard-DAG claim or runtime refusal threshold derived from the small workload tests.

Structural lens edits keep Node carriers and exact identity/state constraints on
rebuilt ancestors, invalidating their retained construction evidence. They do not
assert compatibility of a retained state with the new configuration. Node
authoring snapshots arrays and freezes containers inside role assertions as well
as ordinary arguments, so caller mutation cannot invalidate pure preparation.

The production tests cover the admitted scalar/container, symbolic, factory,
immutable Core leaf, quotation, reference, facet and omission grammar, fresh-token
round trips, malformed records, cache behavior and both schedulers. The earlier
restricted prototype's 88 cases are not the qualification for this implementation.

### Integration Boundary

`PreparationView` names an explicit policy and optional pure root transformation.
The name participates in key identity and is checked by the reader; it does not
implement a policy by itself. Creation-CDef, compatibility and lifecycle/object
policies and their Store/managed integration belong to later units. Retained
construction handles and realization overlays are encoded now, without claiming
those later policy validations or availability guarantees.

## Existing CDef V2 Consumers

CDef V2 represents a rooted construction graph. It has two identity projections: structural equality/hash over class and fully bound parameters, and graph equality/hash over the same data plus sharing topology. A private node token distinguishes realizations internally but is never serialized, compared publicly, or shown in diagnostics.

The legacy `stable_hash`, `graph_hash`, V2 identity record and
`encode_cdef_graph`/`decode_cdef_graph` paths remain for live consumers awaiting
their owning cutover. Their bytes/domains are not relabeled as identity 3. The
legacy exporter and graph digest reject new Node facets or retained construction
evidence instead of silently discarding it. Unbound existing V2 consumers retain
their format; graph equality now compares exact evidence rather than structural
digest equality alone. Use the new Node APIs/codec for the full Node contract.

The legacy structural query catalog compares `stable_hash` plus `recipe_equal`,
including across authored ConcreteNode views and freshly decoded V2 records.
It does not use Node handle equality. These structural candidates retain the V2
index semantics; final selector matching still requires exact graph evidence for
fixed CDef anchors, including their sharing topology. The adapter introduces no
new persisted identity or fallback scan.

Current `ObjectRef`/`StateRef` values and Store `DefinitionRecord` matching use
V2 reference/quotation value scopes, including when the supplied unbound recipe
is a Node. V2 serializes those values independently: sharing within each scope,
roles, configuration, ObjectIds and state selections remain exact, but aliases
crossing independently transported values are not V2 authority. Publication,
read-back and restore preflight compare that exact scoped evidence rather than
handle equality or digest equality alone. DefinitionRecord matching rejects
Node facets and retained construction, including inside embedded references;
the captured metadata format remains V2. Full `Node.graph_equal` and identity-3
keys still compare the complete closure and its cross-value alias partition.

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
