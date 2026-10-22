# Reference Values

## Node-local constraints

`Node.object_id` and `Node.state` independently describe exact optional facets on
one immutable occurrence/view. `Definition(...)` authors Node graphs;
`concretize()` produces a ConcreteNode over the existing normalized construction
record. Concreteness, identity completeness and saved completeness are separate.
Node handle equality/hash do not compare recipes; use `recipe_equal` or
`graph_equal` for those questions. See [Objects and Definitions](objects_and_defs.md)
for binding, navigation, edits and the retained construction-view distinction.

`StateKey(cdef_digest, codec_hash)` is a validated compound address: a lowercase
64-hex creation-CDef digest and a codec-qualified content hash. Equal codec hashes
under different creation CDefs are different selections. The key by itself makes
no availability or retention claim.

`LocalState(key, creation_cdef)` is detached immutable metadata for one exact
selection, without a donor ObjectId or an open Store handle. `creation_cdef` must
be normalized construction evidence; `.codec` reads the qualified codec name.
Metadata authoring checks types and address spelling. It does not compute or
authenticate creation digests, validate compatibility, or open payloads; those
are responsibilities of the consuming operation. Equal selections compare the
compound key and semantic construction evidence, not evidence-handle identity.

The explicit metadata boundary is `local_state.validate_for(destination,
codec="pkl")`. It checks that origin evidence is normalized without root
identity/state, matches its creation digest, is graph-compatible with the
normalized destination, and uses the consumer-admitted codec. The compatibility
fingerprint only narrows candidates; exact verification retains class,
configuration, topology/sharing, roles and quoted data. No arbitrary user
compatibility callback is accepted. Failure raises `TypeError` or `ValueError`
without payload reads or hooks. Success returns the **same** LocalState: its full
donor StateKey is not replaced by a destination-derived address, even when equal
codec bytes came from different creation definitions. Availability and runtime
application require their owning consumer's additional checks.

Use `node.creation_view()` for origin evidence, `node.compatibility_view()` for
destination comparison, and `node.object_view()` for identity-preserving,
state-weakened operation metadata. Their independent hash domains and adoption
semantics are documented in [Objects and Definitions](objects_and_defs.md#creation-compatibility-and-operation-views).
Binding or generating metadata may retain an incompatible selected state;
validation must reject it before application rather than silently removing it.

`IdentitySpec(node)` and `StateSpec(node)` are rooted Mapping adapters over the
same Node facets. They expose only present owned assignments, one primary path
per occurrence, and may be passed to `Node.bind`. StateSpec never assigns donor
identities. Extract a subtree with `.at(...)` first to obtain rebased views.
Ordinary extraction excludes Ref and quoted ownership even when metadata
navigation explicitly descends through those boundaries. No payload reads,
construction, ObjectId allocation, or publication occur during these operations.

`SavedNode` is a named typing refinement; `require_saved` validates completeness
and returns the supplied ConcreteNode itself. It is not an availability proof.

## Existing exact-reference consumers

`ObjectId` is a namespace plus framework-generated UUID nonce for one `Serializable` lineage. `ObjectRef(definition, objects)` combines V2 graph topology with all owned ObjectIds. `StateRef(object, states)` adds immutable local-state hashes. These values are import-free canonical leaves and their equality includes complete topology and IDs.

`Ref(ObjectRef | StateRef)` is non-materializing. Bare or `Mat(ObjectRef | StateRef)` expands owned materializing topology. Materializing exact state references load through the exact StateRef path; a raw ObjectRef may be built only through its registered declaration and claim.

`StateSelectorRef(object, alias)` is soft only. Canonicalization resolves it once through a Repo to an exact StateRef before a CDef or finalized query exists. Object aliases name ObjectRefs; state aliases are scoped by complete ObjectRef. Alias movement never mutates a finalized CDef.

`Definition.concretize(repo=resolver)` also accepts a metadata resolver exposing
`resolve_state_selector(selector)`, as does `to_canonical`. Repeated occurrences of
one selector resolve once per normalization. Missing-alias errors propagate, and a
resolved StateRef outside the selector's complete ObjectRef scope is rejected.

Use [Signatures](signatures.md) for `Ref`/`Mat` conversion, automatic selection,
quotation data, and the distinction between metadata selection and Repo realization.
Query V3 treats complete ObjectRef and StateRef values as identity members without
dereferencing them; see [Graph Querying](graph_querying.md).

The existing reference `object_projection()` and ObjectSelector APIs retain their
V2 query semantics, including explicit inspection options. They do not define the
new operation-projection hash domain. Legacy call-surface/query exports reject
identity-3 Node facets and retained construction evidence when they cannot
represent them; the authenticated Node codec carries those constraints losslessly.
