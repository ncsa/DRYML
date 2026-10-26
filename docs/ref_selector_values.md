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

## Identity And Selection

`ObjectId` remains a namespace plus framework-generated UUID nonce for one
`Serializable` lineage. `object_namespace(...)` controls new identities in a
thread/task-local context. Neither allocation nor decoding publishes authority.

Use `repo.describe(obj)` to inspect retained live identities, and `.object_view()`
to obtain state-weakened metadata. Use explicit queries to discover and choose
state; normalization does not resolve a soft alias or invent missing facets.
`IdentitySet.object_projection()` returns a fixed IdentitySet of Node object
views, merging evidence for equal projections.

`ObjectRef`, `StateRef`, `StateSelectorRef`, `ObjectSelector`, `ObjectSelectorSet`,
and their codecs are removed. See [API retirement](repo_store_v3_retirement.md).
Use [Signatures](signatures.md) for `Ref`/`Mat` conversion and quotation data, and
[Graph Querying](graph_querying.md) for Node discovery without materialization.
