# Release Notes

Status: draft.

## Selective Session And Runtime Port

This release promotes the destination implementation from `dryml.core2` to
`dryml.core` as a clean persisted-identity break, then selectively ports v1.1
formats, explicit annotations, local world planning, PID-bound runtime
publication, framework import controls, persistent sessions, and strict
definition-only orchestration. It does not replay the source branch's older
materialization, Store, or query implementation.

Persisted Python globals, symbolic references, hashes, and Store paths naming
`dryml.core2` are unsupported and receive no alias, translation, or decode
migration. Source-v1 environment, annotation, world, runtime, and ad hoc session
metadata are also rejected rather than migrated. Supported `dryml.core` V1 raw
identities and destination V2 semantic identities remain distinct as described
below.

The public session modes are `python`, `managed`, and `orchestrator`.
Orchestration permits definition, graph, reference, query, CDef hydration, and
derived-index recovery operations while rejecting supported live-Object paths
before side effects. Environment/world/runtime annotations remain declarations;
world/runtime requirement axes are retained for parity but have no automatic
consumer in this release.

A process must be restarted when a watched framework was imported before a
required visibility-changing transition, after terminal publication failure,
or after unsafe activated runtime state is inherited across `fork()`. Session
configuration does not add dispatch, future-worker state, probes, direct-call
wrapping, records/provenance, or automatic requirement inference.

A broader migration inventory, compatibility review, and third-party
diagnostic/security hardening pass are deferred until the pre-release
compatibility review. Keep backups and test Stores with the exact software
version that will open them.

## Bound Concrete Definitions (V2)

New exact `ConcreteDefinition` identities are V2 fully bound semantic
constructor records. They capture declared defaults, normalize equivalent
positional and keyword calls, and expose canonical values through direct
non-reserved attributes, `parameters`, and semantic graph paths such as
`$[@param("model")]`. `parameters[name]` remains the collision-safe access
surface when a constructor name conflicts with a CDef API member.

`Definition`, `Selector`, and `SearchSpace` remain partial expressions:
their semantic mappings include supplied fields only, and omitted selector
fields do not constrain V2 candidates. Semantic inspection, graph extraction,
hashing, query planning, and index rebuilding read V2 parameter records without
resolving the referenced class. V2 `.args` and `.kwargs` are compatibility
accessors that project the persisted record using the current class signature
at materialization time; they may resolve/import that class and can report a
current-signature error.

Supported `dryml.core` V1 records remain readable with their original raw `cls`/`args`/`kwargs`,
hashes, and paths. They are not migrated, equated, or substituted with V2.
Symbolic V1 records can be inspected without resolution, while raw-class V1
pickles retain their normal import requirement. DRYML does not recover a
historical omitted default from a V1 record.

### Store Support Matrix

| Operation | New DRYML software | Old DRYML software |
| --- | --- | --- |
| Read supported `dryml.core` V1 Store | Supported | Supported |
| Read V2 or mixed V1/V2 Store | Supported | Unsupported |
| Write a new exact identity | Writes V2 | May write only an untouched V1-only Store |
| Downgrade after V2 authority exists | Restore a pre-V2 backup | Unsupported in place |

This matrix does not include persisted `dryml.core2` values, which are rejected.

Object roots and mutable Store references are staged and atomically replaced
only after validation. SQLite query sidecars are derived state and rebuild from
authoritative object files; failed rebuilds do not replace a valid sidecar or
expose a partial ready index. These guarantees apply to supported local Store
filesystems and do not extend to arbitrary `IOBase` implementations or
unsupported filesystem semantics.

## SQLite Query Index Sprint

The core repo/query path now supports Store-owned persistent SQLite query indexes for `DirStore`.

Highlights:

- `DirStore(query_index="auto" | "sqlite" | "memory" | "none")` controls query-index backend policy.
- SQLite indexes persist graph nodes, local feature postings, direct definition edges, stored-root membership, semantic versions, and generation metadata.
- Repo queries federate Store indexes and the live cache overlay while preserving Store-priority replica metadata.
- Existing ready SQLite indexes support stored and nested queries without hydrating every Store root.
- Saves publish object files before activating roots in the index; dirty markers and reconciliation recover object/index divergence.
- Processes use process-local SQLite connections and committed writes become visible to other read transactions.
- Index status, validation, rebuild, and reconciliation are exposed through `Repo` and `DirStore` administration APIs.

Known deferred work:

- Public query-backed paginated ResultSets are not implemented yet.
- SQL-native candidate relations are deferred.
- Online side-by-side rebuild with manifest switching is deferred.
