# Formats

Current CDef V2 authority is closed and versioned. CDef graph records contain deterministic graph-local labels, class references, fully bound parameters, and stateful-role bits. ObjectRef records contain graph authority plus canonical primary ObjectId paths. StateRef records contain an ObjectRef and exactly matching state-hash paths. Private node tokens and Store locations are never encoded as durable identity.

Repo-definition v1 is a separate inert canonical-JSON envelope (`schema: "dryml-repo-definition"`, `version: 1`) for portable Repo configuration. It records an ordered descriptor table for supported DirStore/ZipStore locations, the default-table index, normalized routing selector descriptors, and bounded declarative settings. Its closed selector grammar preserves Definition/CDef sharing, omitted positional arguments, links, supported `Par` tags, symbol representation, and exact ObjectRef/StateRef descriptor leaves without resolving or materializing them. Mapping and JSON inputs reject duplicate keys, unknown fields/tags, invalid selector/reference topology, non-finite or non-JSON values, and values above the shared depth/node/container/string/integer limits or 16 MiB encoded limit. Repo definitions are not Store authority and cannot yet recreate connected resources; existing-only reconstruction is deferred to U6.

DirStore format v2 writes a `store-format.record`, digest-sharded DefinitionRecords, explicit `stored-roots/` membership records, StateRef records, declarations, claims, aliases, and `local-state/<shard>/<graph-hash>/<codec>-<digest>/` directories. First creation serializes root creation and atomic format-gate publication with a durable sibling `.dryml-bootstrap-<sha256(os.fsencode(normcase(realpath(root))))>.lock` advisory file; it is derived coordination state outside the authority root, is retained rather than unlinked, and aliases of the canonical root share it. DefinitionRecords may be closure-only; rebuilds activate only explicit stored-root membership and roots recovered from StateRefs, declarations, main refs, or object aliases. Each local state contains `data/`, `def.pkl`, and an exhaustive v2 `manifest.record`; readers verify graph topology, role bits, manifest contents, and file hashes before hooks run. The state digest covers codec plus payload files, while the containing graph hash and definition metadata select and authenticate the graph-specific directory, allowing an identity fork to rebind unchanged payload state without changing its state hash.

Missing versions, raw CDef tuple/dict records, previous Store layouts, mutable current-state records, unsupported manifests, and old query metadata are incompatible authority. They are rejected with observed and supported versions and corrective action before hydration or mutation. No migration or conversion format exists.

Environment, world, runtime, and session format families retain their documented
schemas; those names do not denote CDef or Store compatibility. Annotations have
no kernel-owned envelope, ID, serialization, or persistence format.

Managed lifecycle control is separate bounded canonical-JSON authority in the
selected control Store: `dryml-managed` v1 gate, `dryml-managed-current` v1
current snapshot, and `dryml-managed-pending` v1 replacement intent. It records
operation/attempt identity, lifecycle state, interruption request, and associated
checkpoint/final StateRef digests, never state payloads or a Python continuation.
Unsupported, malformed, incomplete, unreadable, or pending control data or path
component fails reconciliation; it is not interpreted as completed or not-started
work. These v1 records have no
migration or compatibility reader.
