# Formats

Current CDef V2 authority is closed and versioned. CDef graph records contain deterministic graph-local labels, class references, fully bound parameters, and stateful-role bits. ObjectRef records contain graph authority plus canonical primary ObjectId paths. StateRef records contain an ObjectRef and exactly matching state-hash paths. Private node tokens and Store locations are never encoded as durable identity.

DirStore format v2 writes a `store-format.record`, digest-sharded DefinitionRecords, explicit `stored-roots/` membership records, StateRef records, declarations, claims, aliases, and `local-state/<shard>/<graph-hash>/<codec>-<digest>/` directories. DefinitionRecords may be closure-only; rebuilds activate only explicit stored-root membership and roots recovered from StateRefs, declarations, main refs, or object aliases. Each local state contains `data/`, `def.pkl`, and an exhaustive v2 `manifest.record`; readers verify graph topology, role bits, manifest contents, and file hashes before hooks run. The state digest covers codec plus payload files, while the containing graph hash and definition metadata select and authenticate the graph-specific directory, allowing an identity fork to rebind unchanged payload state without changing its state hash.

Missing versions, raw CDef tuple/dict records, previous Store layouts, mutable current-state records, unsupported manifests, and old query metadata are incompatible authority. They are rejected with observed and supported versions and corrective action before hydration or mutation. No migration or conversion format exists.

The separate environment, annotation, world, runtime, and session envelope families retain their documented `v1.1` schemas; those names do not denote CDef or Store compatibility.
