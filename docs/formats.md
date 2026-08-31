# Formats

Current CDef V2 authority is closed and versioned. CDef graph records contain deterministic graph-local labels, class references, fully bound parameters, and stateful-role bits. ObjectRef records contain graph authority plus canonical primary ObjectId paths. StateRef records contain an ObjectRef and exactly matching state-hash paths. Private node tokens and Store locations are never encoded as durable identity.

DirStore writes a `store-format.record`, digest-sharded DefinitionRecords, StateRef records, declarations, claims, aliases, and `local-state/<shard>/<graph-hash>/<codec>-<digest>/` directories. Each local state contains `data/`, `def.pkl`, and an exhaustive `manifest.record`; readers verify graph topology, role bits, manifest contents, and file hashes before hooks run.

Missing versions, raw CDef tuple/dict records, previous Store layouts, mutable current-state records, unsupported manifests, and old query metadata are incompatible authority. They are rejected with observed and supported versions and corrective action before hydration or mutation. No migration or conversion format exists.

The separate environment, annotation, world, runtime, and session envelope families retain their documented `v1.1` schemas; those names do not denote CDef or Store compatibility.
