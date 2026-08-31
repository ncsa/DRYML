# Query Index Backend Contracts

Query indexes are derived acceleration for current Store authority. DefinitionRecords, declarations, StateRefs, aliases, and verified local-state directories remain authoritative. A backend must reject incompatible schema or semantic metadata before row decoding or ready activation, then rebuild only derived files from validated records.

Indexes preserve V2 CDef graph topology, typed graph paths, ObjectRef IDs, StateRef hashes, and reference occurrence paths. They must not reduce exact references to definitions. Candidate filtering may be approximate, but authoritative Python verification determines results.

SQLite connections are process-local. Rebuild and mutation use staged activation so a reader sees a prior complete index or a new complete index. Rebuild notices are visible because scans can be long. Supported behavior assumes local SQLite and filesystem locking semantics; distributed filesystems are outside this contract.
