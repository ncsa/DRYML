# Query Index Backend Contracts

Query indexes are derived acceleration for current Store authority. DefinitionRecords, explicit stored-root membership, declarations, StateRefs, aliases, and verified local-state directories remain authoritative. Closure-only DefinitionRecords hydrate graph nodes but do not become stored roots during rebuild. A backend must reject incompatible schema or semantic metadata before row decoding or ready activation, then rebuild only derived files from validated records.

Indexes preserve V2 CDef graph topology, typed graph paths, ObjectRef IDs, StateRef hashes, and reference occurrence paths. They must not reduce exact references to definitions. Save publication may append advisory reference rows incrementally, while rebuild replaces them from Store records. Candidate filtering may be approximate or incomplete, so authoritative Python verification always determines conflicts and complete results.

SQLite connections are process-local. Rebuild and mutation use staged activation so a reader sees a prior complete index or a new complete index. Rebuild notices are visible because scans can be long. Supported behavior assumes local SQLite and filesystem locking semantics; distributed filesystems are outside this contract.
