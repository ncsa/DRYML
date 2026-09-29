# SQLite Query Lowering

SQLite lowers safe CDef and reference predicates to candidate relations, then DRYML verifies results against authoritative V2 values. Graph paths use the current typed `Parameter` codec and query rows include current CDef/reference semantics. A mismatched query schema or semantic-version bundle is rejected before row decode, not interpreted as an empty index.

The sidecar is derived. Dirty, missing, corrupt, or incompatible SQLite files are rebuilt visibly from current Store records; no rebuild mutates definitions, StateRefs, aliases, or local-state directories. Read transactions are short and connections are process-local. WAL behavior depends on the selected local filesystem and SQLite runtime.
