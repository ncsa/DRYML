# Formats

DRYML persistent formats are closed, versioned authority formats. They are not a
safe-deserialization or hostile-input boundary: Stores, definitions, symbols, and
serialized Object payloads are trusted same-host inputs. Readers reject malformed,
incomplete, unsupported, or incompatible authority before hydration or activation;
there is no migration, conversion format, or fallback reader.

## Object And Store Authority

CDef V2 graph records contain deterministic graph-local labels, class references,
fully bound parameters, and stateful-role bits. `ObjectRef` records add canonical
primary ObjectId paths. `StateRef` records add exactly matching state-hash paths.
Private graph node tokens and Store locations are never durable identity.

DirStore format v2 publishes a `store-format.record`, digest-sharded
DefinitionRecords, explicit `stored-roots/` membership records, StateRef records,
declarations, claims, aliases, and
`local-state/<shard>/<graph-hash>/<codec>-<digest>/` directories. Each local state
has `data/`, `def.pkl`, and an exhaustive v2 `manifest.record`; readers verify
topology, role bits, manifest entries, and file hashes before state hooks run.
The state digest covers codec and payload files. The graph hash and definition
metadata select and authenticate graph-specific storage, allowing an identity fork

Definition, local-state, StateRef, declaration, claim, alias, and main-reference
records are authoritative. SQLite query indexes, record-reference indexes, caches,
and dirty markers are derived state. Rebuild may be visible and costly but must not
replace authoritative Store records. First DirStore creation uses a retained
sibling bootstrap lock derived from the canonical root; aliases of that root share
the lock. This coordinates trusted local creation and is not a distributed or
hostile-race guarantee.

## Repo Definition V1

Repo definition v1 is the inert canonical-JSON envelope
`{"schema": "dryml-repo-definition", "version": 1, ...}` for portable Repo
configuration, not Store authority or an exact-state selector. It carries an
ordered table of supported DirStore and path-backed ZipStore descriptors, default
selection, normalized routing selector descriptors, and supported declarative
settings. Paths are absolute persistent locations: a ZipStore descriptor never
contains its temporary extraction directory.

The selector grammar is closed and preserves Definition/CDef sharing, omitted
positional arguments, links, selector-as-data wrappers, supported `Par` matcher
and generator tags, symbol representation, and exact ObjectRef/StateRef leaves.
It does not permit arbitrary Python objects or live Store handles. The codec
rejects duplicate mapping keys, unknown fields/tags, invalid selector/reference
depth 32, 65,536 nodes, 4,096 entries per container, 1 MiB strings, 4,096-bit
integers, and 16 MiB encoded JSON.

`RepoDefinition.from_data()` and `from_json()` validate and detach the envelope
without opening Stores, resolving symbols, constructing selectors, activating a
session, or materializing Objects. Symbolic ImportRef/SourceSpec operands stay
symbolic; only explicit reconstruction creates live selector operands. `to_data()`
settings `config` mapping is exported when it fits this grammar, so callers must
handle exported configuration as sensitive when it contains sensitive values.
Errors identify a configuration field without rendering arbitrary input values.

`Repo.from_definition()` is the separate connected boundary. It validates all
descriptors before opening fresh existing DirStore or ZipStore handles. A DirStore
must have valid current authority; a ZipStore must be a nonempty valid committed
archive. Missing, inaccessible, wrong-type, malformed, or incompatible authority
raises `RepoDefinitionError`; reconstruction does not initialize, repair, or
silently omit storage. On failure it closes only resources it opened. On success
the returned Repo owns the fresh handles, while directly supplied Store handles
remain borrowed. Export rejects dirty archives and never commits them.

## Managed Control Formats

Managed lifecycle control is separate bounded canonical-JSON authority in the
caller-selected control DirStore. Its records are independently versioned:
`dryml-managed` v1 is the format gate, `dryml-managed-current` v2 is the current
generation snapshot, and `dryml-managed-pending` v1 is a replacement intent.
They contain operation/attempt identity, lifecycle state, interruption request,
Store/ObjectId ownership evidence, and associated checkpoint/final StateRef
digests. They never contain Object payloads, Repo handles, portable Repo
definitions, or a Python continuation.

Control publication uses short same-host locks and atomic replacement. Pending
intents and replacement snapshots are written as same-directory temporary files
before publication. POSIX publication syncs the parent directory; the Windows
adapter syncs regular files and atomic replacement without claiming unsupported
directory-descriptor fsync. Unsupported, malformed, incomplete, unreadable, or
pending control authority fails reconciliation rather than becoming a fabricated
status. There is no compatibility reader.
