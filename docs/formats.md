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

DirStore format v3 publishes a `store-format.record`, digest-sharded
DefinitionRecords, explicit `stored-roots/` membership records, declarations,
claims, aliases, lineage/current-metadata records, and complete snapshot
directories. A snapshot is authoritative only when
`snapshots/<shard>/<state-ref-digest>/` contains matching `state-ref.record`,
`placement.json`, `metadata.json`, and `snapshot.json` siblings. Its
`local-state/<graph-hash>/<codec>-<digest>/` directories contain `data/`,
`def.pkl`, and an exhaustive `manifest.record`; placement can instead name exact
routed child StateRef projections. Readers verify association IDs, complete state
coverage, topology, role bits, manifest entries, and file hashes before state
hooks run. Metadata-only readers validate the snapshot association without
opening payload bytes.

The state digest covers codec and payload files. The graph hash and definition
metadata select and authenticate graph-specific storage while an identity fork
can preserve validated payload bytes and rebind exact ObjectIds. Claims and
mutable aliases do not enter CDef, ObjectRef, or StateRef identity.

Snapshot metadata is a `dryml.record.v1.1` envelope with kind
`dryml.core.snapshot_metadata`; lineage and current annotation records use the
same generic envelope family. Snapshot capture is immutable and write once.
ObjectRef/StateRef current mappings are separate atomic whole-map LWW records.
Environment and requirement values retain their closed v1.1 domain envelopes.

Store format v2 is unsupported. Opening a v2, malformed, incomplete, or mixed
layout raises an authority error without rewriting it. There is no in-place
migration, dual reader, or fallback to top-level `state-refs/` or shared
`local-state/` authority.

Definition, snapshot directory, local-state payload, lineage, current annotation,
declaration, claim, alias, and main-reference records are authoritative. SQLite
query indexes, metadata projections, record-reference indexes, caches, and dirty
markers are derived state. Rebuild may be visible and costly but must not replace
authoritative Store records. DirStore durably marks query-visible definition,
stored-root, declaration, and alias changes dirty under its writer fence before
publishing authority; failed authority publication may leave a harmless marker,
while idempotent writes publish none. First DirStore creation uses a retained
sibling bootstrap lock derived from the canonical root; aliases of that root share
the lock. This coordinates trusted local creation and is not a distributed or
hostile-race guarantee.

Path-backed ZipStore contains the same v3 tree in one buffered transaction.
Completing an extracted snapshot does not commit archive authority;
`ZipStore.commit()` validates and atomically replaces a non-stale archive.
Borrowed snapshot/payload paths point into the extraction and expire on
`ZipStore.close()`.

Direct Store and derived-index publication use `dryml.filesystem`: staged file
contents and snapshot trees are flushed before names change. POSIX persists
changed directory entries after atomic replacement or native exclusive rename.
Windows uses same-volume `MoveFileExW` publication with
`MOVEFILE_WRITE_THROUGH`; missing components are installed through write-through
moves, and logical deletion first moves the authoritative name to an ignored
sibling tombstone before best-effort cleanup. Existing `.store-removed-` residue
from the earlier Store-private adapter remains unrecognized as authority and is
still excluded from ZipStore commits. Extended Win32 spellings remain private.
The SQLite index uses the same public concern for dirty tokens and canonical
sidecar replacement and never reaches through an owning Store for native helpers.
Publication fails explicitly when the required local-filesystem primitive or
barrier is unavailable; exceptions can follow visible publication and never
assert rollback. See [Filesystem Publication](filesystem.md).

The checked-in `tests/fixtures/store_v3/` directory and committed archive carry
fixed public synthetic timestamps, identities, environment fields, requirement
outcomes, annotations, local payloads, and routed-child placement. Its manifest
records byte hashes and semantic expectations. This beta fixture commitment
covers framework-owned v3 records and associations, not arbitrary author codec
migration or Python API stability.

Fixture authority is checked out byte-for-byte via `.gitattributes`, without
line-ending conversion. Git for Windows needs long-path support to check out the
snapshot directory fixtures; CI enables `core.longpaths` for its Git processes.

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
the returned Repo owns fresh noncached handles, while directly supplied and active
Session-cache Store handles remain borrowed. Export rejects dirty archives and
never commits them.

The same per-Store descriptor grammar is available through `Store.to_definition()`
and `Store.from_definition()`. It is a detached opening instruction, not a new
durable Store format or a snapshot of Store contents. Its only supported kinds are
`dir` with an absolute path and built-in query-index policy, and path-backed `zip`
with an absolute archive path. Store reconstruction validates existing authority;
it never creates, repairs, or commits it. A dirty or file-like ZipStore cannot be
exported, although an already cached dirty path-backed ZipStore can remain one live
local transaction for a matching open request. Session resource-cache teardown
discards cache-owned buffered Zip work without committing it.
See [Repos and Stores](repos.md) for reconstruction ownership and
[Session](session.md) for cache lifetime; [Generic Execute](execute.md) accepts
only the narrower direct-DirStore worker transport.

## Artifact Value V1

`Value` local state contains one trusted dill protocol-5 `value.pkl` envelope:
`{"format": "dryml.artifacts.value", "version": 1, "present": bool,
"result": object}`. The four keys are exact, `version` is the exact integer
`1` rather than a boolean, and an absent result has `present: false` with
`result: null`. A present `null` result is valid and distinct from absence.

Readers reject missing, corrupt, incomplete, unknown-format, and unknown-version
payloads before replacing an existing result. The format contains only the result
envelope, not input references, live iterators, managed controls, or transient
computation state. It is a framework-owned beta format commitment; subclasses own
any separate hook files and their domain validation. As with all pickle payloads,
the reader accepts trusted same-host data only and is not safe deserialization for
hostile input.

The checked-in `tests/fixtures/artifact_value_v1/` fixtures identify this v1
envelope with an independent manifest and byte hashes. They exercise reader
compatibility only: they do not serialize a source Dataset/model, iterator,
accumulator, or managed-control record, and they do not define a compatibility
format for subclass-owned files.

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
