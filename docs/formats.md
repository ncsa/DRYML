# Formats

DRYML persistent formats are closed, versioned authority formats. They are not a
safe-deserialization or hostile-input boundary: Stores, definitions, symbols, and
serialized Object payloads are trusted same-host inputs. Readers reject malformed,
incomplete, unsupported, or incompatible authority before hydration or activation;
there is no migration, conversion format, or fallback reader.

## Object And Store Authority

### Shared Node Graph V1 / CDef Identity 3

`dryml.core.node_codec` introduces a separate, closed JSON-compatible family with
`format="dryml-node-graph/v1"` and exact integer `identity_version=3`. The packet
contains authenticated `tables`, authenticated `values`, and compact `views`.
Domain/version tags are explicit SHA-256 preimage components; digests are lowercase
64-hex text. The domains are `dryml-node-table/v1`, `dryml-node-values/v1`,
`dryml-node-value/v1`, `dryml-node-value-atom/v1`, `dryml-node-value-step/v1`,
`dryml-node-topology/v1`, and `dryml-node-graph/v1`.
Semantic certificates carry `dryml-node-exact/v1` and the explicit view name.

Each table has unique occurrence `rows`, local `overlays`, and declared root-view
slots. Occurrences may have several immutable local construction structures, for
example supplied soft and normalized concrete views of the same occurrence.
Overlays select an occurrence/structure, exact identity/state facets, typed target
view slots, and an optional retained construction handle with required
`view="construction"`. Tagged immutable values, including shared symbolic operands
and array bytes, are interned into value blocks with authenticated block references.
Pure-value sharing is a storage optimization, not Node occurrence identity; exact
comparison normalizes it without expanding the value DAG. A graph link is a local
slot reference; no root view contains a
complete descendant graph export. Neither local slots nor private process tokens
are semantic identities.

A compact root view carries its authenticated table hash, root slot, semantic-view
name, `key_format="dryml-node-exact/v1"`, and expected exact key. The reader
authenticates blocks, validates all declarations/links and reachability, rejects
cycles and invalid view kinds, realizes fresh occurrence/view tokens, and verifies
advertised root keys with shared preparation. Multiple roots can share a table;
the selected semantic key covers only its reachable view, so unrelated declared
roots, slot placement and block placement do not alter it. Retained construction
views preserve the original explicit selections alongside realization facets.
Construction handles must select the associated occurrence and the declared view
kind. General metadata can retain a supplied soft view across concretization;
the later creation-policy boundary owns normalized creation-authority admission.

The grammar preserves list/tuple/set/frozenset distinctions, exact string/integer
map keys (not bool keys), symbolic ImportRef/SourceSpec values, supported symbolic
arithmetic/repetition and Match leaves, factories, immutable Core semantic leaves,
references, quotations, ObjectIds and LocalStates. Nested reference/state tags
carry their own exact integer version 1, independently checked on read. There is
no opaque serialization fallback for unsupported live/custom values or nonportable
callbacks. Symbol-addressable `Satisfies` predicates carry explicit import/source
evidence and resolve only when the existing matcher is called, not during metadata
decoding. A descriptive predicate name alone does not replace its exact callable
evidence. Arrays are nonobject snapshots with dtype (including structured
offsets/alignment/subarrays), shape, byte order and logical contents; structure
padding (including x87 extended-precision ABI padding) is canonicalized, not
treated as constructor data. Extended float/complex layouts carry numeric ABI
evidence so a different long-double interpretation is refused on read. NumPy
scalar codecs preserve empty strings and trailing NUL content as well as numeric
and structured scalar values.
Registered nonobject NumPy extension dtypes retain an explicit importable scalar
type, name, byte order, item size and format string rather than degrading to a
void dtype. Their type is resolved lazily on read and its representation is
revalidated; an unavailable or incompatible extension is rejected.
Managed/variable-width storage dtypes require a separate interpretation codec and
are rejected before raw storage bytes can enter this fixed-storage format.

Embedded ObjectRef bindings name target occurrences (or imported ObjectIds), and
StateRef selections name ObjectIds. These unordered bindings are canonicalized
with the complete graph. Context-local legacy SetMember ordinals/fingerprints do
not enter their semantic preimages. The existing ObjectRef constructor retains
pure target-binding facts while validating its public path map; the new reader
reconstructs that map in the decoded graph's path context. This preserves pins and
cross-boundary aliases even when fresh tokens change unordered path ordinals.
Query-only ObjectSelectors have an explicit version-1 tag retaining their query
definition and ObjectRef association as graph-aware shared-table values. This
preserves recursive state-weakened selectors, directly constructed exact pins,
and their occurrence/ObjectId bindings without nested rooted exports. Node graph
transport does not admit these query values as legacy CDef/Store authority.
The legacy V2 reference data layout remains separate; it is not admitted as
format-4 Store authority.

Readers reject unknown fields/tags, invalid slots/block references, unused
declarations, bad authentication, incompatible CDef identity/key versions and
semantic-key mismatches without rewriting the source. V2 records are not accepted
as identity-3 records or converted. The V2 reader likewise rejects new packets.
The new reader performs metadata allocation only, with explicit enum/extension
dtype type codecs resolving those value types; it does not materialize targets,
read private state or open Stores.

Creation/compatibility policies use the same preparation/encoding. The format-4
Store APIs below consume these blocks through their mutation journal.
Repo/reference consumer integration, new query coverage, active-use leases and
managed-format cutovers remain separate work.

### Store Format 4: CDef-Owned State And Metadata Readers

New DirStores have a `store-format.record` containing exact integer
`format_version=4`, `identity_version=3`, and a fresh persistent canonical
`store_uuid`. Reopening preserves that UUID. Existing format-3 (and older/future)
Stores are rejected before authority effects. There is no conversion or dual
reader. The format gate retains its closed trusted Store-record frame; the new
Node/state families use canonical JSON frames:
`DRYML-METADATA/<kind>/1\n<canonical-json>`.

Every new record, including nested records, requires its exact `kind`, integer
`version=1`, `domain="<kind>/v1"`, and integer `identity_version=3`. Families are
separately named: `dryml-state-key`, `dryml-node-blocks`, `dryml-node-root`,
`dryml-state-variant`, `dryml-state-availability`, `dryml-node-membership`,
`dryml-declaration-anchor`, `dryml-declaration-retired-anchor`,
`dryml-declaration-required`, `dryml-node-name`, `dryml-store-authority`, and
`dryml-store-mutation-intent`. Legacy
`local-state-manifest` v2/v3, V2 definitions and old reference records do not pass
these readers, even beneath a forged current Store marker. Duplicate/noncanonical
JSON, extra fields, trailing data, incompatible nested versions/domains, bad
authentication and path/semantic-key mismatches are errors, never repair inputs.

```text
store-format.record
authority.record                            # Store UUID / committed epoch / pending UUIDs
mutation-prepare.record                     # bounded intent-registration crash bridge, when present
mutations/<mutation-uuid>.record             # separately named v1 intent / classification / replay
definitions/blocks/<hh>/<block-digest>.record  # shared Node tables and values
definitions/roots/<hh>/<recipe-key>.record    # compact recipe root view
graphs/<hh>/<saved-key>.record                # compact exact saved root view
states/<first-two-cdef-hex>/<cdef-digest>/
  def.pkl                                   # compact construction root view
  <codec>-<content-digest>/
    manifest.record                         # full StateKey + codec/inventory
    availability.record                     # publication UUID and status
    data/...                                # only codec-private files
membership/{recipe,saved}/<hh>/<root-key>.record
declarations/{anchors,required}/<hh>/<object-id-address>.record
names/<name>.record                         # bounded CAS to an explicit recipe/saved root
```

`def.pkl` is framed root-view metadata, not arbitrary pickle. It names a shared
block digest, authenticated table/slot, required creation-view policy, key format
and expected exact semantic key. One `install_creations()` batch exports one
shared Node table for its roots; headers do not embed descendant graph blobs.
Readers authenticate shared blocks through `node_codec`, validate the selected
semantic policy and carried creation digest, and return detached metadata.
Equivalent block placement does not change creation identity or replace an
existing semantically equal header. Conflicting headers raise an authority error.

The StateKey is `(cdef_digest, codec_hash)`; addressing uses the carried strings,
without hashing or discovering a graph. CodecHash uses `dryml-codec-content/v1`
and authenticates codec identity, positive integer codec version, canonical JSON
interpretation metadata, normalized sorted file inventory, sizes and file hashes.
ObjectIds, creation digest, producer identity, time and Store paths are excluded.
Same-full-key installations reuse one variant; equal codec bytes under distinct
CDefs remain independently installed and available. Codecs receive only `data/`.

Metadata knowledge, explicit recipe/saved membership and available data are
different facts. A header alone publishes neither local state nor root membership.
Manifests, headers and shared blocks remain inspectable when `data/` is missing.
Availability requires a validated full-key publication record with `published`
status plus a real data directory, not eager byte integrity or future retention.
Reinstallation has a fresh incarnation: `prepared` status prevents an interrupted
data reinstall from advertising completion. A retry authenticates already visible
same-key data before publishing its new incarnation.

`inspect_states()` is a direct **single-bucket inventory**, not indexed paging. It
decodes the CDef once outside writer-blocking fences, reads no payload bytes,
and constructs no Objects. Short scopes capture/revalidate the Store cut and exact
metadata byte/directory witnesses. Changed evidence raises `StoreCutChanged`
instead of mixing cuts or silently restarting reads. `read_state()` validates fresh header/manifest/data
evidence, optionally validates destination/codec/codec-version compatibility before payload IO,
and lends only private data within an owner-checked context. Context exit or Store
closure invalidates the borrowed source; detached LocalState/manifest metadata
survives. This is not an active-use or retention lease.

`publish_node()` validates ObjectId-addressed declaration anchors using canonical
`dryml-object-projection/v2` evidence and source contributions, then publishes
explicit membership last. Legal state-pin changes share an anchor; configuration
conflicts fail before publication effects. A separate immutable declaration witness
makes an established missing anchor an authority error, including when a new
source would otherwise appear to declare a fresh identity. Published-root reads
also validate their required anchor/source associations. Metadata-only construction
installation does not declare the identities present in its constructor evidence.

### Store Mutation Journal V1

Every supported query-visible built-in mutation enters one Store-owned protocol:
creation/state installation, root/membership/declaration publication, membership
withdrawal and Node-name CAS/delete/rename. The old main/ObjectRef/StateRef alias,
snapshot, claim and annotation writers remain explicit refusals. Node names are
separate format-4 authority, not a second V2 reference persistence path.

`authority.record` holds the persistent Store UUID, committed nonnegative epoch,
and exact unacknowledged mutation UUIDs. Each separately named intent stores its
stable mutation UUID, Store UUID/base epoch, affected scopes and paths, expected
before/after bytes, owned staging tokens, confirmed relative publication
references, actual resulting epoch, and replayable projection work. Byte evidence
uses canonical base64 within closed JSON, not pickle or user callbacks. File
projection rows carry exact record bytes; variant rows carry exact manifest,
availability incarnation and data-presence evidence. Multiple transitions to one
mutable record coalesce to one actual projection row. This is the batch seam for
delta/coverage consumers; the journal alone does not certify SQLite coverage.
Enabled persistent completion applies that work and coverage in one SQLite
transaction, rereads committed receipts, then acknowledges exact captured work.

Preparation, graph encode/decode/equality, compatibility validation and payload
authentication run outside the Store writer fence. A detached batch carries the
Store cut and exact byte/directory witnesses, revalidated under that fence before
effects. Stale write preparation retries a bounded number of times, then explicitly
refuses instability. The fence covers bounded byte/association checks, prepared
installs, cut publication, the short prepared SQL transaction and exact
acknowledgement. Key protection must precede it when added;
callers must not retain an external writer/read fence around high-level graph APIs.

Before effects, persist the bounded `mutation-prepare.record` bridge, the named
intent, then its pending registration in `authority.record`. Remove the bridge
durably only after registration. Recovery can adopt that exact bridge without
scanning historical intents. Native filesystem helpers flush staged files and
changed directory entries; rename visibility alone is not completion. Each complete
variant child installs independently; a populated CDef bucket is never replaced.
`def.pkl` equality is verified semantically before the fence. A no-op at a current
cut creates no intent, epoch or projection churn.

After effects, classify actual affected records and persist classification **before**
the resulting authority cut. An interrupted cut publication can finish from that
persisted epoch without incrementing twice. An error after visibility is a partial
failure, never a rollback: `StoreMutationError` (also an `OSError`) exposes
`mutation_id`, `cut`, `confirmed_refs`, `index_disposition="repair_pending"`, `recovery_attempt`, and
the original cause. Cancellation propagates with the same evidence attributes.
Unconsumed staging and authoritative partial work are preserved.

`recover_authority()` classifies exact persisted work only, without graph preparation,
payload IO, capture, restore, body execution or deletion of authoritative files.
It leaves work pending; it does not claim missing transitions happened. This
authority boundary remains usable independently of SQLite compatibility or
availability.
`replay_mutation(uuid)` can finish prepared transitions from their exact persisted
evidence. It authenticates any required surviving staging outside the writer fence,
re-persists already visible metadata/parent entries, journals repair, and refuses
a later Store epoch, changed authority or missing staging rather than
recapturing or overwriting a later writer. `pending_mutations()` exposes exact
projection work; `acknowledge_mutation(uuid, epoch)` acknowledges only that captured
classified mutation/cut, requiring its exact committed SQLite receipt when enabled.
Later pending work remains pending even after an earlier
repair. Missing/corrupt/future journal records are refused intact, not reset.

Withdrawal removes only the withdrawing root's membership and anchor contributions;
it does not reclaim data or delete graph/manifest/projection history. A last-source
anchor becomes a separately named retired-anchor v1 record at the same path.
Emptying an active-anchor record is still corruption. A still-named root cannot be
withdrawn. Name replacement/deletion/rename uses bounded mutable CAS; interrupted
rename can leave both complete names, with that actual cut recorded for replay.

New initialization installs the initial authority cut before the format gate.
An interrupted gate write can resume only from a sole, valid epoch-zero/no-pending
authority record, preserving its UUID and bytes. A marked Store missing its cut is
refused; the reader does not invent an empty replacement. Old Store formats and
pre-journal format-4 roots without the required cut are not converted.

Legacy Repo snapshot writers and Query V3 views/coverage remain explicit format-4
refusals. The Store-owned persistent projection uses **SQLite schema 10 / coverage
contract 1**. It keeps source records, independent recipe/saved memberships,
declarations, creation compatibility prefilters, full-StateKey manifests and
availability, typed block/table/overlay postings, and names distinct. Capability
entries for requested metadata, retention and lifecycle remain explicitly
unavailable until their owning adapters integrate their authority and guards.
Neither old schema 9 nor incomplete coverage is ready. Owned derived state can be
rebuilt from compatible metadata without payload reads or source-format changes;
foreign/future sidecars are refused intact. A detected incomplete authority cut
invalidates old derived coverage and fails rebuild rather than certify omissions.

`none`/`memory` have no persistent-maintenance obligation; `auto` enables usable
SQLite, with declared nonpersistent fallback only for optional-backend
unavailability. Explicit `sqlite` never falls back. Derived
`.dryml/query-index-destination.record` binds one normalized persistent
destination/settings to the Store UUID, checked across handles/processes and on
cached-backend reuse before effects. `mutation_index_disposition` reports policy,
not a current certificate. Successful ordinary nonpersistent mutations acknowledge
their exact work synchronously; their explicit prepared-batch seam leaves work
pending for its owner. Enabled prepared batches complete projection and exact ack
before return. Repo consumers must use Store-owned APIs, not mutate bucket paths.
See [Query Index Backend Contracts](query_index_backend_contracts.md) for readiness,
changed-record replay, failure recovery and maintenance-free index reads.

ZipStore has the same logical tree in one buffered extraction. State installation
receipts say `durability="buffered"`; durable archive completion requires a
successful `commit()`. The extraction retains nonpersistent indexing and makes no
durable SQLite projection claim. Archive conflicts/barrier failures propagate and retain
dirty/partial evidence. `committed_archive_cut` remains the last confirmed durable
archive cut (or None before first commit). `ZipStoreCommitError` carries the stable
commit mutation identity, buffered cut/confirmed references, previous confirmed
archive cut and `durability="buffered"`. Failure before replacement leaves reopening
on the old archive; failure after visibility is explicitly ambiguous. Retry accepts
only the original baseline or this handle's exact attempted archive bytes, and
repeats the native persistence boundary. It never silently rebases onto unrelated
archive contents. Closing discards the extraction and expires borrowed
sources. File-like ZipStores are read sources only and refuse writable publication.
Archive-open baseline hashes and inode evidence come from the same open file as
the extraction, so a concurrent atomic archive replacement cannot rebase an old
extracted view. Unpublished `.staging/` files are excluded from archive commit.
Publication capabilities explicitly distinguish `durable`, `buffered`, `read-only`
and undeclared durability; `require_durable()` refuses a buffered completion claim.
New consumer preflight passes `record_family="dryml-node-graph/v1"` independently
of its diagnostic operation name; omitted/legacy families fail before hooks.

### Historical Store V3 / Remaining CDef V2 Consumer Codecs

The following describes the old Store family for contrast and the codecs still
used by consumers awaiting their cutover. **New built-in Stores neither open V3
authority nor publish this layout.** Standalone legacy codec admission is not
current Store admission.

CDef V2 graph records contain deterministic graph-local labels, class references,
fully bound parameters, and stateful-role bits. `ObjectRef` records add canonical
primary ObjectId paths. `StateRef` records add exactly matching state-hash paths.
Private graph node tokens and Store locations are never durable identity.
Legacy CDef-to-CDef comparison retains the reference/quotation value scopes that
V2 transports independently. It compares normalized occurrence bijections and
complete values, including embedded Node facets, rather than legacy hashes.
Comparisons involving an identity-3 Node root retain full cross-boundary topology.
Legacy V2 export/digest paths refuse Node-local facets and retained construction
handles that they cannot represent. Unbound V2 consumer bytes remain unchanged.

Core owns the portable `dryml-reference-json` v1 envelope for embedding exact
`ObjectRef` and `StateRef` authority in closed JSON formats. It tags canonical
Definition atoms, frozen containers, symbols, factories, tensor specifications,
and reference identity without loading Objects or state. Store authority records
continue to use their separate trusted dill frame; `StateRef.to_data()` alone is
a token-neutral Python record and is not promised to be JSON-compatible.

DirStore format v3 previously published a `store-format.record`, digest-sharded
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

## Fold Progress V1 And V2

`fold-progress.pkl` is a trusted pickle envelope with exactly `format`, `version`,
`processed_count`, and `progress`. Format is `dryml.artifacts.fold-progress`;
version is the exact integer 1 or 2, and processed count is a nonnegative exact
integer. `progress=None` represents no remaining continuation (uncomputed or
completed); a separate `value.pkl` v1 envelope determines result presence.

Non-null progress has positive `count` equal to processed count and the existing
fields `state`, `carry`, `source_binding`, `source_spec`, `observed_spec`,
`carry_spec`, `observation_devices`, `program`, `backend_devices`, `operation_id`,
and `attempt_id`. Specs, tagged lossless host carry, backend/device records, and
source/program/attempt compatibility checks retain their existing meanings.

| Version/state | Additional fields | Recovery |
|---|---|---|
| v1 active | None | Compatible positional resume |
| v1 exhausted | None | Loadable ambiguous state; explicit fresh managed rerun required |
| v2 active | None (`eof` forbidden) | Compatible positional resume |
| v2 exhausted | Exactly `eof="root-output-eof"` | Trusted source-free finalization retry |

Only positive root-output termination authorizes new exhausted progress; closure,
failure, child EOF, and strict replay shortfall are not evidence. Unknown versions,
extra/missing fields, invalid EOF tokens, and inconsistent carry/count evidence
fail restoration before assignment. Valid legacy exhaustion is rejected at Fold
body admission rather than in restore, so the receiver remains usable for explicit
rerun. Unchanged restored v1 envelopes save as v1; new checkpoints and completed
computations write v2. No existing immutable snapshot is rewritten.

`tests/fixtures/fold_progress/` freezes synthetic v1 active/exhausted/completed and
v2 active/exhausted payload bytes, wrapped in base64 for byte-preserving checkout.
One manifest records decoded and encoded SHA256 hashes, semantic expectations,
and identity rebinding for synthetic managed tests. Its independent generation
recipe never runs in reader assertions. A frozen v1 validator from `ba368afc`
qualifies v2 rejection at its version gate **before carry restoration**.
Downgraded readers cannot resume v2 progress. Choosing a retained earlier
compatible snapshot may replay work; this is neither automatic rollback nor
repair. Historical completed truncated Values cannot be identified automatically.

## Symbolic Definition And Generator Selector V1

Quoted symbolic Definition data uses the closed
`{"schema": "dryml-definition-expression", "version": 1, ...}` graph format.
It preserves typed scalar values, symbolic targets, graph aliases, parameter,
arithmetic, and repetition nodes, factories, exact references, and `QuotedDef`
quotation without importing a target, constructing an Object, or calling a
factory. The format rejects cycles, unknown tags/versions, dangling or duplicate
labels, non-finite values, live handles, custom providers, and payloads over 16
MiB.

`GeneratorSelector` retains the established closed
`{"schema": "dryml-template", "version": 1, "kind": "template-selector", ...}`
payload tag and canonical bytes for compatible built-in integer-range and
finite-choice distributions. It decodes directly into Definition-based state.
Runtime providers and `Generator` instances are nonportable. The retained
`dryml-template` name is a format-compatibility tag only; it does not restore a
runtime Template value.

Fresh `Ref[Definition]` and Template-role constructor values persist as a Ref
link to `QuotedDef`. The quoted payload is terminal local data, opaque to
materialization and ordinary symbolic traversal unless an operation explicitly
opts into Ref traversal. Existing Store-v3 authority is unchanged; query sidecar
version changes invalidate derived indexes only and never rewrite definitions or
Store records.

## Workflow Records

`ExperimentData` persists its authoritative `experiment_data.json` as the closed
`dryml-experiment-data` v3 format. It stores declared column kinds, ordered row
facts, exact `ObjectRef`/`StateRef` values, artifact-reference mappings, and
status cells without opening an Experiment, checkpoint, Dataset, or Artifact
payload. V3 delegates every embedded exact reference to core's versioned
`dryml-reference-json` envelope. The reader retains the closed v1 and v2 value
grammars, and an existing history migrates under its established current-history
alias when its next immutable v3 snapshot publishes.
Pandas is only a lazy inspection dependency; it is not part of this format or a
reader requirement.

Completed `Value` Artifacts retain their result through the
`dryml.artifacts.value` v1 envelope. A `TrainState` is ordinary Object state, not
a separate file format: its named state and supported historical three-slot state
are validated by its state hooks before restored progress is installed. Experiment
artifact mappings persist individual Template-role Definitions as quotation data
in canonical key order; there is no aggregate recipe-container format.

The small `tests/fixtures/qualification_reader_v1/` fixtures are reproducible reader inputs
generated by `generate.py`. Use `PYTHONPATH=src python
tests/fixtures/qualification_reader_v1/generate.py --output <empty-directory>` to make a
clean candidate; omitting `--output` deliberately regenerates the source
directory. The ExperimentData and Value fixtures exercise their actual readers.
`TrainState` is a code-free named `__setstate__` compatibility vector, not a
portable pickle codec claim. The manifest schema pins every payload filename and
SHA-256 before reader tests decode it. The fixtures contain no backend objects,
TFDS content, Store snapshots, or live payloads.

## Repo Definition V1

Repo definition v1 is the inert canonical-JSON envelope
`{"schema": "dryml-repo-definition", "version": 1, ...}` for portable Repo
configuration, not Store authority or an exact-state selector. It carries an
ordered table of supported DirStore and path-backed ZipStore descriptors, default
selection, normalized routing selector descriptors, and supported declarative
settings. Paths are absolute persistent locations: a ZipStore descriptor never
contains its temporary extraction directory.

The selector grammar is closed and preserves Definition/CDef sharing, omitted
positional arguments, links, selector-as-data wrappers, supported `Match` tags,
symbol representation, and exact ObjectRef/StateRef leaves. Symbolic Definition
quotation and generator-selector domain payloads use the separate bounded codecs
above; retired SearchSpace generator tags are not accepted.
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

## CachedDataset Format V2 And Fixture Set V1

Completed CachedDataset local state uses exact JSON `cache-manifest.json` format
`dryml.cached-dataset`, version `2`, plus an ordered `chunks/` inventory. The
manifest stores mode, selected built-in codec, encoded NumPy-backed SpecTree,
total yield count, and closed codec descriptors. It stores no source value,
absolute path, work token, attempt identity, iterator, or live backend object.
Restore validates format, fields, spec, names, counts, descriptor shape, and
declared file presence before installing readiness. Store deferred-file manifests
authenticate chunk bytes; iteration then performs codec-level validation before
delivering any yield from a chunk.

NumPy chunks are non-pickle NPZ component arrays with explicit logical dtype,
shape, and bounded-segment descriptors. Parquet chunks use typed value streams,
offsets/shapes, paired complex components, UTF-8 bytes, and explicit bfloat bits;
they require `pyarrow>=25.0.1`. NetCDF chunks require `netCDF4>=1.7.4`, use
`NETCDF4` fixed typed component streams and explicit offsets/shapes, disable
mask/scale authority, and serialize native library access within the process.
All codecs reconstruct only supported dense logical trees as NumPy leaves. There
is no object/VLEN reconstruction authority, silent coercion, codec fallback, or
public extension grammar.

Working snapshots use the same format/version with mode `working`, expected and
sealed counts, descriptors, exact owner/operation/attempt evidence, and an opaque
Store-local work token. They deliberately refer to external retained work and are
not independently iterable. Finished snapshots use mode `completed`, include no
work identity, and own every declared chunk through normal snapshot-local state.

`tests/fixtures/cached_dataset/v1/` is fixture-package version 1, not an older
cache manifest version. Its complete Store-v3 snapshots contain finished NumPy,
Parquet, and NetCDF format-v2 payloads, including empty, nested dynamic/dtype, and
polynomial cases. The manifest hashes every authority file and records PyArrow
25.0.1/netCDF4 1.7.4 writer qualification. Compatibility tests read these bytes
directly with supported readers and never regenerate them in the assertion path;
no progress snapshot, staging dependency, work token, or machine-local path is
part of the fixture commitment.

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
