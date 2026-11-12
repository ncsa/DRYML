# Query Index Backend Contracts

## Format-4 Store Projection (Schema 12, Coverage Contract 4)

Every supported query-visible built-in DirStore mutation owns its derived work:
persist intent and authority, classify the committed Store cut, apply the coalesced
projection delta **and** coverage in one explicit SQLite transaction, reread its
committed evidence, then acknowledge the exact captured mutations. Filesystem and
SQLite publication are separate boundaries. A complete public mutation returns
only after both finish. The Store-owned prepared-batch seam applies the same
protocol once per logical Store commit; Repo/managed owners supply prepared
facts through that seam, rather than maintaining a second index themselves.

Cooperating writers may advance authority while another worker prepares a detached
projection. Maintenance retries that stale cut up to eight times, yielding between
attempts outside Store writer fences (100ms exponential backoff, at most 12.7s of
waiting). Each attempt recaptures and validates current metadata and acknowledges
only the exact resulting receipts. It does not rerun the already-published
authoritative mutation, state capture, callbacks or workload. Genuine IO/schema
failures and cancellation are not retried as stale cuts; exhausted contention
still returns explicit failure with pending publication/recovery evidence.

`auto` enables usable SQLite. Only optional-backend unavailability admits its
declared nonpersistent fallback. Explicit `sqlite` never falls back; detected
unavailability, foreign/future ownership, and conflicting configuration fail before
authority effects. `none`/`memory` have no persistent-maintenance obligation.
`mutation_index_disposition` reports policy, not readiness. Each authority domain
durably binds one normalized persistent destination/configuration in derived
`.dryml/query-index-destination.record`, under its writer fence. Other enabled
destinations/settings are refused across handles and processes. Removing this
binding is not an API for maintaining independent sidecars over one domain.

The closed relation families are `record`, `recipe_root`, `saved_root`,
`declaration`, `creation`, `manifest`, `availability`, `node`, `edge`, `facet`,
`name`, `requested_metadata`, `retention`, `lifecycle`, `discovery`,
`discovery_feature`, `discovery_link`, `record_bytes`, and `query_graph`. Compact root membership
joins to its authenticated block/table/overlay slot. Node, typed edge, and facet
postings retain those correlations and typed value paths; they do not flatten
different occurrences into one match. Manifests and availability retain the full
creation-CDef/codec-content key. Creation rows also retain a domain-separated
compatibility prefilter, computed through one shared preparation for the block's
creation views; it is a candidate key, not an equality or payload-use proof.
Creation headers and known graph blocks never
imply independent root membership or available data. Withdrawal replaces only its
source contributions and preserves immutable history.

`retention` is implemented as authoritative per-full-key grant postings with
owner UUID, root/intent token and graph reference. Owner affected-resource ledgers,
pending heads and exact removal-incarnation records also feed the closed `record`
relation through the same journal. They do not imply managed associations. Rebuild
uses those records without trusting reverse counts or reading private bytes.

`requested_metadata` is supplied by closed `dryml-node-metadata/v1` authority:
current object/state annotations, first saved-graph capture, and lineage evidence.
SaveAnnotations, current-mapping replacement/removal, and immutable captures use
the same journal/projection/acknowledgement boundary. Rebuild never reconstructs
captured annotations from current mappings.
`lifecycle` uses the managed owner's registered, versioned control reader and
relation projector. It covers current, history, root-intent, preparation,
result-association and result-content contributions through the same journal.
Core never imports managed to discover policy. A missing/incompatible required
owner feed refuses visibly rather than certifying an empty relation. Unknown
families and prepared paths are refused. Owner adapters must supply closed readers,
transitions, projections and publication guards together. Managed lifecycle
association policy remains separate from Core physical protection. Older schemas
or coverage-1/2/3 certificates require visible metadata-only derived recovery; they
cannot claim coverage 4 or authorize stale negative/count results. Store 4 bytes
and identity/Node encoding versions are unchanged by this derived-schema revision.

Coverage support is local to the certified Store cut, not ambient imports. An
absent owner namespace contributes no records and needs no adapter for general
Core readiness, positive queries, or complete negative/count answers. A sidecar
built by a Core-only process remains ready in a managed-loaded process and vice
versa, without refresh, inventory, or generation changes. Presence of managed
authority requires its complete owner reader for readiness and maintenance;
missing/incomplete readers fail closed even if SQLite already contains lifecycle
rows. An unsupported (`supported=0`) family cannot advertise readiness by matching
an unavailable process-local reader. Explicit `projection_rows("lifecycle")` still requires the registered owner,
including when the namespace is absent. Core never imports that owner itself.

Readiness requires matching Store UUID/epoch, no pending authority work, index
UUID/schema/coverage version, the exact closed capability set, and clean mutation
guards. Projection row inserts/updates/deletes dirty their family automatically.
Transactional count/XOR witnesses are updated only from changed source rows;
`validate(thorough=True)` independently checks those witnesses. A missing family,
guard, certificate, or incompatible cut cannot advertise readiness. Schema 9 and
incomplete coverage require recovery. Valid foreign owners and future schema,
semantic, or coverage versions remain intact and are refused.

`index.projection_rows(family, refresh=False)` is the current **index-level** read
seam. It returns detached `(source_path, correlation_key, canonical_json_bytes)`
rows under a short certified Store/SQLite cut. False performs no initialization,
replay or rebuild and refuses missing/stale/pending/incomplete coverage. True
permits maintenance first. The ordinary Repo/Store consumer separately uses bounded
`discovery` and manifest pages under the same certificate. Discovery contributions
retain source membership, root/block identity, a table/overlay locator, result
domain and a shared local-fact key. They never persist process-local tied-set
ordinals as occurrence identities. The authenticated decoder resolves the locator;
root reachability, named projection, declarations and final digest are checked
outside the short read cut. Local identity, full-StateKey and scalar tokens are
written once per prepared occurrence view in `discovery_feature`, and direct typed
edges once in `discovery_link`, under the block source. Root contributions do not
embed a feature closure or an encoded subtree. Withdrawing membership removes its
contributions without promoting block-local history to independent roots.
Local-fact keys use authenticated table/overlay locators and named derived views,
not preparation traversal slots. A later membership-only delta therefore joins
existing block facts even after tied-set traversal order changes.
Discovery never resolves an authored recipe to name positional fields. Unresolved
soft calls retain their literal positional edges and explicit keyword facts;
unknown positional names supply no named scalar posting. Quotation remains an
opaque edge boundary even though its Nodes share the authenticated block table.
An independently published soft recipe still contributes its own root and real
unquoted children; quotation alone does not create that membership or containment.

Qualified Node predicates seek token indexes, correlate all local constraints to
one occurrence and propagate indexed child/parent joins before seeking contribution
rows. The adapter forces the contribution seek: an ordering-index scan with a
posting filter is not a selective plan. Ordered local edge paths and link roles
constrain joins; unordered assignment, aliases, quotation, class rules and remaining
facets retain the exact Node residual. Supported anchors are exact ObjectIds, full
StateKeys and supplied local `str`/`int`/`bool`/`None` fields. Boolean conjunctions
can use a safe anchored conjunct; disjunction requires an anchor on every branch.
Subclass-only, absence/negation-only, unsupported scalar/support-only shapes without a
safe anchor visibly use scan policy, including refusal by `require_indexed()` or
`scan_policy("forbid")`, **before** explicit refresh or recovery can inventory.
Enumerating an explicitly requested complete domain without a selector still uses
its certified ordered relation. Exact-origin manifests seek their bucket source
range; compatibility-only discovery currently admits a derived-family scan.

Otherwise unanchored concrete recipe predicates in object-projection/saved domains use
the existing occurrence-correlated `facet` class row as a necessary filter when
the exact class is an `ImportRef`. Its block source and table/slot locator must
match the discovery occurrence. This avoids decoding unrelated classes, including
for quoted-input recipes without scalar anchors. Immediate quoted-root ObjectId
pins also join the existing `edge` row at the codec's ordered typed value path to
that quoted occurrence's `facet` identity row in the same block/table. Unordered
positions are excluded. This is necessary metadata filtering, not traversal into
quoted discovery children or a substitute for exact quoted graph equality.
Soft/subclass selectors and creation/local-state domains are excluded. These
filters retain residual matching and scan admission; they do not themselves
qualify as selective indexed seeks or change schema/coverage versions.

The optional `store_facet_class` expression index adds an exact importable-class
seek over existing class facets. New indexes include it; ready current-schema
sidecars can install it transactionally after scan admission under automatic
refresh. Installation runs outside Store fences, changes no projection rows,
coverage or authority generation, and is never performed by `refresh(False)`.
Without it, class filtering remains scan-controlled. An installed exact-class
seek may anchor an otherwise unanchored pattern; subclass matching cannot.
Schema/coverage compatibility does not require this optional physical access path.

Eligible exact, facet-independent nested target patterns recursively join existing
typed discovery links to a necessary owner relation. Association-derived views
remain conservative candidates and acceptance still uses retained relationship
traversal. Complete unbounded root capture evaluates this relation once, stores
only source/row-key addresses in a disposable SQLite table independent of the
sidecar, and hydrates bounded pages in block order. There is no retained source
reader or unmanaged sidecar handle between pages. Canonical rooted evidence, not
digest equality, merges duplicate identity contributions. Final output returns to
normal identity ordering after complete validation of the retained cut.

Eligible object-only pages additionally carry an optional identity-facet hint from
the existing occurrence-correlated `facet` row. Consumers use it only as an address:
the ObjectId must occur in the current source membership, the current declaration
must name that contribution, its exact canonical projection must match the derived
digest and selector, and every identified dependency must have consistent required
declarations across all participating Stores. The projection is authoritative;
unrelated containing snapshot graphs need not be reconstructed for that object
view. Missing/unidentified hints fall back to full discovery. Alias and scoped
metadata paths retain their source-root validation. This is a read-plan change,
with no coverage or persistent-schema version change. Page-local expansion reuse
does not retain authority proofs across terminal cuts.

The object-projection terminal may fuse a default-domain, unbounded ordinary
Repo/Store selection into that object domain only for proved state-independent
plain Node chains. The bounded metadata proof rejects state pins, branching or
repeated Node slots, quoted/matcher/callback/symbolic adapters and other execution
restrictions. Source evidence and fixed membership remain equivalent to explicit
collect-then-project; explicit domains, state-sensitive selection, bounds and
composed sources retain that original ordering. This is terminal planning, not a
change to the source domain or saved-Node matching contract.

All discovery families, guards and seek indexes participate in readiness, including
empty/count queries. Exact residuals and direct authority validate acceptance.
Legacy Query V3 read views and
coverage claims remain explicitly unavailable for format-4 authority.

`maintain_store()`/enabled mutation completion replay classified pending records
without inventory when the indexed epoch and pending journal form a continuous
cut. Preparation and shared block validation happen outside writer-blocking
fences. The short writer scope revalidates byte/cut witnesses before SQL and exact
acknowledgement. Clean incremental work reads changed paths and their direct
metadata dependencies; it never inventories unrelated Store growth. A current
authoritative no-op changes no epoch, dirty token, generation or SQL row. A
previous SQL commit awaiting acknowledgement is recognized by its exact receipt
and does not repeat the delta transaction.

Before that replay/reuse, Store-owned recovery reestablishes any unresolved native
publication barriers for the exact classified authority. Classification records
visible bytes, independently of the mutation intent's `persistence_confirmed`
boolean. A committed SQL receipt cannot set that confirmation. Repeated native
barrier failure preserves pending evidence and raises `StoreMutationError` with
the mutation/cut receipt, including during automatic query refresh. Successful
barrier repair performs no capture, restore, body execution or payload reads and
does not execute missing transitions. Coverage still describes the actual partial
cut rather than the intended complete publication.

Missing/stale/owned incompatible or corrupt sidecars use a visibly reported,
explicit metadata-only inventory and staged replacement. The staged certificate,
row witnesses and SQLite integrity are validated before activation; the Store
cut is revalidated before replacement. Rebuild reads no `data/` bytes, runs no
application capture/restore, and changes no product records. A ready explicit
refresh uses its current certificate without a historical scan; `rebuild()` is
the separate unconditional inventory/replacement operation. Legacy unscoped
dirty tokens request rebuild and only captured tokens are removed.

A rebuild waiter that observes another maintainer's ready replacement rechecks
current authority before proceeding. If a sibling publishes between that ready
observation and the recheck, the changed cut raises `StoreCutChanged`; ordinary
Store publication uses its bounded preparation retry. This does not acknowledge
the sibling's pending work or report the earlier replacement as still owned.
Unchanged-cut contention retains the explicit `QueryIndexBusy` failure.

After ambiguous SQL commit, independently reread the exact mutation receipt and
coverage before acknowledgement. Pending work blocks ordinary indexed reads even
when SQL is already committed. A post-authority index/ack failure raises
`StoreMutationError` with `confirmed_refs`, `mutation_id`, the actual `cut`,
`index_disposition="repair_pending"`, and `recovery_attempt`; cancellation carries
the same evidence. Repair uses persisted metadata, not another application
capture. A stale detached repair fails rather than overwrite/acknowledge a later
writer. Public `acknowledge_mutation` requires the exact SQLite receipt when
persistent indexing is enabled. This never promises a filesystem-plus-SQL rollback.

Connections and read leases are process/thread-local and also asyncio-task-local
when called inside a task. Explicit transactions roll back on cancellation.
Staged replacement preserves leased readers and uses the existing checkpoint,
handle-retirement and native activation policy; contention fails actionably.
ZipStore retains its extracted nonpersistent/buffered policy. Its journal cut
becomes durable only after successful archive commit; no persistent SQLite claim
is made for an uncommitted extraction.

### Certified scalar-count execution

Coverage 4 additionally certifies immutable semantic interpretation. Each
`record_bytes` row records SHA-256 and byte length for the actual framed source
bytes observed during detached preparation. A `query_graph` row belongs to its
immutable block source and carries primitive Node descriptions, named parameter
values, stable block locators, named-view keys, full normalized equality evidence,
typed direct/association edges and operation-independent guard partitions.
State selections are validated before a supported graph is certified. A graph
that cannot supply this proof advertises residual-only execution.

Graph envelopes use version 1, bounded zlib-compressed/base64 canonical JSON, and
an explicit uncompressed byte count. Decompression is bounded at 64 MiB; evidence
that exceeds the producer budget is not certified for compact execution. Primitive
records do not contain live objects, callbacks, payloads or process-local tokens.
Graph-local guard integers encode the equivalence partition, not token identities.
Full rooted equality evidence distinguishes semantic digest collisions and sharing.
This derived equality evidence is compressed together with the shared graph; it
does not expand discovery contribution rows or replicate posting feature closures.

The ordinary unbounded identity and nested count planners share a closed exact program for
ordered Node/factory patterns and supported pure matchers, retaining explicit
residual disposition for other patterns. Local nested scalar/factory-target and
sequence-shape tokens narrow targets before source contributions are requested.
For covered programs, the scalar consumer authenticates current memberships/root
headers, exact immutable block bytes and declarations in every participating
Store, then interprets primitive records and counts typed paths without Node
decoding. State availability and payload-use admission remain separate operations.
Graph interpretation is byte-bounded and terminal-local; declaration checks are
shared across the terminal but backed by retained fresh byte witnesses. All
captured files and cuts are revalidated before the scalar answer escapes.

Covered state-facet pins compare the entire encoded key and the candidate creation
slot's full normalized equality witness with a separately prepared pattern witness.
They do not infer creation equality from the key's digest or merge creation aliases
with the surrounding match assignment. Ordered quotation matching uses strength-2
assignments: raw argument/keyword spelling, soft/concrete kind, and missing facets
remain exact, and an alias first proved weakly is rechecked when encountered in a
quotation. Quoted Par records are literal data; they are neither evaluated nor
loosened. Unordered quoted assignments and selector quotations retain residual
execution. These capabilities use existing coverage-4 records without migration.

Direct `IdentityQuery.count()` lowers its sel/kind restrictions to contribution
candidates and reads those rows once in bounded fetches before block grouping.
It uses the same proof reader and exact primitive matcher as nested counts, but
counts matching identities instead of paths. Its key retains the contribution's
explicit domain kind, semantic digest bucket, and full rooted equality evidence;
the natural kind of the decoded graph cannot replace that domain tag. Initial
Repo cache knowledge joins under the same complete identity key. A duplicate
contribution is authenticated before being merged, including a nonmatching
connected Store's relevant declarations. Unsupported restrictions/bounds keep
the ordinary consumer. Coverage and authoritative formats are unchanged.

Target contribution source addresses are buffered once; contribution records are
read in bounded source batches and compact graphs by block. This complete scalar
plan does not construct a fixed Node result set or expose an iterator continuation.
Explicit verification/depth/output bounds, local-state domains, source algebra,
metadata scopes and unsupported residuals use the existing page consumer instead.
Index replacement, authority changes and cancellation retain the same retry and
reader-cleanup rules as other unpublished terminals. Same-epoch source-byte damage
is an authority error, not a cache hit or success.

### Detached decoded pages

Covered unbounded `IdentityQuery.collect()` shares the scalar identity proof
reader and predicate compiler. It authenticates every contributing membership,
root, block byte witness and relevant connected-Store declaration, including
duplicate contributors, before merging complete kind/digest/exact certificates.
Only matching certificates are decoded from authoritative block bytes. Each
returned Node is compared with the full certified normalized identity, not just
its digest. Cached Repo matches and all contributing source evidence are retained.

Certified named-view/state compatibility proofs need not be recomputed after
decoding unchanged bytes; grammar, transport locators and complete result evidence
are still verified. A terminal-local 32 MiB byte-charged owner LRU groups matching
block consumers. Oversized owners remain transient rather than accumulating.
All file, authority-cut and SQLite-incarnation witnesses are revalidated after
decoding. Interruption returns no partial fixed set. No payloads, imports of
receiver classes, authoritative-format changes or new index families are needed.
Explicit bounds and unsupported/scoped/composed plans keep the ordinary consumer;
incremental iterators keep their existing page/output discipline.

The current consumer captures bounded keys and immutable record/block bytes under
short Store/SQLite cuts. It releases both before preparation, exact matching and
metadata expansion, revalidates that same cut before accepting the page, and
releases again before user output. One decoded preparation serves captured views
of an authenticated block; operation-local reuse compares exact bytes/views and
uses conservative `_content_facts` ownership sizing. Different advertised views of
the same exact block bytes reuse its decoder table and individually checked view
keys. Derived discovery tables share that owner's bounded fact LRU, reserving the
child key budget; clear/eviction repeats pure work, never external authority checks.
Oversized/nonresident blocks/views can repeat preparation. Exact hard cross-branch
joins, bitset frontiers, unordered canonicalization and output-dependent root/source
contributions retain their measured costs; no universal linear DAG bound is claimed.
Nested complete captures can allocate an entry up to their terminal's total
decoded-owner budget so one large current block can span several small pages.
Charges include the decoded table and its reserved fact LRU. Smaller declaration
projection preparations may be retained independently of unrelated snapshot
graphs. Declaration reuse cannot evict the current clustered source block; all
entries still share the same total byte limit. Neither cache supplies current
membership or validation witnesses.

Tokens retain Store UUID/epoch/pending evidence, index file identity/UUID,
schema/coverage/generation, producer and external-scope Repo topology, and keyset
ordering position. Before external output, physical cut changes discard all
dependent buffered rows, complete facts, combination proofs, page tokens and scan
projections. The terminal recaptures on up to eight attempts with bounded backoff
outside Store fences and SQLite readers. Spent admissions and other budgets,
initial Repo topology/cache membership and effective refresh/scan policies remain
in force. Eight invalidated attempts raise `QueryError`; genuine validation/IO
errors and cancellation are not retried as drift. After output begins, a changed
continuation fails explicitly without replaying the delivered prefix. Repo
topology changes still refuse rather than admitting newly attached sources.
Per-Store cuts are not a global transaction. Full-StateKey exact
lookup is an authority-only plan, including a negative lookup, and does not open
SQLite. Ready selective queries do not inventory unrelated Store contents.

All coverage families and publication guards are checked even for empty/count
queries. Applicable declaration anchors are read in every connected Store, not
only Stores contributing candidates. Pure cache hits are never freshness,
membership, availability or completeness evidence. `refresh(False)` disables
maintenance, not these checks. Query maintenance supplies `scan_admission` to the
backend: required rebuild inventory is admitted before it starts, including if
readiness changes after preflight. `maintain_store()` without that callback retains
its ordinary explicit maintenance policy.

## Historical Store-3 Design And Standalone CDef Indexes

The Store-3 reference, snapshot, alias, inventory and rebuild paths described
below are historical and have been removed, including their refusal stubs.
They do not govern Store-4 Node sources. The standalone manually populated CDef
catalog still supports graph registration, lowering, read views and validation;
it has no Store authority for rebuild or explicit reconciliation. Store-4
maintenance uses the projection contract above. See
[API retirement](repo_store_v3_retirement.md); old Stores are not migrated.

Query indexes are derived acceleration for current Store authority. DefinitionRecords, explicit stored-root membership, declarations, complete v3 snapshots, aliases, lineage records, current annotation records, and verified snapshot-local directories remain authoritative. Closure-only DefinitionRecords hydrate graph nodes but do not become stored roots during rebuild. A backend must reject incompatible schema or semantic metadata before row decoding or ready activation, then rebuild only derived files from validated records.

Indexes preserve V2 CDef graph topology, typed graph paths, ObjectRef IDs, StateRef hashes, reference occurrence paths, current object/state mappings, immutable snapshot fields, and lineage facts. Schema 9 also stores graph-exact V3 identity and direct-relationship candidate rows per root, plus a graph-complete `(root_graph_hash, root_blob)` stored-root projection and a mutation guard for keyset pages. The encoded-root tie key preserves structurally equal roots with different sharing topology and genuine digest collisions. Transactional count/XOR-of-row-hashes witnesses detect deleted or partial graph projection rows on explicit verification; incremental publication updates witnesses using only changed rows, without rescanning historical projections. The stored-root guard makes unmanaged mutation of either legacy or V3 root rows visible before a bounded page can claim completeness. The guarded `cdefs().stored().take(n)` plan may therefore satisfy `require_indexed()` only when every returned CDef has direct Store authority proof; candidates established solely by aliases, declarations, snapshots, or a main reference require complete authority fallback and are rejected by indexed-only or no-scan execution. A disabled refresh never initializes or rebuilds a missing, dirty, corrupt, or incompatible sidecar. Other witnesses prove sidecar integrity only, not full V3 authority coverage, so no broader V3 indexed-only capability is advertised. In particular, `DirStore.write_main_ref()` intentionally has no dirty token, so main membership is read under the authority fence when it matters. Indexes must not reduce exact references to definitions or reconstruct current mappings from captured snapshot annotations. Save publication appends advisory reference/metadata rows for its published StateRefs and replaces current mappings only for those targets, including stateless roots. It does not rescan historical snapshots on a clean incremental save. Rebuild replaces each projection from its proper Store authority. Candidate filtering outside the guarded stored-root plan may be approximate or incomplete, so authoritative Python verification always determines conflicts, predicate errors, and complete results.

Metadata dirty tokens identify an exact object or state target separately from definition tokens. DirStore publishes the applicable durable token under its writer fence before each definition, stored-root, declaration, or query-visible alias authority mutation; an interrupted authority write may therefore leave harmless dirty state, while an idempotent write leaves no new token. Incremental projection holds the Store authority fence through the SQLite write and clears only captured tokens for indexed targets. Unrelated or unscoped tokens require a full rebuild, and tokens published after the captured cut survive for recovery. A save whose derived index remains dirty reports a partial publication failure without discarding the complete snapshot. Legacy unscoped tokens remain valid rebuild requests.

Empty dirty-token captures skip cleanup-only closure-node eligibility and root-key
authority reads. Empty graph/root registration still checks readiness, but does
not enumerate tokens solely for acknowledgement. Nonempty registration and rebuild
retain all indexed-content writes, root-row metadata checks, staged validation,
and publication fencing; only captured covered tokens can be removed.

Broad Query V3 capture and SQLite rebuild authority capture consume the Store's
protected validated definition/root inventory seam. Its compatibility fallback
returns the existing complete record collections. DirStore validates marker paths
and targets against the definitions decoded in that same fenced capture, so each
authoritative DefinitionRecord decodes once during inventory acquisition; SQLite's
reference projection reuses those records. ZipStore uses the same seam only inside
its own open extracted transaction. Standalone marker readers and subsequent
standalone index validation retain their complete checks. Missing or malformed
markers, targets, main references, aliases, and conflicting authoritative ObjectIds
remain visible. Capture reads no state payload bytes and never writes authority.

CapturedStoreFacts and SourceCapture own detached identity inventory, independent
membership, metadata eligibility, and cut-local proof reuse. A complete proof key
uses sorted unique authority-fence domains and their fresh terminal-local capture
tokens; selective direct reads are not complete inventory. Metadata-demand growth
creates a replacement token even if authority is unchanged, invalidating dependent
proofs/results under the existing bounded retry policy. A SQLite generation is
derived-index state, not a Store authority generation or cross-terminal proof.

The terminal-owned SourceCapture also retains effective outer refresh, scan,
indexed-only, verification, witness, and depth controls for all nested consumers,
including external restriction and occurrence-target scopes. Authority capture
and explicit recovery reject prohibited scans before inventory IO; warning mode
reports the first required scan before it starts. Explicit refresh grants no scan
permission. Owner/target projection preserves controls and existential semantics.

Verification budgeting counts graph-exact detached identity admissions, not
sidecar rows, decoded blobs, authoritative reads, metadata holders, or replica
checks. Ordinary/fixed execution reserves its detached domain before restrictions;
exact and paged execution reserve before direct verification, including rejected
candidates. Unverified ordering lookahead is fetch/decode work, while verified
collision-bucket lookahead is admitted. Unchanged-cut stages and replicas share
charges; distinct collision identities do not. Replacement tokens, demand
recapture, instability retry, and selective/paged-to-inventory fallback discard
reusable admissions without refunding cumulative expenditure. The fallback's
new inventory domain is charged afresh. `take(0)` reads and admits nothing.
Existing QueryExplanation fields report admission (`verified_count`), actual
started inventory/recovery (`store_scan_count`), and page work independently;
this correction introduces no new stats schema or persisted format.

Bounded stored-CDef paging requires completeness as well as valid candidates.
First resolve index handles and policy-permitted recovery outside all retained Store
fences and SQLite views. Then acquire unique authority-fence domains in stable sorted
order and enter protected **non-repairing** transactions. Ordinary `read_view()`
entry is unsuitable inside these fences: readiness may initialize/rebuild or wait
for a build claim whose owner still needs Store capture.

Inside the retained cut, recheck dirty tokens, schema/build state, the stored-root
mutation guard, current main, and the transaction's actual UUID/generation/file
identity. Inspect main before even an empty frontier. A main-only closure definition
outside direct-root completeness requires complete inventory fallback or indexed-only/
no-scan refusal; malformed main data or a missing/malformed target remains an
authoritative failure. Retain views/fences through paging and direct verification.
On failure, release all of them before fallback/recovery, discard attempt results and
selective proofs, and retain SourceCapture's cumulative ledger and existing retry
limits. No main dirty-token expansion, schema change, or global snapshot is implied.
When a prior terminal stage already captured complete Store facts, bounded stages
reuse those facts rather than acquiring a newer selective cut or readmitting its
unchanged domain.

Read views lease existing manager-cached handles through transaction cleanup.
Build-claim barriers never tear down a leased reader. Replacement barriers retire
idle same-path handles across every thread in the current process, while preserving
leased readers and active transactions. Transactions remain pinned to
their original sidecar; later unleased use detects replacement identity. WAL readers
may cause an explicit bounded checkpoint/rebuild failure until they release. Direct
unleased connection users still coordinate teardown themselves.
Windows cannot replace a main file while its protected handles remain open;
replacement raises retryable `QueryIndexBusy` without closing those handles or
changing the canonical index. Close the retained reader, then retry the rebuild.
Store status validation and maintenance inspection preserve this typed contention
error; they do not relabel it as incompatible/corrupt evidence or use it to start
a metadata rebuild. Their read transaction and lease unwind before propagation.
Discarded connection managers best-effort close their owned current-process
handles. Unleased pools close in weak-reference finalization before SQLite's own
connection finalizers, including cyclic owner graphs.
Asyncio task completion also retires that task's cached handles and file-identity
entries, including after failure or cancellation. Active read leases defer this
cleanup until lease exit. Completion callbacks hold a weak manager reference and
run cleanup under its process-local lock; they never close inherited parent handles.
Task cleanup can therefore reduce live cached-handle counts before pool teardown.
Explicit close remains the deterministic way to observe cleanup errors;
collection in a forked child does not close an inherited parent pool.

Read views also belong to their entry process. A forked child must open a fresh
view; inherited view operations raise `QueryIndexError` before touching the
parent's SQLite handle. Child closure or natural normal/exceptional unwind of an
inherited context invalidates only its Python view: it performs no temporary-table
cleanup or rollback on the inherited handle and removes no parent lease key from
child bookkeeping. Fresh child views use independent handles and leases. The
parent view retains its transaction and temporary relations until its own ordinary
cleanup, including strict owning-process lease removal.
Manager construction and path-close barriers reset the class registry lock and
registry by PID before acquisition. Inherited managers reset their instance lock,
connection/file-identity bookkeeping, and leases before child access; fresh child
managers and handles are independently usable even when a parent thread held the
registry or instance lock at fork.

SQLite connections are process-local. Rebuild and mutation use staged activation so a reader sees a prior complete index or a new complete index. Dirty, absent, corrupt, or incompatible metadata projections trigger authoritative fallback or visible rebuild; dirty state is cleared only for the authority generation that was indexed. Index failure never rewrites snapshot/current authority or silently makes stale rows current. Rebuild notices are visible because scans can be long. Rebuild claim descriptors use `dryml.locking` but retain SQLite's claim creation, stale-eligibility, wait, and cleanup policy. Dirty-token and complete main-file activation use `dryml.filesystem` directly after SQLite closes/checkpoints the applicable WAL and handles; the index does not depend on Store-private native helpers. A persistence failure can follow visible sidecar activation, so recovery reconciles status rather than assuming rollback. Supported behavior assumes local SQLite, filesystem rename/persistence, and locking semantics; distributed filesystems are outside this contract.

`dryml.core.query.sqlite.open_connection(config, *, readonly=False)` is the public standalone connection owner. `config` must be a `SQLiteQueryIndexConfig`, `readonly` must be exactly `bool`, and `config.path` must be present; validation occurs before importing SQLite or creating a parent/database. Invalid argument types raise `TypeError`, while an absent configured path raises actionable `QueryIndexError`. The connection applies the same foreign-key, timeout, trusted-schema, journal, and durability configuration as the query-index connection manager and always closes its connection. Writable opening creates missing parent directories; read-only opening requires an existing database and creates neither the target nor its parent. SQLite open/configuration errors remain SQLite errors, while a missing optional backend is reported as `QueryIndexUnavailable`.

The outer `open_connection` context is a handle scope, not a commit scope. Its connection uses standard-library deferred transactions and creator-thread affinity; callers use an inner `with con:` block to commit on successful exit or roll back on exceptional exit. When the outer scope ends, any transaction still open is rolled back rather than committed, then the handle is closed on normal exit, exceptions, and cancellation. Manager-cached index connections retain autocommit mode because index read views and mutations issue explicit `BEGIN`, `BEGIN IMMEDIATE`, `COMMIT`, and `ROLLBACK` statements. Manager cache keys remain process/thread/task/access-mode local and handles are never returned across keys. Cached handles disable SQLite creator-thread enforcement only so manager teardown can close an inactive worker's handles from a coordinator thread; callers must not race handle use with teardown, and a failed close remains visible and retained for recovery.

## Explicit Backend Reconciliation

`StoreQueryIndex.refresh(policy, *, stats=None)` is a one-call backend seam.
Direct `True` and `"always"` calls independently perform an explicit cycle every
time. SQLite performs strict reconciliation, not unconditional replacement:
a correct ready index retains its UUID, generation, and file identity. `False`
does no work, while `"auto"` retains readiness recovery. Query owns any shared
request cadence; the backend stores no cross-call completion state or authority
proofs. Memory's explicit calls independently rehydrate its catalog, and Repo
federation propagates configured backend unavailability for both explicit policies.
`rebuild_query_index()` and direct SQLite `rebuild()` remain unconditional
replacement seams by default.

A ready explicit SQLite cycle retains one Store authority fence and a leased,
non-repairing transaction throughout its substantive checks. It captures one
validated authoritative root inventory, retains integrity/decodability, legacy
stored-root metadata/membership and internal V3 projection checks, and compares
the full `(root_graph_hash, encoded root_blob)` set against V3 stored roots.
This detects main-only additions, structurally equal roots with different sharing
topology, and digest collisions; neither ordinary CDef equality nor sidecar
witnesses establish authority agreement. Before releasing the cut it checks the
active destination's UUID/generation/readiness/owner and file continuity, including
changes behind a retained WAL snapshot. A successful check proves this maintenance
cut only, not permanent cleanliness or a reusable query inventory.

Recovery releases the fence and reader before claiming or waiting for replacement.
Owned replacement completes only after authority preflight, staged validation,
activation, and successful captured-token acknowledgement. A declined build claim
does not mean replacement succeeded: the caller substantively validates the actual
current candidate. Divergence under a live claimant raises retryable
`QueryIndexBusy` without completion credit or an unbounded repair loop. Authority
failure, backend unavailability, and interruption propagate through strict refresh;
an exception after visible activation does not imply rollback. Later dirty tokens
survive the captured-token acknowledgement.

`reconcile()` wraps the same strict cycle with a `ReconcileReport`. Ordinary errors
become error issues, while cancellation propagates. Historical validation issues
remain after successful repair. `validated` means validation work occurred, not
success, and normal report return is not completion evidence. `changed` reflects
this caller's actual visible activation, including activation followed by
acknowledgement failure. Action records the performed phase (validation or owned
rebuild), including a failed rebuild with no activation; refused maintenance does
neither. A non-owner's successful validation reports no replacement.
Report diagnostics expose started `store_scan_count` and actual `refresh_action`.
Optional `QueryStats` charges each started validation/rebuild inventory once, before
authority IO can fail; validation, staging, activation, acknowledgement failures and
candidate retries never refund expenditure. Maintenance scans are IO, not query
identity-verification admissions.

Valid persisted foreign Store ownership refuses maintenance before scanning and
again after replacement ownership is obtained. It is not an older compatible
metadata version. Known older owned metadata and owned corrupt sidecars retain
derived-only recovery; unsupported future metadata remains refused. One consistently
configured destination per Store dirty-token domain is supported. Independently
maintained destinations sharing consumable Store-wide dirty tokens are unsupported;
this backend seam introduces no global cross-client destination registry.

## Terminal Target Resolution And Demand Recovery

Query retains Repo Store topology through `SourceCapture.repo_stores()` before
maintenance. A single presentation traversal records every producer and scoped
obligation, including occurrence metadata scopes, without authority inventory.
It uses Store-owned lazy openers, never live Repo bulk refresh. The resolver
retains producer/scope associations and a compatibility descriptor independently
of request state; optional absence produces no target and no completion evidence.
Configured unavailability propagates for `True` and `"always"`. Query True owns a
process-local one-shot request shared by ordinary derivatives/shallow copies;
always performs a cycle per participating terminal. Operand-local bindings survive
algebra; newly attached scopes extend inherited requests, while explicit outer
overrides replace them. Fixed result refinements are detached. Requests retain only
synchronization, attempt outcomes, and compatible completed target identities, never
facts, captures, proofs, results, or readers. Compatibility is rechecked on reuse.

SQLite targets coalesce by normalized actual connection-manager destination only
after backend/configuration/path agreement and every visible presentation's
authority domain, source key, dirty-token domain, canonical version, and effective
settings are checked. Different destinations sharing one dirty-token domain or
incompatible presentations raise `QueryIndexIncompatible` before any maintenance,
even inside a Repo where source-key bindings previously hid later handles. Valid
persisted foreign ownership is inspected for every accepted target before starting
any cycle; the backend also rechecks ownership after replacement
claim acquisition. Other backends retain instance identity. A consistently chosen
custom SQLite destination is supported. Independently maintained destinations over
one invalidation domain across clients remain unsupported; the terminal-local guard
does not globally repair that configuration or introduce a cross-process destination
registry. Attachments after the retained cut participate next terminal.

`"auto"` performs no terminal-wide sweep or syntax-based special-case refresh.
Authority-only broad/exact/metadata paths leave unused sidecars untouched, even
when dirty. Actual bounded consumers resolve/preflight their indexes before any
retained fence or reader and recover only consumed non-ready targets. `False`
never maintains derived state; it uses ready views or authority fallback/refusal.
Pending explicit cycles require scan/indexed-only admission before claiming or
joining an attempt, including ready validation. Completed True uses auto demand
behavior; otherwise-supported indexed prefixes can reuse it without a new scan.
Warning mode warns once before the first inventory. Construction,
ordinary explanation, and identity/occurrence zero prefixes acquire nothing.

Each terminal retains cumulative admissions and started inventory work through
failure, fallback, and recapture. Backend scan counters merge in `finally`, including
interruption. Query `refresh_action` aggregates actual actions by precedence
`rebuild > validate > waited > reused > none`; demand recovery reports its phase,
and waiters receive no owner inventory charges.
Strict successful backend refresh is maintenance evidence; a report's normal
return or `validated` field is not. No authority proofs or readers cross terminals.
Repo owns the inherited-topology guard: it rejects a different PID before acquiring
its configuration lock. Query never resets another owner's locks or caches; a
fresh child Repo remains usable.

Request coordination freezes terminal obligations, acquires all applicable request
locks in deterministic order, and claims no pending pair while an older applicable
attempt flies. It releases locks before observing those exact outcomes, rechecks
after successful joins, and owns at most one new cohort per target. Only claimed
pairs receive success credit. Failure or owner BaseException notifies waiters and
leaves pairs retryable; partial target success survives later failure. Joined old
failure is never replaced by a later retry's success. Owner exceptions remain
original. Published attempt outcomes are immutable; later owner interruption does
not reverse completion, and unwind repeats notification if publication crossed an
interruption before waking waiters. Unpublished interrupted-owner outcomes become
actionable QueryIndexError for waiters. Same-thread
self-reentry raises QueryIndexBusy; waiter interruption detaches only that waiter.
Maintenance precedes separate fresh evaluation cuts and cannot lend authority proof.
Request locks/completion reset by PID before child acquisition, retaining derivative
sharing but no parent completion. Existing Store/import/SQLite lifecycle limits
remain; no new query transport or deepcopy contract is introduced.

## Reference-Aware Containment

Reference-aware nested queries do not add persisted rows, schema versions, or a
second authority format. Existing sidecars can remain in place and reopen with
their current compatibility rules. Missing, dirty, corrupt, stale, or
incompatible sidecars follow the existing visible rebuild or fallback policy;
they never replace current authoritative root capture or turn a required
containment scan into an empty result.

Eligible materialize-only CDef containment continues to use the indexed
definition/owner and captured materializing-edge paths. Reference-inclusive,
reference-bearing, and exact ObjectRef/StateRef containment instead captures
each selected Store's authoritative roots inside its read fence, then verifies
the retained graph in memory. The derived index can prioritize that work but
cannot exclude a root. `refresh=False` disables refresh, not this authority
check. Each source contributes one valid cut; federated queries may combine
separate source cuts, but no occurrence mixes two sources.

Producer selection or `in_source()` selects contribution scope before capture,
deduplication, cardinality, ordering, or the global raw occurrence limit. A
relationship residual reports its required authoritative root verification.
`scan_policy("allow")` permits the residual, `"warn"` reports it before the
first authority or recovery scan, and `"forbid"` and `require_indexed()` reject
it before recovery. The valid materialize-plus-reference-bearing combination is
deterministically empty and requires no scan.

The residual only inspects retained definition structure. Exact ObjectRef and
StateRef terminals retain their complete identities and declared hop kinds; they
are not dereferenced, loaded, or promoted to CDef children. Consequently an
occurrence is a definition-graph fact, not evidence of target ownership,
independent storage, payload availability, loadability, restoration, or cleanup
eligibility. ObjectId, alias, metadata, and reference restrictions remain in the
same V3 identity query contract.
