# Query Index Backend Contracts

Query indexes are derived acceleration for current Store authority. DefinitionRecords, explicit stored-root membership, declarations, complete v3 snapshots, aliases, lineage records, current annotation records, and verified snapshot-local directories remain authoritative. Closure-only DefinitionRecords hydrate graph nodes but do not become stored roots during rebuild. A backend must reject incompatible schema or semantic metadata before row decoding or ready activation, then rebuild only derived files from validated records.

Indexes preserve V2 CDef graph topology, typed graph paths, ObjectRef IDs, StateRef hashes, reference occurrence paths, current object/state mappings, immutable snapshot fields, and lineage facts. Schema 9 also stores graph-exact V3 identity and direct-relationship candidate rows per root, plus a graph-complete `(root_graph_hash, root_blob)` stored-root projection and a mutation guard for keyset pages. The encoded-root tie key preserves structurally equal roots with different sharing topology and genuine digest collisions. Transactional count/XOR-of-row-hashes witnesses detect deleted or partial graph projection rows on explicit verification; incremental publication updates witnesses using only changed rows, without rescanning historical projections. The stored-root guard makes unmanaged mutation of either legacy or V3 root rows visible before a bounded page can claim completeness. The guarded `cdefs().stored().take(n)` plan may therefore satisfy `require_indexed()` only when every returned CDef has direct Store authority proof; candidates established solely by aliases, declarations, snapshots, or a main reference require complete authority fallback and are rejected by indexed-only or no-scan execution. A disabled refresh never initializes or rebuilds a missing, dirty, corrupt, or incompatible sidecar. Other witnesses prove sidecar integrity only, not full V3 authority coverage, so no broader V3 indexed-only capability is advertised. In particular, `DirStore.write_main_ref()` intentionally has no dirty token, so main membership is read under the authority fence when it matters. Indexes must not reduce exact references to definitions or reconstruct current mappings from captured snapshot annotations. Save publication appends advisory reference/metadata rows for its published StateRefs and replaces current mappings only for those targets, including stateless roots. It does not rescan historical snapshots on a clean incremental save. Rebuild replaces each projection from its proper Store authority. Candidate filtering outside the guarded stored-root plan may be approximate or incomplete, so authoritative Python verification always determines conflicts, predicate errors, and complete results.

Metadata dirty tokens identify an exact object or state target separately from definition tokens. DirStore publishes the applicable durable token under its writer fence before each definition, stored-root, declaration, or query-visible alias authority mutation; an interrupted authority write may therefore leave harmless dirty state, while an idempotent write leaves no new token. Incremental projection holds the Store authority fence through the SQLite write and clears only captured tokens for indexed targets. Unrelated or unscoped tokens require a full rebuild, and tokens published after the captured cut survive for recovery. A save whose derived index remains dirty reports a partial publication failure without discarding the complete snapshot. Legacy unscoped tokens remain valid rebuild requests.

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
reports the first required scan before it starts. Forced refresh grants no scan
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
inventory/completed recovery (`store_scan_count`), and page work independently;
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
Build-claim and replacement cache barriers close only unleased connections; they
never tear down another active same-process reader. Transactions remain pinned to
their original sidecar; later unleased use detects replacement identity. WAL readers
may cause an explicit bounded checkpoint/rebuild failure until they release. Direct
unleased connection users still coordinate teardown themselves.

Read views also belong to their entry process. A forked child must open a fresh
view; inherited view operations raise `QueryIndexError` before touching the
parent's SQLite handle. Child closure or natural normal/exceptional unwind of an
inherited context invalidates only its Python view: it performs no temporary-table
cleanup or rollback on the inherited handle and removes no parent lease key from
child bookkeeping. Fresh child views use independent handles and leases. The
parent view retains its transaction and temporary relations until its own ordinary
cleanup, including strict owning-process lease removal.

SQLite connections are process-local. Rebuild and mutation use staged activation so a reader sees a prior complete index or a new complete index. Dirty, absent, corrupt, or incompatible metadata projections trigger authoritative fallback or visible rebuild; dirty state is cleared only for the authority generation that was indexed. Index failure never rewrites snapshot/current authority or silently makes stale rows current. Rebuild notices are visible because scans can be long. Rebuild claim descriptors use `dryml.locking` but retain SQLite's claim creation, stale-eligibility, wait, and cleanup policy. Dirty-token and complete main-file activation use `dryml.filesystem` directly after SQLite closes/checkpoints the applicable WAL and handles; the index does not depend on Store-private native helpers. A persistence failure can follow visible sidecar activation, so recovery reconciles status rather than assuming rollback. Supported behavior assumes local SQLite, filesystem rename/persistence, and locking semantics; distributed filesystems are outside this contract.

`dryml.core.query.sqlite.open_connection(config, *, readonly=False)` is the public standalone connection owner. `config` must be a `SQLiteQueryIndexConfig`, `readonly` must be exactly `bool`, and `config.path` must be present; validation occurs before importing SQLite or creating a parent/database. Invalid argument types raise `TypeError`, while an absent configured path raises actionable `QueryIndexError`. The connection applies the same foreign-key, timeout, trusted-schema, journal, and durability configuration as the query-index connection manager and always closes its connection. Writable opening creates missing parent directories; read-only opening requires an existing database and creates neither the target nor its parent. SQLite open/configuration errors remain SQLite errors, while a missing optional backend is reported as `QueryIndexUnavailable`.

The outer `open_connection` context is a handle scope, not a commit scope. Its connection uses standard-library deferred transactions and creator-thread affinity; callers use an inner `with con:` block to commit on successful exit or roll back on exceptional exit. When the outer scope ends, any transaction still open is rolled back rather than committed, then the handle is closed on normal exit, exceptions, and cancellation. Manager-cached index connections retain autocommit mode because index read views and mutations issue explicit `BEGIN`, `BEGIN IMMEDIATE`, `COMMIT`, and `ROLLBACK` statements. Manager cache keys remain process/thread/access-mode local and handles are never returned across keys. Cached handles disable SQLite creator-thread enforcement only so manager teardown can close an inactive worker's handles from a coordinator thread; callers must not race handle use with teardown, and a failed close remains visible and retained for recovery.

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
