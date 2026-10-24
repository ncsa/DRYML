# Object State and Managed Lifecycle Verification

## Status and scope

This is the U21 integrated-cutover evidence record on `feat/refactor-object-state`,
starting at `32449c39`. **The integrated U01–U21 implementation is complete under
the recorded routine CPU qualification: the authoritative representative gate
passed, and all eighteen validated findings from the ten-lens review are resolved.**
Unit U01–U20 evidence predates this cutover. Current test locations below identify
the maintained proofs, not a claim that every listed test was rerun in U21 or that
their passing counts can be added across repeated runs. U21's command ledger and
qualification limits are distinguished below. Disposable detailed unit reports remain
under `/tmp/dryml`; this document retains the durable contracts and trace.

The governing model is trusted same-host users, ordinary failures and supported
local filesystem/locking/SQLite semantics. Execute is a lifecycle boundary, not a
sandbox. The accepted bounded **raw, unredacted worker traceback** policy remains;
`raw_traceback=False` is available for secret-bearing work. Historical loaded-code,
notebook and class-redefinition limits remain in the
[code-shift learning](solutions/architecture-patterns/2026-09-18-code-shift-in-long-lived-sessions.md).

## Delivered contract

- Definition authors Node; ConcreteNode is normalized construction and SavedNode
  is a typing refinement validated by `require_saved`. Handle equality is not
  graph equality. Freshly decoded graphs use `graph_equal` and the appropriate
  creation/object-projection comparison. No parallel legacy exact-load plan remains.
- State authority is Store 4 / identity 3, full `(creation-CDef digest, codec hash)`
  keys and CDef-owned buckets. Historical metadata, root membership, availability,
  retention and active use are distinct. Old Stores/snapshots are rejected intact,
  without conversion, deletion or authoritative schema replacement for index repair.
- Managed success/replay returns InvocationResult; `.value` freshly reopens and
  validates result authority. Operation is stable by object projection/member, run
  is stable across resume, caller invocation is fresh, producer remains historical.
  Recovery dispositions are explicit non-success outcomes, with no execution fallthrough.
- Current consumers use TrainState 2, Experiment wrapper 2, history 4, Fold progress
  3 and CachedDataset/work 3. Managed area/current/history/root-intent/pending are
  2/4/2/3/2. SQLite is schema 11 / coverage 3. Qualification fixture manifests now
  use version 3 with shared Node metadata; their fixed workload baseline remains 2.
- Core's ordinary callable codec remains independent of managed imports. The
  managed-owned Execute adapter is distinct from the closed durable-result codec.
  Both are version 2. Durable result metadata roots share one authenticated table
  and decode owner, preserving aliases across separately returned parent/subtree
  values. Result bytes remain limited to 8 MiB, while mutation journal frames have
  a separate 64 MiB limit with preflight before effects. Mutation intent version 2
  distinguishes visible classification from native persistence confirmation;
  neither SQL coverage nor an acknowledgement substitutes for native barriers.
  Optional frameworks remain lazy. No sandbox, hot reload, remote Store transport,
  backend provisioning or old-user-Store migration is implied.
- A live-start rerun retires the old final association but preserves its physical
  grant as explicit release-pending debt while deferred input bytes are needed for
  replacement publication. New final association or failed-execution cleanup
  retires the debt. Retirement failure is inspectable and subsequent reconciliation
  terminates without body/capture. This selects no implicit start/initial/latest.
- Store-backed cache cursors freshly acquire exact Store-domain/full-StateKey
  native readers and retain them through exhaustion, close or error. An isolated
  owner thread per cursor preserves thread-affine Core leases while permitting
  foreign-thread close and independent non-LIFO cursors. Deferred chunk capture
  also holds a fresh native reader; a remembered path proves no availability.

## Requirement and acceptance trace

Paths in this table are below `tests/`. Each range is inclusive. Every R1–R80 and
AE1–AE56 is assigned; overlapping rows deliberately identify cross-layer proofs.
Tests are current implementation evidence, with prior-unit execution provenance
where U21 has not repeated them. The final coordinator gate passed as recorded below.

| Acceptance | Requirements | Current proof locations and retained obligation |
|---|---|---|
| AE1–AE2 | R1–R4, R7–R8, R11–R14 | `core/test_immutable_definition_graph_sprint.py`, `test_local_state_compatibility.py`, `test_repo_load_build.py::test_partial_node_build_and_apply_have_distinct_unspecified_state_semantics`: metadata-only composition, independent donor state/identity and unspecified-node construction. |
| AE3–AE4 | R5–R6, R9–R10 | `core/test_immutable_definition_graph_sprint.py::test_node_bind_alias_constraints_replacement_and_extraction`, `test_node_key_contract.py::test_shared_table_views_and_edit_reuse_are_owned_and_bounded`: alias conflicts, immutable replacement, rebased metadata extraction without payload/construction/publication. |
| AE5–AE6 | R11–R12, R14–R16, R18, R36, R53 | `core/test_repo_save_plan.py::test_partial_reuse_captures_only_remaining_distinct_nodes`, `test_plain_save_recaptures_but_reuses_installed_private_data`, `test_repo_load_build.py`: one capture/distinct needed node, shared installed bytes, no capture for explicit reuse; partial apply leaves unspecified nodes alone. |
| AE7–AE8 | R13, R16–R21, R37 | `core/test_state_ref_restore_into.py`, `test_state_bucket_layout.py`, `test_node_key_contract.py::test_shared_codec_rejects_malformed_or_incompatible_inputs_intact`, `artifacts/test_cached_dataset_fixtures.py::test_v1_store_is_rejected_intact_before_source_or_codec_input`: original projection and retained pins; incompatible/missing state and formats fail intact. |
| AE9–AE10 | R22–R24, R32–R33, R45 | `managed/test_completion_result.py`, `test_u15_admission.py`, `test_result_codec.py`: normalized argument compatibility, durable detached values, historical replay with zero body/save/restore. |
| AE11–AE12 | R25–R28, R49 | `managed/test_u14_authority.py`, `test_u15_admission.py`, `test_u16_authoring.py`: nonblocking duplicate-operation refusal, fresh-owner resume, explicit immutable initial designation. |
| AE13–AE16 | R17, R27, R29–R31, R40–R42, R47, R54 | `managed/test_rerun.py`, `test_u15_admission.py`, `test_retention_cleanup.py`; U21 `artifacts/test_cached_dataset.py::test_live_rerun_keeps_borrowed_final_chunks_until_republication`: start precedence, unavailable unselected initial, recovery seeding without recapture, old-final/new-start grant ordering and live borrowed content. |
| AE17–AE18 | R34–R35, R37–R38, R46, R54 | `managed/test_u16_authoring.py`, `test_callbacks.py`, `test_interruption.py`, `test_lifecycle.py`: exact selected final without restore/save; ordinary callback error after association; failure/interruption cannot become completion. |
| AE19–AE20 | R32–R33, R36, R46–R47 | `managed/test_result_codec.py`, `test_completion_result.py::test_each_lazy_access_revalidates_source_even_after_decode`, `test_failure_windows.py`, U21 `managed/test_repo_routing.py::test_replica_or_index_failure_never_associates_partial_state`: closed results, metadata-only inspection, partial publication and exact recovery evidence. |
| AE21–AE22 | R33, R39–R45, R55 | `managed/test_retention_cleanup.py`, `core/test_state_retention.py`, `test_repo_history_authority.py::test_retention_cleanup_preserves_saved_and_independent_recipe_history_on_rebuild`: surviving root reasons, affected-key cleanup, retained completion/value after payload reclamation. |
| AE23–AE24 | R19, R48–R54, R58 | `core/test_execute_managed_integration.py::test_u19_two_worker_branches_preserve_shared_dataset_and_independent_models`, `test_u19_transport_rerun_start_and_resume_restore_once_under_owner`, `test_exact_load_metadata_evidence.py`: independent branches, one selected restore/handoff, grouped local validation. |
| AE25–AE28 | R2, R5, R7, R16, R22–R24, R26, R37–R38, R48–R49, R57 | `models/test_experiment_recovery.py`, `test_experiment_artifacts.py`, `test_train_state.py`, `core/test_object_projection.py`, `test_local_state_discovery.py::test_mixed_facets_are_correlated_to_the_same_occurrences`: domain training policy remains with consumers; mixed graph selection and exact quoted configuration preserve identity rules. |
| AE29–AE30 | R23–R25, R27, R29–R31, R42 | `managed/test_u15_admission.py::test_start_without_rerun_rejects_all_phases_before_realization`, `test_rerun.py::test_rerun_initial_seeds_recovery_and_resume_does_not_resave`: reject supplied start without rerun and never fabricate live-start recovery. |
| AE31–AE32 | R25, R34–R36, R38, R46 | `managed/test_callbacks.py`, `test_interruption.py::test_interrupt_immediately_before_or_after_completion_CAS`: callback association survives; independent control CAS decides accepted interruption versus recorded completion. |
| AE33–AE34 | R16–R17, R29–R31, R42, R46, R59–R60 | `core/test_state_ref_restore_into.py`, `test_live_reuse.py::test_later_greedy_failure_evicts_every_previously_mutated_candidate`, `managed/test_failure_windows.py`: pre-hook failure preserves usability, post-hook failure invalidates affected graphs, failed rerun preparation preserves old authority. |
| AE35–AE37 | R35–R36, R39–R47, R61–R63 | `core/test_state_retention.py`, `managed/test_retention_cleanup.py`, `test_failure_windows.py`: immediate retained/in-use, independent federated continuation, uncertainty protection, only physical garbage permits success with cleanup report. |
| AE38–AE40 | R1, R3–R7, R9–R10, R13, R64–R66, R69, R71, R73 | `core/test_immutable_definition_graph_sprint.py`, `test_definition_concretize.py`, `test_template_support.py`, `test_node_key_contract.py`, `test_local_state_discovery.py`: Node authoring/navigation, exact aliases, Generator correlation, fixed facets and incompatibility. |
| AE41–AE43 | R2, R5, R7, R9, R11–R13, R17, R43, R45, R67–R70, R73 | `core/test_state_bucket_layout.py`, `test_local_state_compatibility.py`, `test_local_state_discovery.py`: distinct creation buckets for equal bytes, origin-only metadata discovery, historical versus available states and occurrence-correlated matching. |
| AE44–AE47 | R20–R21, R47, R51–R52, R56, R58, R65, R68, R70–R74 | `core/test_node_key_contract.py`, `test_local_state_discovery.py`, `test_query_v3_index.py`, `test_query_sqlite_concurrency.py`, `test_exact_load_metadata_evidence.py`: pure reuse versus fresh authority, zero-source take(0), detached pages/changed continuation refusal, eviction/fresh-process semantics and bounded work. |
| AE48–AE50 | R8, R20, R51, R68, R70, R74–R78, R80 | `core/test_store_incremental_index.py`, `test_repo_save_plan.py::test_indexed_multinode_save_has_one_sql_commit_and_shared_export`, `test_identical_plain_save_captures_but_is_an_authoritative_noop`, `test_query_v3_index.py::test_header_and_names_cannot_promote_dependency_roots`: independent memberships, synchronous deltas, no-op/no-inventory and index policy. |
| AE51–AE52 | R46–R47, R63, R72, R76, R78–R80 | `core/test_store_authority_journal.py`, `test_store_incremental_index.py`, `managed/test_lifecycle.py::test_completion_own_projection_failure_withholds_success_and_repairs_without_effects`, `models/test_experiment_data_concurrency.py::test_current_name_index_failure_preserves_authority_without_reporting_success`, U21 `core/test_repo_history_authority.py`: intent/commit/SQL/ack faults preserve authority and refuse stale completeness; R79 is not physical-cleanup success. |
| AE53–AE54 | R11, R43–R45, R47, R67–R68, R73, R75–R80 | `core/test_store_authority_journal.py`, `test_store_incremental_index.py`, `test_repo_history_authority.py`: derived-only rebuild, no payload reads/root promotion, sibling variant publication, header-only crashes and creation conflicts. |
| AE55–AE56 | R4, R10–R12, R40–R47, R51–R52, R56, R61–R65, R67, R71–R75, R77 | `core/test_state_retention.py`, `test_state_bucket_layout.py`, `test_node_key_contract.py::test_small_workload_counts`, `test_local_state_discovery.py`: full-key/Store scoping, incarnation-safe removal, identical eager/deferred keys and explicit hard-join costs. |

### Integrated flows

| Flow | Executable current chain |
|---|---|
| F1 Compose/build | Node identity/binding + partial `test_repo_load_build` + full-key compatibility tests. |
| F2 Partial apply | `test_state_ref_restore_into` including donor key, retained construction, alias identity and pre/post-hook failure. |
| F3 Completed reuse | `test_completion_result` including fresh lazy sources, and actual reclaimed-receiver subprocess replay in `test_execute_managed_integration`. |
| F4 Resume | Experiment/Fold/cache recovery files, plus U21 real failed-read/repaired-worker Fold case. |
| F5 Rerun/start | `test_u15_admission`, `test_rerun`, live-cache grant-debt success/failure/reconciliation. |
| F6 Reused checkpoint/final | `test_u16_authoring`, `test_train_state`, partial capture and public checkpoint/final work budgets. |
| F7 Retain/reclaim | Real native leases and grants in `test_state_retention`, federated `test_retention_cleanup`, current history/routing tests. |
| F8 Shared-data branches | Actual two-worker U19 case; model/artifact consumer cases and Dispatch managed placements. Optional training cells remain unrun here. |
| F9 Author/generate/query | Definition/template/signature tests, independent exact-key oracle and Node mixed-facet query tests. |
| F10 Discover/reuse | `test_local_state_discovery`, bucket/compatibility tests and explicit foreign-compatible apply. |
| F11 Repeat bounded work | Eager/deferred/warm counters, selected-key addressing, grouped reads, no-op publication and selective-query counts below. |

## Retired assumptions and retained original obligations

Old successful Store/cache/history/TrainState conversion assertions now require
explicit incompatible-version refusal and source-byte preservation. The original
codec math and reader obligations remain in **current-format** fidelity/empty/
polynomial fixtures; NumPy/Parquet still require equal ordered shapes, dtypes and
quadratic values. Current history retains 81 examples, exact predecessor/artifact
references, null/int/bool scalars and missing-column distinction. Value v1 remains
compatible and byte-round-trips. Historical fixture bytes were not regenerated.

Fold receipt/reuse tests still prove fresh source isolation, ambiguous completion
refusal before candidate loading, failure recovery, every reuse policy's active
identity conflict, descendant-state validation and unavailable/corrupt selected
payload failure. Result receipts now use `.final_state_ref`; status uses
`.checkpoint`/`.final`; run IDs stay stable and invocation IDs change on resume.

Routing's old buffered managed publication-event implementation is retired, not
kept as a second authority model. Former commit/reopened-validation/cleanup/dirty-
seed permutations are unreachable because retained buffered publication refuses
before capture/commit. The supported buffered primitive still has its separate
Core `test_zipstore_identity.py` commit, stale writer, cancellation, durability,
reopen and cleanup proofs. Current managed tests retain routed root/child replicas,
first-match closure, changed-route recovery, index/replica/cancellation partial
failure, independent coverage, current reader overrides, corruption, shorthand
borrowing, live-resource result refusal and per-target cleanup. The old snapshot
JSON promotion-count assumptions map to current grouped-key/header/destination
proofs in `test_exact_load_metadata_evidence.py`, not fabricated legacy events.

The former declaration/ClaimRecord history model is not revived. Current history
tests use Node membership/name CAS, including real two-process competing writers,
safe immutable losing branches, no dependency-root promotion, interruption before/
after name publication and explicit enabled-index failure/repair. Consumer-level
history identity/idempotent append/CAS obligations remain in the three
`models/test_experiment_data*.py` owning files.

`tests/test_profiles.json` points to current names, including earlier U13–U17
renames; the documentation roster guard fails on stale names rather than silently
dropping proof. New cache lifetime/history/format regressions are routine entries.
The framework CI list includes native Fold carry and scalar-result boundaries;
the existing native-estimator list/pins remain intact. Curated CI lists do not
authorize a local broad backend run.

## Work and IO evidence

Counts below describe the stated fixtures/layers, not universal asymptotic or
latency guarantees. Prior unit counters were witnessed in their bounded runs;
the fresh U21 preparation observations are separated below.

| Layer | Witnessed counter / scope |
|---|---|
| Pure preparation | Eager/deferred keys agree; a repeated requested view has zero new discovery/fold/hash work. Chain/tree/direct alias have zero exact-fallback rows at the tested sizes. Unordered/cross-branch joins retain exact fallback/tie costs. |
| Shared export | Four-leaf save: 5 exported rows, 4 capture-discovery rows, one shared block, one SQL COMMIT, 4 manifest rows, 1 saved-root row and 2 name rows. |
| Same-CDef load | Duplicate key: 1 key read, 1 header interpretation, 2 destination compatibility validations and 2 restore hooks; destinations retain distinct ObjectIds. |
| Fresh evidence | Cold/warm/cleared exact preflight: 1 key/header/payload authentication each time and no construction/restore. Independent plans do 2 passes; no availability proof survives the operation. |
| Scaled selected closure | K=1,3,7: K selected key/header/payload passes; routed parent adds one. No scan of unrelated sibling variants. |
| Selective query | Origin page of ten at page size 3: one shared-block decode, zero payload authentication. Cache budget zero changes page-size-1 decode count to 11 including lookahead, not result semantics. |
| Replica frontier | Indexed take(1): one semantic admission, two sources, four candidate rows including ordering lookahead; no Store inventory. take(0) is source-free. |
| Publication | Alternating A→B→A routing: one SQL COMMIT per Store and one capture per needed live stateful node. Current authoritative no-op: zero epoch/index COMMIT churn; arbitrary live state still needs capture unless explicitly reused. |
| Managed | Explicit checkpoint + selected final: one capture, one ordinary callback, zero final recapture. Replay: zero body/capture/restore/callback/return-normalization/consumer-hook work. Saved-start association and retention do not serialize payloads. |
| Retention | Affected full-key/root work, not whole-Store liveness inventory. Protected deletion returns retained/in-use immediately. Equal codec hashes in different creation buckets remain independent. |

### Fresh U21 preparation measurements

These are size-3 `test_small_workload_counts` observations, repeated with visible
counter output at closeout. Each case checks eager/deferred equality, then asserts
that warm requests add no discovery, folds or hashes. Hash counts below are the
`hashes` counter, distinct from block/value hashes. Export bytes are serialized
shared metadata, not state payload bytes.

| Shape / mode | Discovery | Edges | Folds | Hashes | Exact fallback rows | Unordered ties | Export bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| Chain / recipe | 4 | 3 | 4 | 12 | 0 | 0 | 7,605 |
| Tree / recipe | 5 | 4 | 5 | 15 | 0 | 0 | 7,529 |
| Direct alias / recipe | 4 | 6 | 4 | 12 | 0 | 0 | 8,220 |
| Repeated diamond / recipe | 10 | 12 | 10 | 30 | 21 | 0 | 15,210 |
| Repeated crossjoin / recipe | 13 | 18 | 13 | 39 | 27 | 0 | 21,591 |
| Unordered / recipe | 10 | 15 | 10 | 30 | 21 | 14 | 15,822 |
| Direct alias / identity + state facets | 5 | 7 | 5 | 15 | 0 | 0 | 10,995 |
| Repeated crossjoin / quoted + symbolic | 14 | 19 | 14 | 42 | 27 | 0 | 24,725 |

| Shape / mode | Frontier bits | Frontier copies | Ancestor work | Block hashes | Value hashes | Partition rounds / rows |
|---|---:|---:|---:|---:|---:|---:|
| Chain / recipe | 12 | 3 | 3 | 30 | 107 | 0 / 0 |
| Tree / recipe | 16 | 4 | 2 | 40 | 153 | 0 / 0 |
| Direct alias / recipe | 12 | 3 | 3 | 36 | 128 | 0 / 0 |
| Repeated diamond / recipe | 99 | 12 | 9 | 72 | 248 | 0 / 0 |
| Repeated crossjoin / recipe | 147 | 15 | 9 | 102 | 377 | 0 / 0 |
| Unordered / recipe | 123 | 15 | 9 | 78 | 257 | 9 / 69 |
| Direct alias / identity + state facets | 17 | 4 | 3 | 49 | 198 | 0 / 0 |
| Repeated crossjoin / quoted + symbolic | 176 | 16 | 10 | 122 | 443 | 0 / 0 |

For these eight cases, `export_rows` and `view_rows` equal discovery;
`edge_rebindings`, `incidence_reuse` and `reused_rows` are zero. This does not
replace field-edit or cache-clear tests with a recipe-only shortcut. A complete
per-layer connection-open, fence-acquisition and OS stat/open census was not
collected by U21; the layer-specific assertions above are not such a census.

U20 observed 6,718,464 canonical tie combinations in an erroneous *logical*
comparison of control records. Correcting exact-source-byte CAS removed that use;
it did not make difficult Node graph canonicalization linear. Output-sensitive DAG
occurrence enumeration, repeated joins and unordered exact ties can remain costly.
Environment distribution introspection also dominated some prior long commands.
No host-wide speedup, universal linear bound, or deadline-as-passing policy is claimed.

### Coupled discovery-posting correction (#5/#6/#13)

The focused production regressions are in `core/test_discovery_postings.py`.
These measurements cover the SQLite page statement itself using a progress handler
at every VM instruction and `EXPLAIN QUERY PLAN`, rather than treating returned
candidate counts as rows examined. At the measured sizes, growing unrelated roots
by 32 leaves identity hit/miss work at 91/45 instructions; scalar hit/miss changes
from 118/45 to 119/45. Adding 16 unrelated saved histories leaves full-StateKey
hit/miss at 91/45 and a ten-match scalar page at 1,246. Nested mixed-facet hit,
correlated miss and absent identity measure 511→512, 481→481 and 115→115.
Plans seek token, child/parent and contribution indexes; temporary ordering and
distinct work remains proportional to the selected intermediate/output relations.
These are statement-level counts on the qualified fixtures, not end-to-end IO or
latency guarantees or a rarity-aware optimizer claim for arbitrary patterns.

Ready synchronous publication measurements include **all** NodePreparation owners,
not only the retained owner. Each recipe case below used four owners across export,
decode and reused derivation, with one exported unique table. Feature counts are
actual persisted local postings for the publication block; roots no longer embed
feature closures.

| Shape / size | Nodes / contributions | Total discovery / folds | Local features / links | Export rows / bytes | Exact fallback rows | Frontier bits |
|---|---:|---:|---:|---:|---:|---:|
| Chain / 8 | 9 / 9 | 18 / 18 | 1 / 8 | 9 / 13,062 | 0 | 180 |
| Chain / 16 | 17 / 17 | 34 / 34 | 1 / 16 | 17 / 23,524 | 0 | 680 |
| Tree / 8 leaves | 15 / 15 | 30 / 30 | 8 / 14 | 15 / 15,126 | 0 | 383 |
| Tree / 16 leaves | 31 / 31 | 62 / 62 | 16 / 30 | 31 / 29,446 | 0 | 1,591 |
| Direct alias / 8 | 9 / 9 | 18 / 18 | 1 / 16 | 9 / 14,702 | 0 | 180 |
| Direct alias / 16 | 17 / 17 | 34 / 34 | 1 / 32 | 17 / 26,811 | 0 | 680 |
| Diamond / 3 | 10 / 10 | 20 / 20 | 1 / 12 | 10 / 11,892 | 42 | 255 |
| Diamond / 6 | 19 / 19 | 38 / 38 | 1 / 24 | 19 / 21,186 | 138 | 978 |
| Crossjoin / 3 | 13 / 13 | 26 / 26 | 4 / 18 | 13 / 17,046 | 54 | 387 |
| Crossjoin / 6 | 25 / 25 | 50 / 50 | 7 / 36 | 25 / 31,498 | 180 | 1,494 |

Saved identity/state-bound fixtures separately measured 12 alias contributions /
6 features / 5 exported rows and 30 crossjoin contributions / 12 features / 11
exported rows. Their total discovery/folds were 24/26 and 36/50; crossjoin fallback
was 90 rows. The contributions are genuine per-root/domain output, not a promise
to eliminate that output cost. Bitset frontier width and exact hard-DAG fallback
still grow nonlinearly. Metadata memoization is bounded/evictable and proves no
payload availability, source membership, freshness or external admission.

Forced tied-set order reversal covers indexed and metadata-scan discovery with
exact own ObjectIds and full StateKeys, including fresh spawned readers. A separate
withdraw/republish case proves that membership-only deltas can reuse earlier block
facts across the reversal. Missing/older/dirty coverage and missing seek indexes
refuse complete negative/count answers and repair derived metadata without changing
Store-4 authority. Existing short-fence/continuation and incremental transaction/
no-op/no-inventory tests remain owning obligations. The disposable command/failure
ledger is `/tmp/dryml/review-fix-5-6-13-report.md`; the coordinator's global gate and
plan-completion determination are recorded separately below.

Final bounded owning verification for this correction selected 487 of 580 collected
items across the 19 explicit query/discovery/index/preparation/save/documentation
files recorded in that ledger: 487 passed, 93 representative-policy deselections,
one existing multiprocessing fork deprecation warning, 169.93 seconds. This is a
bounded subsystem result, not the global representative gate.

## Review disposition

The source review covered the complete feature diff from `f996ea43`, including
committed U01–U20 and integrated U21, through correctness, project standards,
testing, maintainability, security, performance, API contract, reliability,
adversarial and institutional-learning lenses. Initial routing failures were
recovered before findings were validated. A different-model reviewer could not
be attested; the local adversarial fallback supplies no cross-model independence
claim. Review was selective deep inspection, not exhaustive certification of
every changed line, external consumer or optional backend.

All eighteen validated actionable findings were resolved in serial followups.
The original review's **Not ready** verdict is retained as pre-fix evidence,
not represented as a second post-fix review. The table records the coordinator's
dispositions and maintained proofs; final integrated execution is a separate gate.

| Finding | Resolution | Maintained proof below `tests/` |
|---|---|---|
| #1 | Native reader lifetime protects cursor traversal and deferred copying; closed/replaced sources fail explicitly. | `artifacts/test_cached_dataset.py`: real second-process cleanup, exhaustion/close/error/GC, foreign-thread close, independent cursors and restored sources. |
| #2 | Recipe editors preserve bound facets and sharing; genuine edits invalidate construction rather than silently discarding selected state. | `core/test_repo_definition.py`: every retained editor and pre-hook incompatibility refusal. |
| #3 | Ordered single-view transport uses the common typed-position labeler; genuine unordered/multiview fallback remains. | `core/test_node_key_contract.py`: independent equal ordered siblings, canonical bytes and permutation counters. |
| #4 | Ordinary named views and LocalState validation reuse bounded handle-owned pure preparation. | `core/test_local_state_compatibility.py`: warm counters, clear/eviction, concurrent requests and owner lifetime. |
| #5/#6/#13 | Discovery stores occurrence-local facts once, seeks correlated indexed postings, enforces scan admission and uses authenticated portable locators. | `core/test_discovery_postings.py`: VM-step/plan counters, publication costs, tied-set reversal, fresh processes, missing/dirty coverage and repair. |
| #7 | Automatic live references use current SavedNode metadata without legacy receipt attributes. | `core/test_signature_repo_resolution.py`, `test_signature_consumers.py`, artifact Fold/cache and Execute selections. |
| #8 | Facet `values()` derives its owned occurrence map once. | `core/test_reference_values.py`: ownership-walk counters and metadata-only behavior. |
| #9 | Journal-specific frame budgets cover pending, classified, intermediate and acknowledged representations before authority effects. | `core/test_store_journal_budgets.py`: real 7 MiB completion/reopen/next mutation, recovery and over-budget no-effect refusal. |
| #10 | Durable result v2 shares metadata graph ownership across roots; reference/path/adapter versions reject old data intact. | `managed/test_result_codec.py`, `test_completion_result.py`, `test_control_codec.py`, `core/test_execute_managed_config.py`: alias recomposition, fresh kernel, malformed views and real Execute replay. |
| #11 | Restore uses receiver-rebased adopted state paths across unordered children. | `core/test_state_ref_restore_into.py`: forced differing tied-set order and process boundaries. |
| #12 | Realization overlays respect exact quotation scopes, including aliased owned/quoted occurrences. | `core/test_realization_scope.py`: quoted pins, alias preservation and exact no-op overlays. |
| #14 | Recovery reestablishes exact affected native file/directory barriers before acknowledgement or reuse. | `core/test_store_authority_journal.py`, `filesystem/test_filesystem.py`: repeated barrier failure/interruption, receipt preservation and subsequent recovery. |
| #15 | Quoted configuration is excluded from declaration collection; canonical owned projections persist and reopen. Resolved by #12 with dedicated public save proofs. | `core/test_repo_declarations.py`: indexed/unindexed quoted checkpoint saves and canonical projections. |
| #16 | Persisted exact mutation evidence remains repairable after projection acknowledgement; repair is separately journaled and fenced. | `core/test_repo_save_plan.py`, `test_store_authority_journal.py`: pre-variant failure, acknowledged partial work, later writers, ownership and idempotence. |
| #17 | Unreachable legacy publication bodies and their exclusively dead helpers are removed; explicit refusals retain supported readers/archive behavior. | `core/test_store_atomic_publication.py`, `test_zipstore_identity.py`, `test_backend_import_safety.py`. |
| #18 | Recipe-query examples construct with `repo.build`; complete SavedNodes use exact load. | `docs/test_object_state_documentation.py`: execute the actual stateful query example in a fresh process and distinguish initialized versus saved values. |

The testing lens also requested ongoing routine coverage for reclaimed-receiver
replay and recovery-only worker admission. Both are now explicit representatives
in `test_profiles.json`; this adds coverage without weakening existing obligations.
No actionable review finding is deferred. Accepted hard-DAG cost, code-shift,
diagnostic and optional-platform limitations remain distinct from resolved defects.

## Verification invocation and final gate

All U21 test invocations use the child root and this literal prefix:

```bash
source ~/conda.sh && conda activate big_env && TMPDIR=/tmp/dryml PYTHONDONTWRITEBYTECODE=1
```

Explicit paths precede options; common options are `--no-cov -o
tmp_path_retention_policy=failed -o cache_dir=/tmp/dryml/u21-pytest-cache`.
Only one intensive command runs at a time. The available U21 command, failure and
correction ledger is `/tmp/dryml/u21-report.md`. It distinguishes exact closeout
commands from earlier-session evidence retained without a complete raw command
transcript; missing command details are not reconstructed as executed facts.

### Worker terminal results

Selections overlap; these counts must not be added as unique coverage. Earlier
failed selections and their corrections are retained in the disposable report.

| Bounded selection | Terminal result | Scope / exclusions |
|---|---|---|
| Core load/preflight/reuse/save/receipt/deferred/metadata/signature/restore | 170 passed, 4 deselected; 72.75 s | `native_graph_restore` excluded; prior U21 terminal evidence. |
| Managed admission/rerun/authoring/callback/interruption/retention/result and four lifecycle faults | 86 passed; 483.31 s | Affected lifecycle semantics, not the integrated representative gate. |
| Current history and routing | 24 passed | Earlier separate documentation roster failure corrected and verified by closeout docs tests. |
| Work-budget representatives | 21 passed | Earlier U21 terminal evidence; repeated preparation cases below overlap. |
| Qualification contract and lightweight local/CPU/accelerated helpers | 55 passed, 56 deselected; 373.95 s | Explicit real/native/training exclusions; exact selection in report. |
| Fold reuse/execution, cache/frozen inputs, history/TrainState inputs, docs and managed import safety | 52 passed, 2 skipped; 478.62 s | Optional PyArrow fixture cases skipped. |
| Eight visible preparation counter cases and three synthetic device-evidence guards | 11 passed; 5.88 s | Synthetic observations do not qualify native CPU/GPU execution. |
| Scoped lint and whitespace | Passed | `/usr/sbin/flake8 --select=F,E9` on changed Python files; `git diff --check`. |

### Coordinator integration checks

These terminal selections overlap the unit evidence and each other. They are
reported independently, not summed as unique coverage. All use the activation
prefix above and failed-only temporary retention.

| Final followup selection | Result |
|---|---|
| Repair/authority/atomic publication/backend import safety | 110 passed; 2 runner-tier deselections; 23.80 s. |
| Durable result codec/public completion/control codec/managed Execute configuration | 136 passed; 117.50 s. |
| Cache cursors and real subprocess cache execution, routine representatives | 23 passed; 2 policy deselections; 244.59 s. |
| Executable docs/query/current exact load, routine representatives | 20 passed; 9.24 s. |
| Named preparation compatibility after scoped-lint cleanup | 36 passed; 1.94 s. |
| Authenticated NetCDF corruption fixtures and managed Execute configuration after followup | 65 passed; 200 explicit exhaustive-only cases deselected; 164.33 s. The fixture publishes a private malformed logical payload through ordinary capture, then proves native authentication succeeds before the real decoder rejects it without yielding. |
| Scoped occurrence queries and executable documentation after admission fix | 33 passed; 13 routine-policy deselections; 20.08 s. The owning eight-file query/discovery selection also passed 169 tests with 88 policy deselections. |
| Integrated query/reference/Repo/index authority plus managed and model consumers after ObjectId/soft-recipe followups | 1,196 passed; 19 skips; 575 tier/profile deselections; 1,628.81 s. Includes current native dataset-trainer representatives, not the separate heavyweight recovery or real-workload matrices. |
| Method subsystem with explicit CPU selection | 90 passed; 25 policy deselections; 27.32 s. Uses `CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu`, retaining all selected Torch/TF/JAX runtime crossings. |
| Metric factories and qualification consumers after handle-equality assertion cutover | 48 passed; 2 skips; 97 tier/profile deselections; 868.85 s. Symbolic/direct Fold recipes compare exact graphs rather than independent handles; no metric algorithm changed. |
| Routine policy guard and current documentation | 37 passed; 7.35 s. Required boundaries name current retention refusal, separate archive durability, incompatible-history refusal/current fidelity, reclaimed-receiver/recovery-only transport and result alias preservation. |
| Complete runner/tier/profile tooling and documentation | 72 passed; 48.30 s. The renamed documentation file and eight renamed or deliberately replaced node overrides retain their original smoke/medium tiers; current history fidelity has an additional explicit medium pin. |
| Changed tracked Python and four new Python test/helper files | `flake8 --select=F,E9` and `git diff --check` passed. Full-project style/typecheck qualification is not claimed. |

The final representative command is:

```bash
source ~/conda.sh && conda activate big_env && CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu TMPDIR=/tmp/dryml PYTHONDONTWRITEBYTECODE=1 ./tests.sh good-enough --ignore tests/old --ignore tests/dev -x tests -o tmp_path_retention_policy=failed -o cache_dir=/tmp/dryml/u21-coordinator-cache --basetemp=/tmp/dryml/refactor-u21-good-enough-15 --junitxml=/tmp/dryml/refactor-u21-good-enough-15.xml
```

**Passed.** Attempt 15 completed both disjoint runner phases without failures:

| Phase | Passed | Skipped | Deselected | Warnings | Duration |
|---|---:|---:|---:|---:|---:|
| Isolated process-sensitive representatives | 277 | 0 | 120 | 2 | 178.80 s |
| Maintained main representatives | 4,729 | 39 | 1,144 | 19 | 4,958.73 s |
| Combined gate | **5,006** | **39** | **1,264** | **21** | **5,137.53 s** |

The changed tier metadata retains four more main-phase representatives than the
earlier failing attempts. Warnings include intentional warn-mode admission,
fork/multithreaded-runtime diagnostics, malformed-numeric overflow probes and
upstream Keras deprecations. Optional skips and unrun matrices remain limitations,
not passes. Detailed output is `/tmp/dryml/refactor-u21-good-enough-15.log`;
the final main-phase XML is `/tmp/dryml/refactor-u21-good-enough-15.xml` (the runner
reuses its XML destination between phases, so the log retains both summaries).

Earlier attempts exposed stale consumers and recovery defects; focused and
bounded owning checks verified their corrections before the passing gate.

Attempt 8 passed its isolated phase (277 tests) and stopped in the main phase
(83 passed, 2 skipped) at a NetCDF decoder fixture that modified published bytes
without updating native inventory. The corrected fixture retains its exact
`length|offset` and no-first-yield assertions: it now mutates a private copied
payload, publishes through ordinary capture and authenticates the new full key
before testing the real decoder. Bounded codec owners passed (28 passed, 8
optional skips), with eight additional malformed NetCDF layouts passing; Parquet
layouts remain prerequisite-skipped. An intermediate coordinator invocation
omitted representative filtering and timed out after partial progress; it is not
counted as passing or exhaustive qualification. The corrected routine owning
selection passed all 65 retained tests as recorded above.

Attempt 9 passed 277 isolated tests and reached 1,674 main-phase passes (12 skips)
before a scoped occurrence query exposed a Node-admission routing error. Discovery
preflight now admits actual identity producers only, while occurrence metadata
presentations remain in ordinary scope/refresh handling. The original failure
passes unchanged; strengthened fixed-membership tests also observe fresh metadata
after mutation. Scan refusal and indexed coverage remain enforced. Focused,
eight-file owning, and coordinator integration selections passed before attempt 10.

Attempt 10 passed 277 isolated tests and reached 1,741 main-phase passes (12
skips), then exposed invalid authoring spelling in ObjectId selector lowering.
The lowering now uses the supported classless `Definition()` spelling; all four
original reference-query tests pass unchanged. A subsequent 34-file query owning
run collected related failures without fail-fast (716 passed, 12 skipped, one
failure): discovery was trying to resolve positional parameter names of inert
quoted symbolic recipes. Discovery-local field enumeration now keeps literal
positional edges and supplied keyword facts without resolving the class or
inventing named positional postings. Quotation remains a barrier; independent
soft roots and owned identity/full-key anchors survive a fresh-process metadata-
only rebuild. Focused and 400-test owning proofs passed, followed by the 1,196-test
integrated query/managed/model selection above, before attempt 11.

Attempt 11 passed 277 isolated tests and reached 4,387 main-phase passes (30 skips)
before a fresh TensorFlow CPU-conversion process failed during incidental CUDA
initialization (`CUDA_ERROR_UNKNOWN`). The exact unchanged test passed immediately
on retry (5.46 s). CPU validation now explicitly hides GPUs and selects JAX's CPU
platform; hiding GPUs alone left the installed JAX CUDA plugin trying to initialize
without a device (87 Method passes, three initialization failures). With both
documented environment settings, all 90 retained Method tests passed. No runtime,
test assertion, dependency or skip policy was changed for this host failure, and
the final gate is CPU qualification rather than a GPU claim.

Attempt 12 passed 277 isolated tests and reached 4,458 main-phase passes (30 skips)
before a metric-factory test compared independently authored Node handles with
`==`. Its two symbolic/direct recipe comparisons now use exact `graph_equal`,
preserving the original construction, topology, facet and sharing obligations.
The bounded Metric/qualification selection passed before attempt 13; mathematical
metric/training policy and runtime implementations were not changed.

Attempt 13 passed 277 isolated tests and reached 4,628 main-phase passes (39 skips)
before the routine-policy unit guard required retired Zip/history proof names.
The guard now pins current retained-publication refusal, separate buffered archive
fault/durability proofs, old-history refusal and current fidelity. It additionally
requires metadata-first worker replay/recovery and the result alias regression;
the existing selected proofs and policy are not removed or loosened. All 37
policy/documentation checks passed before attempt 14.

Attempt 14 passed 277 isolated tests and reached 4,632 main-phase passes (39 skips)
before the tier metadata still referenced the renamed documentation file. Bounded
runner-tool collection then identified eight stale per-node overrides as well
(71 passes, one metadata failure). All were mapped to their current proofs with
the same smoke/medium tiers, without removing metadata validation. The complete
runner/tier/profile/documentation selection passed 72 tests before attempt 15.
Exhaustive/full/coverage/broad heavy/medium/profile/old/dev gates were not run.

Optional/platform limits remain explicit: the six XGBoost recovery parameters
require 3.4.1 while this `big_env` has 3.4.2; the pin is not weakened. Six
statsmodels recovery parameters have prior partial evidence, not a final combined
qualification. Parquet fixtures require PyArrow 25.0.1 while this environment has
23.0.1. Seven selected JAX cases in three maintained files require unavailable
Flax/Optax. These optional prerequisite skips are not native qualification, and
no installation, pin weakening or codec fallback is attempted. Real W1/W2/W3
training/TFDS/network, Ray services,
GPU, full JAX model training/persistence, alternate Python/platform and native-estimator matrices are not
U21 worker qualification. Earlier installed Torch/TF carry proofs are prior
evidence, not rerun claims. Full lint has pre-existing debt; only scoped F/E9 is
used, and no configured typechecker gate is claimed.
