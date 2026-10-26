# Release Notes

## 0.3.0b2 (unreleased)

Legacy Repo/Store API v3 and compound-reference capabilities are removed, with
no compatibility aliases or refusal stubs. `ObjectRef`, `StateRef`,
`StateSelectorRef`, `ObjectSelector`, old Store record classes, reference JSON
codecs, claim/declaration/fork/alias methods, and snapshot-directory access are
absent. Use Node-local identities and full-StateKey selections. `RepoDefinition`
now uses version 2 without claim settings; version 1 is rejected. Store format 4,
Node identity 3, and SQLite schema/coverage 11/3 are retained. See
[Repo/Store API retirement](repo_store_v3_retirement.md) for the migration table
and long-lived-interpreter limitations.

Identity and occurrence queries now provide an explicit `.iter()` terminal for
single-use lazy consumption. First advancement captures required authority and
metadata; matching or relationship traversal then advances on demand without a
final result collection. Iterators retain no Store locks/readers across yields
and support `.close()` for early cleanup. Candidate inventory and intermediate
stages can still require eager work. See [Lazy Iteration](graph_querying.md#lazy-iteration).

Constructor signature errors now identify an unexpected keyword in the primary
`SignatureError` message and `slot`, including through `Definition(...).build()`.
The stable `"argument binding failed"` reason and original exception cause are
retained; argument values are not rendered. See [Signatures](signatures.md#errors).

Maintained dense CPU estimator adapters now provide XGBoost classification and
scalar regression (`dryml.models.xgb`) and statsmodels OLS/binary Logit
(`dryml.models.statsmodels`). Imports and configuration are passive; fitting and
fitted restore require their exact optional reader stacks. `dryml[xgboost]` pins
XGBoost 3.4.1/sklearn 1.9.0 and `dryml[statsmodels]` pins statsmodels 0.15.0;
both pin NumPy 2.1.3, SciPy 1.18.0 and pandas 3.0.5. Native use requires Python
3.12; core supports Python >=3.10. Linux CI covers Python 3.10–3.14;
Windows and macOS run `good-enough` on Python 3.12 only. Framework-specific
tests share one pip-installed Linux Python 3.12 heavy job, replacing separate
JAX and native-estimator matrices. Notebook coverage runs inside Linux's
Python 3.12 `good-enough` job with its optional kernel dependencies installed.
Automatic test CI runs on pull requests only and cancels superseded runs for the
same PR; explicit manual suite dispatch remains available independently.
The shared heavy stack uses XGBoost 3.4.1 from PyPI; fitted envelopes from the
previous 3.4.2 candidate remain incompatible under the exact reader-version gate.
The combined framework gate also covers explicit NNX candidate-variable copying,
so prediction and training do not mutate the live Flax owner before acceptance.
Extras have no Python markers that silently omit a requested native package.

Both adapters use finite authored dense NumPy Datasets, pure Method/Map metadata
and candidate-first one-shot acceptance: one successful fit step, one epoch and
actual example exposure, with no fabricated loss/iteration telemetry or
intermediate safe points. Retained `xgboost1` UBJSON and `statsmodels1` complete
Results pickle/metadata bytes preserve exact unchanged checkpoint replay. Full
statsmodels Results persist training arrays on disk and occupy memory alongside
their encoded bytes. Legacy XGBoost compute state and incompatible reader versions
fail explicitly without migration or Store rewrites. Existing sklearn, Data,
Method, ExperimentData, TrainState and Store formats are unchanged by these adapters.

Bounded Linux Python 3.12 evidence covers native held-out quality/parity, state,
association-sensitive recovery and real independent same-host Core workers.
Four historical isolated conda-forge Ubuntu/Windows Python 3.12 CI cells passed at
`fa01d177` on 2026-10-15 for statsmodels-only and CPU XGBoost-plus-sklearn: 194
assigned cases with zero native skips, exact-version/extra/dependency/CPU-build
guards and matching installed-wheel worker evidence. Both platforms used Python
3.12.15 with the former XGBoost 3.4.2 stack; this is not qualification of the new
3.4.1 mixed pip stack. Windows Conda workers explicitly receive their selected runtime's native
DLL directory; Execute's minimal-environment policy is unchanged. Other reader
pairs and GPU remain unqualified. Fast backend-neutral
contracts remain routine; native matrices remain targeted heavy qualification.
This rollout does not automatically close issue #38. See [XGBoost](xgboost.md),
[statsmodels](statsmodels.md) and [Testing](testing.md#native-estimator-qualification).

Query `refresh(True)` now creates a lazy process-local **one-shot request**, shared
by ordinary derivation, shallow copies, closure, occurrences, and owner/target
projections. Each compatible persistent destination completes it once. Later
terminals still capture fresh authority and proofs, using demand-driven auto
recovery rather than reusing query evidence. Another explicit `refresh(True)`
creates a new request. Migrate repeated per-terminal maintenance to
`refresh("always")`; it repeats reconciliation, not the old forced rebuild.
Correct ready SQLite sidecars validate without replacement. Use
`store.rebuild_query_index()` for unconditional replacement. Direct backend
True/always calls remain one independent cycle per call.

Algebra preserves operand-local lineage without Cartesian request/target widening.
New external scopes and Repo attachments introduce pending target obligations;
empty filtered results retain their obligations. Explicit outer True replaces
inherited intent with one new request over the expression, while False/auto/always
suppress it on that expression. Fixed IdentitySet/OccurrenceSet refinements detach
from live requests. Construction, ordinary explanations, and identity/occurrence
`take(0)` remain inert. Auto now recovers only indexes consumed by the selected
execution path; authority-only broad, exact-positive, and external metadata queries
leave unused dirty sidecars untouched. False performs no derived maintenance.

Same-request concurrent terminals single-flight targets and observe the exact
joined attempt's outcome. Later-created requests receive no credit from older
attempts; remaining obligations coalesce into at most one new cohort per target.
Failure/interruption is retryable in a later terminal, while successful targets
survive partial failure and later evaluation failure. Owner exceptions remain
original; joined failure/interruption raises actionable `QueryIndexError`.
Waiter interruption does not cancel the owner, self-reentry raises `QueryIndexBusy`,
and published outcomes remain immutable. Request and SQLite synchronization reset
by PID before child acquisition. Inherited Repo topology fails before its parent
lock; open a fresh child Repo. No query transport or deepcopy contract is added.

Explicit SQLite reconciliation compares a fenced authoritative inventory with
graph-exact encoded stored roots and checks active destination continuity. Repair
completes only after staged validation, activation, and captured-token
acknowledgement; normal `ReconcileReport` return or `validated=True` is not success
evidence. Foreign persisted ownership is refused before scanning and after claim
acquisition. Equivalent configured destinations coalesce only after all visible
presentations agree. One destination per Store invalidation domain is supported;
terminal-local checks reject visible conflicts, but independently maintained
multiple destinations across clients remain unsupported.

Pending explicit work requires scan/indexed-only admission even for ready
validation or joining another caller. Maintenance accounting retains started work
through failures and retries, charges owners rather than waiters, and reports
actual actions as `rebuild > validate > waited > reused > none`, separately from
declared policy. Empty captured dirty-token sets skip cleanup-only authority reads
without weakening readiness, staged validation, publication fencing, or preservation
of later/unrelated tokens. Store formats and SQLite schema are unchanged. See
[Query V3 Migration](query_v3_migration.md#refresh-lifecycle-migration).

Stored-CDef indexed prefixes now inspect current main authority before every answer,
including empty pages. A closure-only definition made main after a ready index cut
can no longer disappear: permitted complete inventory fallback returns it, while
indexed-only/no-scan execution refuses unsupported completeness. Missing or malformed
main targets remain visible authority errors.

Readiness/recovery occurs outside retained readers, followed by sorted unique Store
fences and non-repairing SQLite views retained through main inspection, paging, and
direct verification. In-cut readiness and actual sidecar identity/generation are
checked before results. Failure releases all readers before fallback/recovery and
discards attempt proofs without refunding cumulative work. Rebuild cache barriers
protect active same-process read views. Ready direct-root prefixes retain canonical
ordering, replica evidence, and zero broad definition enumeration. Store formats,
SQLite schema, and main dirty-token behavior are unchanged; this adds no global
federated snapshot or broader V3 acceleration.

Query V3 terminal controls now apply through external stored/alias/metadata
scopes, algebra, closure, occurrence targets, and owner/target projections.
Projections inherit controls while remaining existential. Explicit outer
overrides govern nested execution; incompatible algebra policies still fail at
composition. Disabled refresh preserves missing/dirty sidecars; explicit refresh
does not authorize forbidden recovery scans. Warnings precede required scans.

Indexed `take()` now enforces the previously omitted verification-budget check.
Separately, `max_verify` accounting is tightened to cumulative graph-exact identity
admissions across all terminal stages and retries: ordinary/fixed execution
reserves the entire detached domain before filtering, while exact/paged checks
reserve before verification, including rejected candidates. Replicas and repeated
stages share unchanged-cut charges; collisions remain distinct. Cut replacement,
unchanged metadata-demand recapture, instability retry, and selective/paged fallback
readmit candidates without refunding abandoned work. Unverified ordering lookahead
counts fetch/decode IO only; verified collision lookahead is charged. Metadata and
recovery IO remain scan work, not identity admissions. Existing explanation fields
report admission, scan, and page work separately. `take(0)` acquires/verifies
nothing; witness and depth budgets remain separate. No Store format changes.

Selected Method inference now uses converted consumer inputs consistently across
Pipe/Project/AutoEncoder successors, graph ports, Map metadata, Fold roles, and
native training inputs. Raw-result validation stays independent of inference
and candidate ranking. Map resolves metadata during pure constructor selection;
known selection failures can now surface at construction, while backend-only
unknown facts remain cursor-local and shared specs remain unchanged. Metadata
reads and example counting perform no repeated selection or compilation.

Dense CPU handoffs validate and retain one host representation per leaf instead
of extracting twice, then construct independent target storage. Backend adapters
activate lazily through the existing runtime boundary, including previously
unregistered optional endpoints. Native element adaptation retains its crossing,
native x/y setup consumes prepared Dataset output facts before source acquisition,
and experimental JAX zero-work admission remains ahead of native setup.

Fold now uses positive root-output EOF rather than cursor closure to authorize
exhausted checkpoints. Failed traversal cannot become successful truncated
recovery; consumer errors remain primary over cleanup failures. New progress is
version 2 with explicit EOF evidence. Compatible legacy active progress resumes;
valid legacy exhausted progress loads as ambiguous and requires an explicit fresh
`ManagedConfig(rerun=True)` before carry/Method/source effects. Saving unchanged
legacy state preserves its provenance. Genuine EOF cleanup/finalization/publication
failures retain trusted source-free retry and never report completion on failure.
Existing snapshots and completed legacy Values remain readable and immutable;
historically completed truncated results cannot be automatically detected/repaired.
Downgraded readers cannot resume v2 progress; selecting an earlier compatible
snapshot may replay work and is not automatic rollback or repair. Frozen synthetic
compatibility fixtures qualify both versions and the old-reader rejection gate.

`ObjectRef.object_projection()` and `StateRef.object_projection()` now return a
recursive query-only `ObjectSelector`. Each encountered exact reference becomes a
nested selector, while explicitly embedded raw ObjectRef and StateRef values remain
exact. Query V3 and fixed `IdentitySet` results expose `object_projection()` to
deduplicate checkpoints into fixed `ObjectSelectorSet` run identities without
merging separately realized ObjectId graphs.

Symbolic construction now uses `Definition` as its only runtime value. `Par`
expressions, discovery, static `sub(...)`, remapping, and loose selector
projection live on Definition; symbolic resolution means only that no active
expression remains, not that a constructor is complete or valid. `Mat[Definition]`,
`Ref[Definition]`, and bare `Template` are distinct receiving roles. Template is
the annotation-only shorthand for `Template[Definition]`, accepts inert symbolic
quotation data, and is not a constructible runtime recipe value. `QuotedDef`
remains the explicit local-data escape hatch.

`Distribution` is prohibited in Definition structure. Bind fixed values with
`Definition.sub(...)`, then use `Generator(definition, distributions)` with one
exact active-root mapping for sampling, grids, and exact-support selection.
`GeneratorSelector` replaces the beta selector name while retaining compatible
`dryml-template` v1 `template-selector` bytes for built-in distributions. A
Generator and arbitrary providers are nonportable.

Experiment artifacts now accept only `Mapping[str, Template] | None`; entries are
independently quoted and evaluated in canonical string-key order. Runtime
Template values, TemplateBundle, TemplateGenerator, and TemplateSelector are
intentionally removed without aliases, source/pickle compatibility, or persisted
value migration. Previously persisted symbolic `Ref[Definition]` values likewise
have no migration or replay guarantee.

Symbolic errors are `ParameterizationError`, `UnresolvedDefinitionError`,
`ParameterizationLimitError`, and `UnsupportedGeneratorVerificationError`.
`CannotConcretizeParameterizedDefinition` remains the TypeError for attempting
to concretize an active expression.

Core Execute now preserves symbolic `Definition` values through worker transport
instead of degrading their public Mapping view to an ordinary dictionary. This
allows Mat arguments to materialize in workers and keeps `Ref[Definition]` data
symbolic. Dispatch's incomplete-coverage warning now names its bounded diagnostic
categories and directs callers to `dispatch.explain(...)` for the full report.

Native training input is now owned entirely by the Dataset graph. Maintained
TensorFlow, Torch, and experimental JAX trainers consume canonical authored
`(inputs, targets)` batches, retain inspectable `TrainingPreparation` specs and
handoffs, and reject the retired trainer-owned batching, shuffle, count, and path
controls. The ML workflow qualification matrix now includes experimental JAX in
all W1-W3 CPU placements (36 primary cells), binds JAX/Flax/Optax environment
evidence and a fixed JAX seed, and adds credential-free managed local and Core
subprocess exact-restoration representatives. The qualification baseline and
fixture manifest are now schema version 2; existing version 1 authority remains
untouched and must be replaced by preparation into new empty roots. GPU
qualification remains the existing TensorFlow/W1 and Torch/W3 pair.

Core now owns the versioned `dryml-reference-json` codec for portable exact
`ObjectRef` and `StateRef` authority. Experiment history writes
`dryml-experiment-data` v3 and delegates embedded references to that codec, so
native model and Dataset definitions retain source, factory, tensor-specification,
canonical dtype, cardinality, and frozen-container identity without a model-owned
serializer. The reader continues to accept closed v1 and v2 histories and
migrates them in place on their next immutable publication.

Keras, Torch, and experimental JAX trainers now provide saved, recoverable
completed-epoch early stopping. The shared monitor/patience/mode/minimum-
improvement contract retains its best metric, wait count, accepted shortened
target, and, when restoration is enabled, best Model snapshot across pending-
postlude recovery. Optional best-weight restoration changes only Model parameters
and mutable buffers;
optimizer progression and JAX RNG remain at the stopping epoch. The resulting
mixed-point Experiment is published as a new exact terminal StateRef. Keras uses
this dedicated DRYML behavior adapter with managed safe points and supported
validation, without claiming recovery for arbitrary native callbacks or deferred
`validation_freq` scheduling.

## 0.3.0b1 (2026-09-29)

`CachedDataset`, `CacheCodec`, and `CacheIntegrityError` are now concrete public
Artifact APIs. Finite dense Dataset trees can be streamed into built-in NumPy,
optional Parquet (`pyarrow>=25.0.1`), or optional NetCDF (`netCDF4>=1.7.4`) cache
state and restored as ordinary NumPy-backed Datasets. Completed same-codec content
can be republished without its source; codec changes recompute. Metadata is
validated on restore and chunk integrity before delivery. Checked-in fixture-set
v1 freezes finished format-v2 reader evidence for all three codecs and a
deterministic NumPy/Parquet polynomial sample.

CachedDataset progress uses managed checkpoints plus opaque Store-owned work
tokens, fresh-cursor positional resume, and direct-DirStore retained files; final
StateRefs are self-contained. Fold now checkpoints bounded carry and the same
yield-position model. Dataset cursors are independent and closeable, Take is exact
and reports requested/actual exhaustion, and explicit Method iteration
independence allows eligible Map prefixes to skip transformations conservatively.
These are same-host local-Store contracts, not stochastic replay, cache discovery,
remote Store transport, a codec plugin API, or automatic cross-process
synchronization.

Managed declarations can opt into exact completed-StateRef returns and one
definition-transported optional Store publication control. Authored bodies still
declare `-> None`; unexpected body values warn and are discarded, and failures
never return apparent completion. `ml_dtypes>=0.6.0` is a base dependency; the
`parquet`, `netcdf`, and test extras carry the qualified optional readers.

`dryml.dispatch` adds an explicit configuration-first facade for bounded static
environment/world declaration discovery and one selected execution route. Process
defaults and immutable views can select an Execute backend or `InProcess()`;
there is no implicit route, backend fallback, resource growth, worker thread,
environment creation, package installation, Ray deployment, retry, resume, or
lost-response reconciliation. `submit` returns the existing core future only for
an Execute route, while in-process `run` admits fresh held-generation evidence
then directly calls the original synchronous callable once. Coroutine, generator,
and async-generator roots reject before probing; lazy values returned from a
synchronous root are data and are not driven by Dispatch.

Discovery uses shared generic static dependency, environment, and world kernels
for inline and isolated probes. Unresolved-only static coverage remains visible
in `explain` without warning on routine accepted calls; traversal limits still
warn, and explicit probe policy can warn for or reject all incomplete coverage.
Notebook-owned asyncio tasks, futures, and loops reject at the transport boundary
rather than crossing worker serialization, and interruption of pending blocking
dispatch no longer attempts premature cleanup. Conflicts, malformed results,
timeout/crash, cleanup failure, target drift, and hard admission failures stop
work. Reports and
probe envelopes are bounded, redacted, ephemeral observations rather than Store
records or admission tickets. Existing `EnvironmentSpec` selection now flows
through generic/core Execute and Dispatch as an exact point-in-time pin; an
unavailable or mismatched selected environment never falls back. Historical
source/loaded-code drift and post-admission external environment mutation remain
accepted limitations, not hot-reload or environment-locking guarantees.

`dryml.execute` now provides generic trusted-callable execution with explicit
`Executor`/one-off backend configuration, common `ExecutionFuture` and bounded
output APIs, local subprocess execution, and an optional existing same-host Ray
2.56.0 backend. Calls use coordinator-owned bounded spools and pre-GO admission;
they do not use `Repo`, `Store`, managed operations, runtime/session transport,
or an implicit backend. Existing Conda/venv targets and Ray deployments are
caller supplied. Ray is never provisioned or stopped by DRYML, borrowed drivers
survive close, and logical Ray resources do not claim physical isolation or
cross-coordinator accounting. The focused Linux Python 3.12 CI fixture prepares
its own ephemeral test server and environments; remote CI evidence remains
pending until the workflow passes on the pushed revision.

`dryml.core.execute` now supplies the separate core-aware facade over generic
Execute. It freezes call options and one shared `DirStore`/`RepoDefinition`
snapshot at submission; current support requires existing directly shared
`DirStore` authority and rejects archive, absent, ambiguous, and replacement
storage. It transports whole callable/capture graphs through worker setup, runs
one owner-specific function/Method/managed boundary, publishes returned live
Object graphs through ordinary Repo saving, and returns exact references plus
bounded Store-table-relative evidence. Inputs are not auto-saved. Opt-in argument
refresh coalesces maximal roots, can reuse a returned descendant StateRef, and is
not transactional: failed recovery neither rolls back completed publications nor
replays work.

Core futures adapt one generic byte future once, retain separate cleanup evidence,
and close only their reconstructed recovery Repo with `flush=False`; they never
close caller handles. The adapter uses version-local worker/coordinator codecs,
not cross-version RPC. Worker runtime activation precedes Store and payload work;
subprocess grants can be exact while Ray grants are logical only. Each Ray worker
uses one attempt and one runtime boundary. This remains an execution boundary, not
automatic Store/environment/cluster provisioning, distributed transfer, physical
isolation, or a transaction/rollback guarantee.

Stage 4 adds `dryml.requirements`, the dependency-light shared contract for
explicit hard requirements. Its public API is `RequirementSource`,
`RequirementDeclaration`, `RequirementIssue`, `RequirementReport`,
`RequirementResult`, `RequirementError`, `RequirementCombinationError`,
`RequirementBarrierError`, `RequirementCombiner`, `combine_requirements`,
`AdmissionReport`, and `require_admission`. Combination has only empty success,
valued success, and conflict failure outcomes; malformed input and invalid
protocols fail before partial authority is exposed. Explicit admission uses the
policy-independent `admission_ok`, not a domain report's policy-dependent `ok`,
and does not run work or mutate session/runtime state.

`dryml.environments.req(...)` and `dryml.worlds.req(...)` add passive hard
declaration APIs for supported live targets, with `requirements_for(...)` and
`requirements_for_method(...)` for independent domain resolution. The lazy root
aliases `dryml.env` and `dryml.world` are the exact plural owner modules. World
declaration omission is unconstrained, and `roles=` remains an exclusive
complete-role grammar. Requirements are not defaults, candidate selection,
runtime/session state, dispatch, inference, or automatic enforcement.
`EnvironmentRequirement.check(...)` now exposes `CompatibilityReport.admission_ok`
alongside its policy-dependent `ok`; `check_world_spec_satisfies_requirement(...)`
and `check_allocation_satisfies_requirement(...)` return the independently owned
`WorldCompatibilityReport.admission_ok` decision.

This is a clean environment-fragment drop. `RequirementFragment`,
`ENVIRONMENT_FRAGMENT_SCHEMA_VERSION`, `add_req`, `override_req`,
`fragments_for_class`, `compose_fragments`, `requirements_for_class`, and the
`dryml.environments.fragments` module are removed. There is no compatibility
alias, decoder, migration, or legacy-record path. Fresh environment records omit
only `payload.dryml.schema_versions["environment_fragment"]`; the reduced
payload changes their semantic IDs and record IDs, while persistent environment
and world value schemas otherwise remain unchanged.

## 0.3.0.dev1 (unreleased)

Stage 3 adds dependency-light generic code analysis through `dryml.code`. The published
surface covers closed target normalization, immutable program graphs, per-request
kernel DAGs and facts, local static `analyze()`/`probe()`, and bounded
current-thread `trace()`. Results are ephemeral in-process artifacts: consumers
own any persistence, domain facts, and interpretation.

This is not a transformation, registry, serialized-probe, worker, or process
transport API. `dryml.code` does not own cross-process isolation or invocation
policy; future transport and execution isolation remain owned by `dryml.execute`
and `dryml.dispatch`. Method-specific probing and transformations remain
deferred.

The root code-analysis namespace intentionally exports only the documented
principal APIs. Legacy source extraction, compiler hints, analyzer registries,
probe transport, domain fact, Method, Traits, and transformation compatibility
surfaces are absent.

## CDef V2 Completion

CDef V2 is a clean break. Only fully bound V2 CDefs, graph-aware definitions, ObjectRefs, StateRefs, and current Store records are accepted. Pre-port CDefs, missing identity versions, old Store layouts, historical query metadata, and mutable-state generations fail closed. There is no migration, converter, dual reader, or old-release recovery workflow.

Structural CDef identity is separate from graph topology and durable object lineage. `ObjectRef` preserves topology and ObjectIds; `StateRef` names immutable snapshots. Exact load uses `load_state_ref()` and an explicit reuse policy. Generic alias loading, revision selection, instance/build-missing/reuse-weak switches, and graph-save option objects are removed.

Directory checkpoints are the supported local persistence scope. DRYML supplies a payload directory and hashes an exhaustive manifest after codec hooks complete. It does not automatically detect application mutation; callers save after relevant mutation. Local filesystem publication and cooperating-process locking are supported; distributed filesystems and reference transport across unsupported execution backends are not.

Transfer, garbage collection, alternate representations, scalable incremental checkpointing, automatic mutation detection, and distributed Store coordination remain deferred.

## Passive Annotation Kernel

`dryml.annotations` is now a clean-break passive key/value carrier, direct
live-target attachment API, and deterministic static collector. Entries are
process-local, use identity semantics, and are neither serialized nor persisted.
Consumers own keys, values, interpretation, and any separately derived durable
state.

The retained surface is `Annotation`, `ANNOTATION_ATTR`, attachment and direct
lookup, class/method collection, and generic annotation errors. Requirement and
default decorators, environment/world/runtime facades, merge and resolution
APIs, source diagnostics, annotation envelopes and IDs, Definition helpers, and
the corresponding retired annotation modules have no compatibility exports.
