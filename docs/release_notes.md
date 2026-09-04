# Release Notes

## 0.3.0.dev2 (unreleased)

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
