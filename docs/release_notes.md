# Release Notes

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
