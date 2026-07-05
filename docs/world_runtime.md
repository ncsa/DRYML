# DRYML World/Runtime Split

This document summarizes the world/runtime foundation.

## Boundaries

`dryml.environments` describes software compatibility: Python versions, package requirements, capabilities, and DRYML protocol/schema support. Ordinary GPU, CPU, memory, device, and topology requests do not belong in environment requirements.

`dryml.worlds` describes resource and topology intent:

- `WorldRequirement` captures hard constraints such as role replicas, CPU counts, memory, accelerator counts, and basic topology flags.
- `WorldSpec` captures requested/default launch shape. It is not an allocation.
- `WorldAllocation` captures actual backend assignment and can produce process-local runtime allocation views.

`dryml.runtime` describes process-local activation:

- `RuntimeMode` is one of `orchestrator`, `probe`, `worker`, or `inline`.
- The default active runtime is `orchestrator + NoAllocation`.
- `RuntimeContextSpec` captures device visibility, framework bootstrap settings, process-local limits, and environment overrides.
- Device visibility and bootstrap plans can be built and applied before framework imports or object materialization.

World and runtime specs are canonical JSON sidecars written through `RecordStoreIO`. They are not DRYML Objects and do not change `ConcreteDefinition` identity.

## Runtime Setup Order

Workers and explicit inline execution should derive a `RuntimeAllocationView` from `WorldAllocation`, enter runtime mode, build/apply device visibility and bootstrap plans, then import frameworks or materialize objects. Framework-backed DRYML modules use `import_configured_framework(...)` so DRYML does not newly import heavy frameworks before runtime bootstrap. If user code already imported a framework, the helper reuses that loaded module instead of retroactively blocking normal object construction. Existing framework-object ingestion may read dtype/shape metadata without importing a framework; conversions that create framework-native objects require either active bootstrap or an already-imported framework. Bootstrap activation must match the active runtime mode/allocation and records process-local bootstrap state separate from the exported environment marker. By default, runtime bootstrap includes the `plain` adapter and framework adapters named in `RuntimeContextSpec.frameworks`; callers that need strict pre-import checks can pass an explicit `FrameworkBootstrapPolicy(..., strict_preimport=True)`. CPU affinity and hard memory limits require explicit process-control opt-in in reusable current-process activation scopes.

For explicit backend/power-user setup, `dryml.runtime.activate(...)` combines `enter_runtime(...)`, `build_runtime_bootstrap_plan(...)`, and `activate_runtime_bootstrap(...)` into one scoped barrier. It remains a runtime primitive; normal user-facing requirement/default sugar is expected to live in the planned annotation and dispatch layers.

The annotation layer now provides that declaration surface through `dryml.env.req(...)`, `dryml.world.req(...)`, `dryml.world.default(...)`, and `dryml.runtime.default(...)`. These decorators attach sidecar planning metadata only: they do not allocate resources, enter runtime, apply environment variables, import heavy frameworks, or change `ConcreteDefinition` identity. See `docs/annotations.md` for merge, override, conflict-report, and legacy `dryml.environments` compatibility details.

Orchestrator and probe processes default to hidden workload accelerators through the `none` device visibility policy. Worker processes default to `assigned`, exposing only assigned devices such as `CUDA_VISIBLE_DEVICES=0` and hiding unassigned CUDA, HIP/ROCR, and XLA devices.

Legacy `Compute.__compute_reqs__` dictionaries are still supported as a transitional bridge. They are checked against the active `RuntimeAllocationView`; a CPU-only allocation does not satisfy a legacy GPU requirement.

## Legacy Packages

`dryml.context` is retained as a legacy compatibility surface for older code. New code should use `dryml.worlds` and `dryml.runtime` directly.

`dryml.execute` is retained as legacy local pickled-callable execution pending dispatch v2. Its subprocess worker entry path enters `dryml.runtime` worker mode and applies runtime bootstrap before loading the callable or materializing objects. Inline execution stays in the caller's current runtime and rejects legacy resource requirements unless a future explicit inline-allocation path is added.

`dryml.dispatch.LocalSubprocessBackend` is the new spec/record/runtime-aware local path. Its worker enters `RuntimeMode.WORKER` with a real CPU-only `RuntimeAllocationView` by default, applies assigned device visibility, and only then imports target functions or materializes CDef arguments from shared `DirStore` refs. This preserves the runtime setup order required for later multi-worker orchestration.

## Local Multi-Worker Runtime

`dryml.dispatch.run_world(...)` and `Dispatcher.run_world(...)` allocate a requested `WorldSpec` into a stored `WorldAllocation` with one `ProcessAllocation` per role replica. Rank and local rank are deterministic: sorted role name, then replica index. Each worker receives `WorldAllocation.runtime_view(role, replica, world_allocation_id=...)`, so CPU-only workers still have a real `RuntimeAllocationView` rather than `NoAllocation`.

The local coordinator launches all subprocesses, validates their handshakes, and releases a start barrier only after every required worker is ready. Workers do not import target modules or materialize CDefs until after the barrier, then enter `RuntimeMode.WORKER`, apply assigned device visibility, merge `DRYML_WORLD_*` environment facts, and execute user code.

The stored `WorldAllocation` captures actual backend assignment. The launch envelope adds the computed `world_allocation_id` to the runtime view and process environment, avoiding a self-referential ID inside the canonical allocation payload. Per-worker execution records reference that allocation ID and include role/replica/rank/local-rank metadata.

Local-world CPU and accelerator assignment is enforced through the runtime allocation view and device-visibility environment, not by OS-level CPU affinity or hard memory limits. CPU affinity and memory limits remain runtime process-control features that require explicit opt-in and are not enabled by the Sprint 10 local-world coordinator.
