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

`Compute` subclasses use runtime allocation guards only. Resource shape requirements belong in world annotations or `WorldRequirement` values and should be validated before entering worker runtime.

## Local Dispatch

`dryml.dispatch.LocalSubprocessBackend` is the new spec/record/runtime-aware local path. Its worker enters `RuntimeMode.WORKER` with a real CPU-only `RuntimeAllocationView` by default, applies assigned device visibility, and only then imports target functions or materializes CDef arguments from shared `DirStore` refs. This preserves the runtime setup order required for later multi-worker orchestration.

## Local Multi-Worker Runtime

`dryml.dispatch.run_world(...)` and `Dispatcher.run_world(...)` allocate a requested `WorldSpec` into a stored `WorldAllocation` with one `ProcessAllocation` per role replica. Rank and local rank are deterministic: sorted role name, then replica index. Each worker receives `WorldAllocation.runtime_view(role, replica, world_allocation_id=...)`, so CPU-only workers still have a real `RuntimeAllocationView` rather than `NoAllocation`.

The local coordinator launches all subprocesses, validates their handshakes, and releases a start barrier only after every required worker is ready. Workers do not import target modules or materialize CDefs until after the barrier, then enter `RuntimeMode.WORKER`, apply assigned device visibility, merge `DRYML_WORLD_*` environment facts, and execute user code.

The stored `WorldAllocation` captures actual backend assignment. The launch envelope adds the computed `world_allocation_id` to the runtime view and process environment, avoiding a self-referential ID inside the canonical allocation payload. Per-worker execution records reference that allocation ID and include role/replica/rank/local-rank metadata.

Local-world CPU and accelerator assignment is enforced through the runtime allocation view and device-visibility environment, not by OS-level CPU affinity or hard memory limits. CPU affinity and memory limits remain runtime process-control features that require explicit opt-in and are not enabled by the Sprint 8 local-world coordinator.
# Local Inventory And Synthesis

`worlds.local_inventory()` discovers CPU, memory, and explicitly declared local
accelerators without importing ML frameworks or activating a runtime allocation.
`worlds.synthesize(requirement, inventory=...)` returns a
`WorldSynthesisResult`; use `.require_world()` to obtain its requested local
`WorldSpec`, not an allocation. The allocator later assigns actual disjoint CPU
and accelerator identifiers, keeping requested worlds separate from worker
allocations. Default `lightweight` inventory avoids framework imports and uses
CPU affinity, platform-native memory facts, explicit `DRYML_LOCAL_ACCELERATORS` input, and
conservative numeric GPU device files under `/dev`; the opt-in `external` policy
forwards a timeout to an injected command runner without importing framework
bindings. Custom in-process runners are cooperative and must enforce a hard
deadline themselves. Memory capacity honors an explicit cgroup limit when one is
available. Linux uses `/proc/meminfo` constrained by cgroup v1/v2 limits;
Windows uses `GlobalMemoryStatusEx` available physical memory; macOS uses
available pages when exposed and otherwise its native physical-memory fact;
other POSIX hosts use `sysconf` page facts. Unknown capacity blocks positive
memory requests, and
unsupported topology, named resources, and devices fail synthesis rather than
being silently dropped. An explicit `DRYML_LOCAL_ACCELERATORS` declaration is
authoritative and is never broadened by optional external discovery. Device-root
inspection is conservative and bounded; ambiguous visibility or an
oversized device directory reports no accelerator inventory rather than a
partial claim.

Inherited numeric CUDA/NVIDIA visibility restricts device-root and external GPU
evidence alike. Disabled or ambiguous visibility produces no usable GPU capacity,
so optional host discovery cannot broaden a scheduler or container allocation.

Device-file evidence is accepted only for readable/writable character devices;
regular files or stale names such as `nvidia0` never create usable GPU capacity.

`LocalWorldFuture.cancel()` requests cancellation while preserving worker
responses for a later `result()` call. Use `LocalWorldFuture.close()` when no
aggregate result is needed and the local-world work directory should be removed
immediately.

Synthesis visits role names in sorted order and chooses the smallest positive
replica and executable CPU counts permitted by each constraint, plus the minimum
requested memory and accelerator counts. It proves aggregate disjoint CPU,
memory, and accelerator capacity across every replica before returning a local
world; failures report required, available, and shortfall capacity facts.

The local-subprocess backend can enact only one role/replica. It allocates that
requested world into an actual local-subprocess allocation and applies assigned
accelerator visibility before importing the target. Multi-role or multi-replica
worlds use `plan_world(...)`/`run_world(...)`; `world=None` there synthesizes an
omitted hard requirement from the same inventory used for allocation. Inventory
and probe results are intentionally not cached across plans.
