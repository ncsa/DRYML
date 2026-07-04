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

Workers and explicit inline execution should derive a `RuntimeAllocationView` from `WorldAllocation`, enter runtime mode, build/apply device visibility and bootstrap plans, then import frameworks or materialize objects. Framework-backed DRYML modules should import heavy frameworks only through `import_configured_framework(...)` inside an active `activate_runtime_bootstrap(...)` scope. Existing framework-object ingestion may read dtype/shape metadata without importing a framework; conversions that create framework-native objects still require active bootstrap. That activation must match the active runtime mode/allocation and records process-local bootstrap state separate from the exported environment marker. By default, runtime bootstrap includes the `plain` adapter and framework adapters named in `RuntimeContextSpec.frameworks`; callers that need strict pre-import checks can pass an explicit `FrameworkBootstrapPolicy(..., strict_preimport=True)`. CPU affinity and hard memory limits require explicit process-control opt-in in reusable current-process activation scopes.

Orchestrator and probe processes default to hidden workload accelerators through the `none` device visibility policy. Worker processes default to `assigned`, exposing only assigned devices such as `CUDA_VISIBLE_DEVICES=0` and hiding unassigned CUDA, HIP/ROCR, and XLA devices.

Legacy `Compute.__compute_reqs__` dictionaries are still supported as a transitional bridge. They are checked against the active `RuntimeAllocationView`; a CPU-only allocation does not satisfy a legacy GPU requirement.

## Legacy Packages

`dryml.context` is retained as a legacy compatibility surface for older code. New code should use `dryml.worlds` and `dryml.runtime` directly.

`dryml.execute` is retained as legacy local pickled-callable execution pending dispatch v2. Its subprocess worker entry path enters `dryml.runtime` worker mode and applies runtime bootstrap before loading the callable or materializing objects. Inline execution stays in the caller's current runtime and rejects legacy resource requirements unless a future explicit inline-allocation path is added.
