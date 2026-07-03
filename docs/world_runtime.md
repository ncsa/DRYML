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

Workers and explicit inline execution should derive a `RuntimeAllocationView` from `WorldAllocation`, enter runtime mode, build/apply device visibility and bootstrap plans, then import frameworks or materialize objects.

Orchestrator and probe processes default to hidden workload accelerators through the `none` device visibility policy. Worker processes default to `assigned`, exposing only assigned devices such as `CUDA_VISIBLE_DEVICES=0`.

## Legacy Packages

`dryml.context` is retained as a legacy compatibility surface for older code. New code should use `dryml.worlds` and `dryml.runtime` directly.

`dryml.execute` is retained as legacy local pickled-callable execution pending dispatch v2. Its worker entry path now enters `dryml.runtime` worker mode instead of using `dryml.context` as the architectural center.
