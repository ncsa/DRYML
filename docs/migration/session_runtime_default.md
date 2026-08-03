# Session Runtime Default Migration

## What Changed

Omitting runtime/session setup now intentionally means ordinary Python:
`dryml.session.mode()` is `"python"`, runtime enforcement is off, and DRYML does
not change inherited framework or device behavior. This replaces the prior
documentation assumption that an omitted mode behaved as strict orchestration.
The serialized low-level role is explicitly `none` (`RuntimeMode.NONE`), not an
unchecked orchestrator; legacy role data is not silently reclassified.

## Choose The Explicit Behavior You Need

```python
import dryml

# Ordinary unchecked Python (also the default).
dryml.session.set_mode("python")

# Checked current-process work before importing GPU frameworks.
dryml.session.manage(cpus=4)

# Checked planning/dispatch with no current workload allocation.
dryml.session.set_mode("orchestrator")
```

Use `session.worker_env_request(...)` and `session.worker_world_request(...)` to
set concrete default candidates for a future worker. They do not change the
managed notebook allowance, so a CPU-only notebook can request a GPU worker.
Use `session.require_env(...)` only for hard Python/software compatibility, not
GPU selection. `configure(...)` atomically replaces the complete session when
one declaration is preferred.

## Update Old Notebook Setup

Replace manual context lifetimes and a single default role with facade calls:

```python
# Before: nested environments/worlds/runtime scopes and a hand-built role.
# After:
snapshot = dryml.session.manage(cpus=2)
snapshot = dryml.session.worker_env_request(
    dryml.environments.CurrentEnvironmentSpec()
)
snapshot = dryml.session.worker_world_request(cpus=2, gpus=1)
```

Use `dryml.environments`, `dryml.worlds`, and `dryml.runtime` directly only for
advanced temporary scopes, exact/multi-role allocations, candidate registries,
or explicit activation. `runtime.plain()` remains advanced trusted inline work;
it is not dispatch isolation.

## Framework And Direct-Call Boundaries

Call managed/orchestrator setup before raw TensorFlow, PyTorch, or JAX imports.
Registered roots use DRYML hooks; mandatory visibility fails closed and optional
controls report their own status. A visibility-changing transition after a known
framework import requires a fresh process. `reset()` clears facade categories only
when session-owned process effects are safely restorable.

In managed mode, hard-annotated supported functions and methods are checked
before their body. Python mode intentionally bypasses these session checks;
unannotated code has only process-level guarantees. Class-object/custom-metaclass
calls, properties/custom descriptors, post-decoration assignment, and
pre-decoration references remain unsupported direct-call interception.

`managed session` does not mean `managed operation`: Store-backed managed
`compute`/`train` lifecycles keep their existing contract.

## Dispatch And Strict Orchestration Breaking Changes

Explicit dispatch now defaults to strict compatibility on environment, world,
and runtime axes regardless of the caller session. Each launch carries complete
canonical environment, world, runtime, and allocation selections; the child
publishes its strict worker session before handshake or workload setup.

Serialized `none` is required for the new no-role runtime. V1 execution envelopes
are rejected with migration guidance because they cannot express the mandatory
complete worker selections; recreate the dispatch plan rather than relying on
worker-side defaults.

Strict orchestration is intentional breaking behavior for local materialization
and local managed workload execution. It sets a session-wide definition mode:
definition/concrete/selector/space work remains available, while `fresh`,
`load_or_build`, and APIs that newly return live Objects fail with
`Orchestration mode prohibits Object materialization`. Set up orchestration before
framework/project imports, then dispatch the workload or use a managed fresh
process when live execution is required.
