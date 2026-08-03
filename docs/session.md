# Sessions

`dryml.session` is the common, persistent setup surface for notebooks, REPLs,
and scripts. It describes the current process separately from the world a later
dispatch should run. It does not replace the advanced `dryml.environments`,
`dryml.worlds`, or `dryml.runtime` APIs.

## Start With The Intended Mode

Fresh DRYML is intentionally ordinary unchecked Python. Its low-level state is
`RuntimeMode.NONE + NoAllocation + OFF` with no enabled requirement axes.
Importing `dryml` does not change inherited device visibility, allocate
resources, import TensorFlow, PyTorch, or JAX, or add session requirement
checks. Leaving the facade untouched and explicitly selecting Python mode are
equivalent:

```python
import dryml

assert dryml.session.mode() == "python"
snapshot = dryml.session.set_mode("python")
assert snapshot.mode == "python"
```

Use a managed session before importing a GPU-capable framework when the current
process should have a checked, CPU-only allowance:

```python
snapshot = dryml.session.manage()
assert snapshot.mode == "managed"
assert snapshot.allocation is not None
assert snapshot.statuses["visibility"] == "visibility-enforced"
```

`manage()` uses CPUs available to this process under affinity, container, or
scheduler limits and exposes no accelerator. `manage(cpus=4, memory="8GiB")`
sets a smaller allowance. Process-memory allowances participate in annotated
requirement checks, but are currently `declarative`: DRYML does not claim to
stop arbitrary Python allocations. Per-device `accelerator_memory` is separate
from process memory; supporting adapters report their own allocator outcome and
must not be read as one aggregate process cap.

Use strict orchestration without a current workload allocation when the process
only builds definitions, inspects metadata, plans, or launches workers:

```python
snapshot = dryml.session.set_mode("orchestrator")
assert snapshot.allocation is None
```

Strict orchestration publishes `object_mode="definition"` session-wide. Nested
definition, concrete, selector, and space modes remain available, but `fresh`
and `load_or_build` transitions fail at context entry. DRYML APIs that would
newly construct, restore, reuse, or return a live Object raise
`RuntimeTransitionError` containing `Orchestration mode prohibits Object
materialization`. Use `Definition`/`ConcreteDefinition` APIs for control-plane
work, or run the workload in a managed fresh process or dispatched worker.

`managed` is a **managed session**, not a managed operation. Managed operations
remain the Store-backed `dryml.managed` lifecycle documented separately.

## Flat Calls And Complete Replacement

Every mutating call returns an immutable `SessionSnapshot`. Its nested controls
and statuses are immutable too, making a snapshot suitable for notebook display.
The current-process allowance, requested worker environment/world, environment
requirements, requirement axes, per-control status, generation, and health are
inspectable without exposing an internal role in the common representation.

```python
import dryml

snapshot = dryml.session.manage(cpus=2)
snapshot = dryml.session.worker_env_request(
    dryml.environments.CurrentEnvironmentSpec()
)
snapshot = dryml.session.worker_world_request(cpus=2, gpus=1)
snapshot = dryml.session.require_env("numpy>=1.26", python=">=3.10")
assert snapshot.requested_world is not None
```

`manage(...)` and `allocate_world(...)` replace the current-process allowance.
`worker_env_request(...)` and `worker_world_request(...)` replace only their
respective default candidates for later explicit dispatch. Repeated
`require_env(...)` calls merge hard software compatibility requirements
atomically. The current Python interpreter is implicit for ordinary direct work;
environment requirements describe software compatibility, not GPUs or device
visibility.

`enforce_requirements(environment=..., world=..., runtime=...)` atomically
replaces the three requirement axes. The shared enforcement action and enabled
axes affect compatibility reports, warnings, and failures only; they never
weaken allocation, materialization, worker-role, or framework-visibility guards.

Use `configure(...)` when one cell should replace the complete session. Its
`mode` is mandatory; omitted non-mode sections clear or return to the selected
mode's defaults, and invalid input leaves the prior generation unchanged:

```python
snapshot = dryml.session.configure(
    mode="managed",
    resources={"cpus": 2, "memory": "8GiB"},
    environment={"requirements": ["numpy>=1.26"], "python": ">=3.10"},
)
```

`allocate_world(...)` is the advanced exact-allocation path. It accepts a typed
`WorldAllocation` or canonical envelope and needs explicit `role` and `replica`
selectors for a multi-process allocation. Multi-role worlds, exact device IDs,
candidate registries, and scoped low-level activation remain advanced APIs.

## Current Process And Workers

The session deliberately keeps two resource facts separate:

- The managed allowance controls direct work in this process.
- The requested worker environment/world are inputs to later dispatch planning.

Therefore a CPU-only notebook can request and dispatch a GPU worker without
making its own GPU visible:

```python
dryml.session.manage(cpus=2)
dryml.session.worker_world_request(cpus=2, gpus=1)
# dryml.dispatch.run(...) allocates the requested worker world, not the notebook allowance.
```

Dispatch captures one immutable session generation. Environment precedence is
explicit, annotation default, context-local, session request, resolver, then
fallback; world precedence is explicit, annotation default, context-local,
session request, synthesis, then fallback. Session requirements apply to worker
resolution outside Python mode; Python contributes only its configured worker
candidates. An explicit dispatch policy cannot make the current-process allowance
into worker capacity.

Direct managed calls preserve their direct or local Store-backed lifecycle and
never auto-dispatch. Explicit dispatch defaults independently to strict
compatibility with all three requirement axes, then resolves complete canonical
environment, world, runtime, and allocation selections for the child.

## Framework Hooks And Statuses

Setup stays lightweight. Registered root imports of TensorFlow, PyTorch, and JAX
traverse DRYML's pre-import and post-import hooks when a managed or orchestrator
generation is active. Raw `import torch`, `import tensorflow`, and `import jax`
remain ordinary Python syntax. Mandatory visibility is established before module
execution and fails closed; optional thread, allocator, or memory controls report
their actual per-adapter status as `pending-import`, `visibility-enforced`,
`framework-configured`, `declarative`, `unsupported`, or `failed`.

Do setup before importing a known GPU framework. Repeating identical setup after
a controlled import is safe, but any transition that could change visibility
after such an import raises restart guidance and preserves the prior generation.
`reset()` and `set_mode("python")` restore the ordinary baseline only when
session-owned process effects can safely be restored. A failed terminal session
also requires a fresh process.

Direct hard-annotated functions and supported methods are checked in managed
mode before user code. Python mode intentionally bypasses session-derived checks;
unannotated direct code receives only process-level controls, not an inferred
allocation guarantee. Supported automatic boundaries are functions, bound and
unbound methods, static/class methods, managed descriptors, and explicitly
decorated `__new__`, `__init__`, and `__call__`. The class-object/custom-metaclass
invocation, properties and other custom descriptors, post-decoration assignment,
and pre-decoration references are unsupported automatic boundaries.
Dispatch and tracing use a private scoped bypass only while reconstructing a
known target; workers perform their own allocation checks.

Long-lived checked calls hold a generation lease. A visibility or other
process-effect transition that would invalidate an active lease fails busy rather
than changing the running call's process state. Declarative-only updates may
publish for future work.

## Reset And Advanced APIs

Call `dryml.session.reset()` when the notebook can safely return to its ordinary
Python baseline. A reset clears facade categories, but it cannot retroactively
undo an imported framework or another process's changes. Handle restart-required
examples only in a disposable child process.

For temporary context-local planning defaults, multi-role worlds, exact runtime
activation, environment candidate registries, and explicit enforcement scopes,
use `dryml.environments`, `dryml.worlds`, and `dryml.runtime` directly. Those
low-level APIs remain supported; see [world/runtime](world_runtime.md),
[environments](environments.md), and [dispatch](dispatch.md).
