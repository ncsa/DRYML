# Sessions

`dryml.session` is the common, persistent setup surface for notebooks, REPLs,
and scripts. It describes the current process separately from the world a later
dispatch should run. It does not replace the advanced `dryml.environments`,
`dryml.worlds`, or `dryml.runtime` APIs.

## Start With The Intended Mode

Fresh DRYML is intentionally ordinary unchecked Python. Importing `dryml` does
not change inherited device visibility, allocate resources, import TensorFlow,
PyTorch, or JAX, or add session requirement checks. Leaving the facade untouched
and explicitly selecting Python mode are equivalent:

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

Use checked orchestration without a current workload allocation when the process
only plans or launches workers:

```python
snapshot = dryml.session.set_mode("orchestrator")
assert snapshot.allocation is None
```

`managed` is a **managed session**, not a managed operation. Managed operations
remain the Store-backed `dryml.managed` lifecycle documented separately.

## Flat Calls And Complete Replacement

Every mutating call returns an immutable `SessionSnapshot`. Its nested controls
and statuses are immutable too, making a snapshot suitable for notebook display.
The current-process allowance, requested worker world, environment requirements,
per-control status, generation, and health are inspectable without exposing an
internal role in the common representation.

```python
import dryml

snapshot = dryml.session.manage(cpus=2)
snapshot = dryml.session.request_world(cpus=2, gpus=1)
snapshot = dryml.session.require_env("numpy>=1.26", python=">=3.10")
assert snapshot.requested_world is not None
```

`manage(...)` and `allocate_world(...)` replace the current-process allowance.
`request_world(...)` replaces only the default worker intent. Repeated
`require_env(...)` calls merge compatible software requirements atomically. The
current Python interpreter is implicit for ordinary direct work; environment
requirements describe software compatibility, not GPUs or device visibility.

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
- The requested worker world is input to later dispatch planning.

Therefore a CPU-only notebook can request and dispatch a GPU worker without
making its own GPU visible:

```python
dryml.session.manage(cpus=2)
dryml.session.request_world(cpus=2, gpus=1)
# dryml.dispatch.run(...) allocates the requested worker world, not the notebook allowance.
```

Dispatch captures one immutable session generation. Its world precedence remains
explicit dispatch input, annotation default, context-local world, session request,
synthesis, then fallback. Session requirements apply to worker resolution outside
Python mode; an explicit dispatch policy cannot make the current-process
allowance into worker capacity.

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
