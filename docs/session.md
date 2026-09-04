# Sessions

`dryml.session` is the persistent, process-wide configuration facade. Every
successful mutation publishes an immutable runtime generation. It is separate
from `dryml.core.session`, whose `dryml.configure(...)` and `dryml.config(...)`
APIs select a repository, cache policy, and object-construction mode.

## Public Modes

| Public mode | Low-level runtime | Current-process allocation | Object behavior |
| --- | --- | --- | --- |
| `python` | `NONE` | None | Ordinary core configuration applies |
| `managed` | `INLINE` | One exact role-qualified process | Live Objects are permitted after controls publish |
| `orchestrator` | `ORCHESTRATOR` | None | Strict definition-only floor |

Use `session.set_mode(...)` for a mode transition, `session.manage(...)` for a
concise local allocation, or `session.allocate_world(...)` to select one process
from an exact `WorldAllocation`.

```python
from dryml import session

managed = session.manage(cpus=2, gpus=0)
assert managed.mode == "managed"
assert managed.allocation.role == "main"

session.set_mode("orchestrator")
assert session.mode() == "orchestrator"
```

An allocation selection is always role-qualified and identifies its replica,
global rank, local rank, exact CPU IDs, memory, accelerator IDs, environment,
and bounded metadata. DRYML does not infer a future worker from this selection.

## Requirement Axes

Session configuration contains exact boolean `environment`, `world`, and
`runtime` axes. `python` defaults all three to false; `managed` and
`orchestrator` default all three to true. `enforce_requirements(...)` replaces
the complete mask.

The environment axis validates a non-empty managed environment requirement.
The world and runtime axes are parity-preserving configuration and identity in
this release; they have no automatic enforcement consumer. No axis can weaken
device visibility, publication integrity, Store safety, or the orchestrator
materialization floor.

The passive `dryml.annotations` kernel has no session integration. It attaches
consumer-owned process-local metadata only; it does not resolve requirements,
mutate the session, launch work, or activate frameworks.

Hard declarations from `dryml.environments.req(...)` and `dryml.worlds.req(...)`
remain process-local passive metadata. They are not session or worker defaults,
do not select a runtime or allocation, and receive no automatic enforcement from
this facade. Consumers resolve, check, and explicitly admit them outside session
mutation; see [Hard Requirements](requirements.md).

## Persistent And Scoped Configuration

`dryml.session.configure(...)` atomically replaces persistent process-session
categories. In contrast, root `dryml.configure(...)`, `dryml.config(...)`, and
`dryml.status()` are aliases for `dryml.core.session` and control object/repo
behavior in the current context.

During orchestration, core status reports the configured mode as
`requested_object_mode`, the projected `definition` mode as
`effective_object_mode`, and `orchestrator_floor=True`. Definition-like scopes
(`definition`, `concrete`, `selector`, and `space`) remain usable. Public
`fresh` and `load_or_build` selection fails before context mutation.

## Framework Lifecycle

Managed and orchestrator publication installs mandatory visibility before a
watched TensorFlow, Torch, JAX, or JAXlib module executes. Successful import
then publishes framework status in a new immutable generation within the same
control epoch. JAX and JAXlib share one adapter lifecycle.

Importing a watched framework before a visibility-changing session transition
requires a fresh process. Terminal publication failure and unsafe inherited
post-fork state also require restart. Contention with an admitted generation
lease fails explicitly rather than publishing incompatible controls.

Public snapshots retain environment keys but redact values, recognizable
credentials, and direct local paths. Diagnostics are bounded. This release does
not claim exhaustive redaction of third-party exceptions or that semantic IDs
make weak declared secrets safe.

## Scope

This session facade configures only the current process. It does not publish
future-worker state, dispatch functions, wrap direct calls, start workers,
select providers, run probes, install packages, or migrate persisted data.
Existing `dryml.execute` and legacy context APIs are independent subsystems and
are not integrated by these session declarations.

See [World And Runtime](world_runtime.md), [Objects and Definitions](objects_and_defs.md),
and [Repos and Stores](repos.md).
