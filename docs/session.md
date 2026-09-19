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

## Resource Cache

`session.resource_cache()` opts the current thread/task into one inspectable,
process-local resource cache. It does not allocate a runtime, open a Store,
reconstruct a Repo, or change `current_repo`. The selected core Repo and its
already-connected Stores are registered as borrowed entries when activation
begins; later core selection changes register the new borrowed Repo without
removing the old entry.

```python
from dryml import session

with session.resource_cache() as cache:
    assert session.current_resource_cache() is cache
    selected_repos = cache.repos
    selected_stores = cache.stores

assert session.current_resource_cache() is None
```

`repos` and `stores` return fresh immutable membership tuples containing the
actual local handles. Inspection has no I/O or ownership effect: it does not
scan a Store, reconstruct resources, run work, or permit a caller to close an
entry. The outermost exit clears membership. Retaining an earlier tuple does not
extend a cache-owned lifetime or make a resource valid after its owner closes it.

Nested activation in the same thread/task reuses one cache and only the outer
activation tears it down. Copied task or thread contexts reject cache use rather
than silently sharing it; independently entered noninherited contexts receive a
separate cache. While an entry is registered, raw `Repo.close()` and supported
Store `close()` calls fail. Borrowed resources are never closed by cache teardown.
Core selection changes that would close a leased owned Repo fail before replacing
the old selection, and a temporary owned `dryml.config(repo=...)` scope is
rejected before entry when an active cache could prevent safe restoration.

Repos reconstructed through an active cache borrow their shared Stores and are
leased until outermost exit. Teardown first removes public membership, then
closes cache-created Repos with `flush=False`, followed by cache-owned Stores.
It never commits a dirty ZipStore or runs deletion-save work. If deterministic
Repo or Store cleanup fails, `RepoReconstructionError` retains the failed
dependency set for explicit idempotent `cleanup()` retry; an ordinary workload
exception remains primary and keeps that cleanup owner attached.

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

`session.snapshot_for_generation(generation)` projects the exact immutable
`SessionGeneration` already held by a publication lease. It does not call
`publication.current()`, reread the host, or synthesize a newer session view.
Admission owners use this seam while holding their lease so the evidence they
check is tied to the same generation that remains protected through their final
operation.

Dispatch's explicit `InProcess()` route uses this projection while holding the
publication lease through final requirement checks and one synchronous direct
call. It does not reconfigure Session, manufacture an allocation from host
inventory, or extend the lease to later consumption of returned lazy data.
An incompatible concurrent publication receives `PublicationBusyError` through
the runtime owner.

## Scope

This session facade configures only the current process. It does not publish
future-worker state, dispatch functions, wrap direct calls, start workers,
select providers, run probes, install packages, or migrate persisted data.
Generic `dryml.execute` and legacy context APIs are independent subsystems and
are not integrated by these session declarations. `dryml.core.execute` reads the
submission caller's core Repo/cache defaults once when preparing a call, but it
does not transfer this process session. Its worker setup installs a temporary,
task/thread-owned core session only after runtime activation; it restores it on
exit and never changes caller session state.

See [World And Runtime](world_runtime.md), [Objects and Definitions](objects_and_defs.md),
and [Repos and Stores](repos.md).
