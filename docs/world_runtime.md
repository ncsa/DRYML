# World And Runtime

DRYML separates portable declarations from observations and process effects.
No declaration API launches work or reserves host resources.

## Layers

| Layer | Primary values | Responsibility |
| --- | --- | --- |
| Environment declaration | `EnvironmentRequirement`, `EnvironmentSpec` | Describe software constraints or a selector |
| Compatibility | Environment/world/runtime reports | Compare supplied evidence without activating it |
| Requested world | `WorldRequirement`, `WorldSpec` | Describe roles, replicas, and requested resources |
| Exact allocation | `WorldAllocation`, `ProcessAllocation` | Bind roles and ranks to exact local resources |
| Runtime declaration | `RuntimeContextSpec` | Describe effect-free visibility/framework intent |
| Runtime activation | `RuntimeState`, session generation | Publish `NONE`, `INLINE`, or `ORCHESTRATOR` state |
| Controls and adapters | Publication statuses | Report visibility, affinity, memory, and framework outcomes |

`dryml.annotations` carries consumer-owned process-local key/value metadata. It
has no built-in environment, world, or runtime semantics or resolution, and
does not serialize or persist annotations. See [Annotations](annotations.md).

## Passive Hard World Declarations

`dryml.worlds.req(...)` attaches one process-local hard `WorldRequirement` to a
class, standalone function, method definition, supported `staticmethod` or
`classmethod`, or supported custom descriptor. It returns the exact target
unchanged and never wraps or calls it, probes a host, reserves resources, or
activates runtime/session state. `dryml.world` is the lazy alias for the plural
world owner.

```python
from dryml import world


@world.req(cpus=2)
class Worker:
    def run(self):
        pass


result = world.requirements_for_method(Worker, "run")
assert result.has_value
```

`requirements_for(target)` combines direct or inherited declarations, while
`requirements_for_method(owner, method_name)` combines inherited class
declarations with one statically selected method. Instance owners use their
exact class without reading instance state or binding a descriptor. A
`RequirementResult` is empty when no world declaration exists, valued with one
complete `WorldRequirement` when compatible, or valueless with a bounded
conflict report.

World diagnostic paths preserve ordinary dotted spellings such as
`roles.main.resources.cpus`. To keep legal dotted role or resource names
unambiguous, those individual segments use deterministic JSON-style brackets,
for example `roles["trainer.gpu"].resources.named["license.v2"]`. This syntax
applies only to diagnostics; world declaration and resource-name contracts are
unchanged.

Omitted flattened constraints are unconstrained, so an omitted constraint such
as a replica count is not invented by `cpus=2`. Supplying `roles=` selects the complete multi-role form and
rejects simultaneous flattened fields rather than silently merging grammars.
World combination is independent of environment combination and ignores
environment annotations. Hard declarations are not defaults, selection,
automatic enforcement, code inference, dispatch, or active runtime/session
state. Compatibility returns a domain report whose policy-independent
`admission_ok` may be passed to the explicit shared admission barrier; neither
declaration nor reporting admits or runs work automatically.

## Planning A Local World

```python
from dryml.worlds import (
    CountConstraint,
    ResourceRequirement,
    RoleRequirement,
    WorldRequirement,
    assign_local_world,
    local_inventory,
    synthesize,
)

requirement = WorldRequirement({
    "main": RoleRequirement(
        replicas=CountConstraint(min=1, max=1),
        resources=ResourceRequirement(cpus=CountConstraint(min=2)),
    ),
})
inventory = local_inventory()
result = synthesize(requirement, inventory=inventory)
if not result.ok:
    raise RuntimeError(result.diagnostics)
requested = result.world
allocation = assign_local_world(requested, inventory=inventory)
```

Synthesis chooses the deterministic smallest feasible local shape. Assignment
produces disjoint exact CPU and accelerator bindings and rejects oversubscription
in this public scope. Unknown capacity, unsupported topology, and incompatible
per-device memory fail honestly. Planning does not create a process, worker,
reservation, dispatch handle, or remote backend.

## Runtime Publication

`python` publishes `NONE` with inherited visibility and no allocation.
`managed` publishes `INLINE` with exactly one selected process allocation.
`orchestrator` publishes `ORCHESTRATOR`, hides workload accelerators, and
enforces definition-only core behavior.

Publication validates the creating PID before locks, stages reversible effects,
checks fresh inventory dimensions, and atomically commits one immutable
generation. An admitted operation holds a generation lease, so an incompatible
transition receives `PublicationBusyError`. Uncertain irreversible effects or
failed rollback publish terminal failure and require process restart.

## Process Controls

The status vocabulary is closed to `undeclared`, `not-applicable`,
`pending-import`, `visibility-enforced`, `framework-configured`, `enforced`,
`declarative`, `unsupported`, and `failed`.

| Adapter | Visibility | Threads | Process memory | Accelerator memory |
| --- | --- | --- | --- | --- |
| Plain controls | Mandatory environment visibility; affinity may be unsupported | Best effort | Platform best effort or declarative | Not applicable |
| TensorFlow | Mandatory before import | Best effort after import | Declarative unless process-enforced | Best effort per device |
| Torch | Mandatory before import | Best effort after import | Declarative unless process-enforced | Best effort per known device |
| JAX/JAXlib | Mandatory shared group | Unsupported unless proven | Declarative | Best effort for supported uniform fractions |

Framework factories are lazy. Registration freezes on the first active
managed/orchestrator publication or watched-root observation. Late registration,
overlapping roots, direct callable factories, and roots already imported or
observed fail explicitly. A late framework import that defeats mandatory
visibility requires a fresh process.

## Authority And Lock Order

Store and query-index authority is independent of runtime publication. The
effective order is PID check, publication transition or generation lease,
watched-import epoch, framework registry/adapter state, process-effect ownership,
then any admitted core/Store operation. Code must not call session publication
while holding Store/index authority. Derived runtime status never replaces CDef,
Store, alias, revision, Object-state, or query-index recovery authority.

See [Sessions](session.md), [Environments](environments.md), and
[V1.1 Formats](formats.md).
