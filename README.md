# DRYML

DRYML provides immutable construction definitions, exact graph references, and local checkpoint publication for Python machine-learning objects.

## Composable Object State

`Definition` authors an immutable Node graph. Concretizing it produces a
`ConcreteNode`: normalized construction structure, independently of saved-state
completeness. Each node can select an ObjectId and a compatible local StateKey
independently. `recipe.bind(identities=..., states=...)` composes constraints;
contradictory assignments through aliases fail rather than silently overriding.

Node handle equality identifies an occurrence/view; use `graph_equal` for exact
graph semantics and `object_projection_equal` for state-independent realized
identity. SavedNode is the complete-saved typing refinement of the same graph,
validated by `require_saved`, not another graph model. Metadata inspection does
not construct Objects or read private state. Concrete `.args`/`.kwargs` call
projections may resolve the current class; use `.cls`/`.parameters` for inspection.

```python
from pathlib import Path
from dryml import Repo, Serializable
from dryml.core.store.dir import DirStore


class Counter(Serializable):
    """Persist one mutable integer independently of its construction recipe."""

    def __init__(self, value=0):
        super().__init__()
        self.value = value

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        """Write the current value to the framework-owned private data directory."""
        Path(dest_dir, "value").write_text(str(self.value), encoding="ascii")

    def restore_state_from_dir_imp(self, src_dir, *, codec):
        """Restore a previously authenticated value; invalid integer data raises."""
        self.value = int(Path(src_dir, "value").read_text(encoding="ascii"))


repo = Repo(DirStore("./checkpoint-store"))
counter = Counter(3, repo=repo)
counter.value = 5
state = counter.save(repo=repo)
restored = repo.load_state_ref(state, reuse_live="never")
assert restored.value == 5
assert restored.object_id == counter.object_id
assert repo.describe(restored).object_projection_equal(state)
```

`Object.save()` publishes a retained SavedNode through connected durable Stores.
Each distinct live stateful node needing fresh state is captured once; explicit
`states=` reuse avoids recapturing selected bytes. A last-save receipt does not
prove unchanged mutable state. Store 4 uses CDef-owned state buckets and full
StateKeys; old Stores are rejected intact, without automatic migration.

Managed methods return `InvocationResult`, with lazy `.value`, detached
`.status`/`.execution`, and `.final_state_ref`. Compatible completed calls replay
without running, restoring or saving the receiver. Explicit rerun controls start
selection; missing selected state never silently initializes or falls back.

See [Objects and Definitions](docs/objects_and_defs.md), [Repos and Stores](docs/repos.md), [Formats](docs/formats.md), and [Graph Querying](docs/graph_querying.md).
See [Managed Operations](docs/managed_operations.md) and the
[verification record](docs/object_state_lifecycle_verification.md) for retention,
recovery, measured work and integrated verification results.
