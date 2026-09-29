# DRYML

DRYML provides immutable construction definitions, exact graph references, and local checkpoint publication for Python machine-learning objects.

## CDef V2

`Definition` is a soft declaration. Concretizing it produces a `ConcreteDefinition` (CDef): a fully bound structural recipe. CDef equality and `stable_hash()` compare class and semantic parameters; they intentionally do not include graph sharing or object identity. `CDef.graph_equal()` and `graph_hash()` additionally preserve shared versus independent equal nodes.

`ObjectRef` adds durable `ObjectId` lineage for every owned `Serializable` node. `StateRef` adds one immutable local-state hash for each of those paths. Inspecting CDefs, ObjectRefs, StateRefs, and structural queries is import-free. CDef `.args` and `.kwargs` are call projections and may import the current class; use `.cls` and `.parameters` for inspection.

```python
from dryml import Definition, Object, Repo, Serializable
from dryml.core.store.dir import DirStore


class Counter(Serializable):
    def __init__(self, value=0):
        super().__init__()
        self.value = value


repo = Repo(DirStore("./checkpoint-store"))
counter = Counter(3, repo=repo)
state = counter.save(repo=repo)
restored = repo.load_state_ref(state)
assert restored.object_ref == state.object
```

`Object.save()` publishes immutable directory state and returns a `StateRef`. `federated=False` is self-contained by default; `federated=True` may retain immutable dependencies in connected Stores. `deep_capture=True` captures each owned serializable node once. There is no automatic mutation detection: call `save()` after relevant mutation.

See [Objects and Definitions](docs/objects_and_defs.md), [Repos and Stores](docs/repos.md), [Formats](docs/formats.md), and [Graph Querying](docs/graph_querying.md).
