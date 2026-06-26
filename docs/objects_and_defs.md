# Objects and Definitions

Status: draft.

The object/definition split is the core DRYML concept. A runtime object can have mutable state, allocated resources, and backend-specific handles. A definition is the stable description of how that object is constructed.

## The Three Layers

`Definition`:

An unresolved construction recipe. It records a class plus constructor arguments, but those arguments may still contain other definitions or values that require repo/config resolution.

`ConcreteDefinition`:

A fully resolved, canonical definition. This is the stable identity used for hashing, equality, graph indexing, save paths, and query matching.

`Object`:

The live Python instance associated with one concrete definition. It may hold runtime state, backend objects, open resources, caches, or trained weights.

## Typical Flow

```python
from dryml.core2 import Object, Definition, Repo


class Layer(Object):
    def __init__(self, width):
        super().__init__()
        self.width = width


repo = Repo()

soft = Definition(Layer, width=32)
concrete = soft.concretize(repo=repo)
obj = Layer(width=32, repo=repo)

assert obj.definition == concrete
```

Most users construct objects directly. DRYML still captures the concrete definition behind the object.

## Definition Mode

DRYML can capture constructor calls as definitions instead of building runtime objects.

```python
from dryml.core2 import definition_mode

with definition_mode():
    layer_def = Layer(width=64)
```

Inside definition mode, class construction returns a `Definition` rather than a live object. This is useful for composing object graphs declaratively.

## Concrete Definitions As Identity

Concrete definitions are stable identities. They are used to:

- compute stable hashes
- place object state in stores
- detect repeated subgraphs
- query stored objects
- compare exact object definitions
- rebuild runtime objects later

A concrete definition is not the same thing as runtime state. For example, trained weights, cached datasets, and generated artifacts should be stored as object state or artifact payloads, not as constructor identity.

## Object Graphs

Constructor arguments can contain other DRYML objects or definitions. DRYML can view these as a graph of concrete definitions.

```text
Experiment
    -> model
        -> encoder
        -> decoder
    -> dataset
```

The graph is useful for saving nested objects, querying owners of nested definitions, and determining which runtime objects must be available to materialize a larger object.

## Saving State

`Object.save_state_to_dir()` writes the definition and delegates runtime state persistence to `save_state_to_dir_imp()`.

```python
class Counter(Object):
    def __init__(self, start=0):
        super().__init__()
        self.count = start

    def save_state_to_dir_imp(self, dest_dir, revision=None):
        # Write runtime state here.
        pass

    def restore_state_from_dir_imp(self, src_dir, revision=None):
        # Restore runtime state here.
        pass
```

Subclasses can use `Serializable` or `Pickleable` when the default persistence behavior is enough.

## Definition Versus State

Use constructor arguments for stable identity. Use saved state for values that are produced after construction.

Good definition fields:

- architecture choices
- hyperparameters
- dataset source identity
- preprocessing configuration
- dependency objects

Good saved-state fields:

- trained weights
- fitted scalers
- cached outputs
- generated metric values
- external resource snapshots

## Common Pitfalls

- Mutating constructor arguments after object creation can make behavior confusing.
- Putting large runtime values in definitions can make hashes and queries expensive.
- Random IDs in constructor arguments make every object identity unique.
- Backend handles should normally be runtime state, not definition identity.
- Equality and storage identity follow the concrete definition, not Python object identity.

## Related Docs

- [Repos and Stores](repos.md)
- [Models API](models.md)
- [Artifacts API](artifacts.md)
