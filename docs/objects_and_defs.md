# Objects and Definitions

Status: draft.

The object/definition split is the core DRYML concept. A runtime object can have mutable state, allocated resources, and backend-specific handles. A definition is the stable description of how that object is constructed.

## The Three Layers

`Definition`:

An immutable partial construction expression. It records only the class and
arguments that the caller supplied. Omitted constructor fields remain omitted:
this is what makes a `Definition` suitable for selectors and search spaces as
well as construction recipes.

`ConcreteDefinition`:

An immutable, fully bound canonical definition (CDef). New CDefs use the V2
identity format: they capture every effective declared parameter, including
defaults, and are the stable identity used for hashing, equality, graph
indexing, save paths, and exact matching.

`Object`:

The live Python instance associated with one concrete definition. It may hold runtime state, backend objects, open resources, caches, or trained weights.

## Typical Flow

```python
from dryml.core import Object, Definition, Repo


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

This materializing example applies in `python` or `managed` mode. In strict
`orchestrator` mode, constructor syntax produces definitions through the public
object-mode floor and supported live-Object entry points reject before side
effects.

## Partial Expressions And Exact CDefs

`Definition` and `ConcreteDefinition` deliberately have different omission
rules. A partial `Definition` records intent without applying defaults:

```python
soft = Definition(Layer, width=32)
assert soft.parameters["width"] == 32
# A selector built from soft does not constrain omitted Layer parameters.
```

Concretization fully binds the prepared constructor call, applies declared
defaults, applies argument roles, and canonicalizes the resulting values. New
V2 CDefs therefore normalize equivalent call spellings:

```python
first = Definition(Layer, 32).concretize()
second = Definition(Layer, width=32).concretize()

assert first == second
assert first.stable_hash() == second.stable_hash()
```

If `Layer` declares another defaulted parameter, that parameter is present in
`first.parameters` even when it was omitted by the caller. The effective value
is captured when the V2 CDef is created. A later call after a Python default
changes produces a later identity; existing V2 CDefs retain their captured
value.

Every effective default, including one supplied dynamically by
`__prepare_args__`, must cross the same canonical boundary as an explicitly
passed constructor value. An unsupported runtime value therefore cannot be a
V2 default: use a canonical identity value instead, or create the dynamic
resource as runtime state after construction. Concretization raises `TypeError`
with the semantic parameter path, such as `client` or `options/client`, so the
failing default can be replaced with a canonical value or moved out of the
definition.

This does not make partial expressions exact. `Definition.parameters`,
`Selector.parameters`, and `SearchSpace.parameters` contain supplied fields
only. When a live constructor signature is safely available, positional and
keyword selector spelling for the same supplied field constrain the same V2
semantic parameter. An unresolved `ImportRef` or `SourceSpec` selector must
use semantic keyword spelling or `SKIP_ARGS` when positional spelling would
need class resolution; DRYML does not resolve code merely to interpret it.

## Semantic Parameter Access

V2 CDefs expose their persisted semantic parameters directly and through an
immutable mapping. The mapping is the complete collision-safe interface:

```python
experiment = Definition(Experiment, model=Definition(Layer, width=32)).concretize()

model = experiment.model
assert model is experiment.parameters["model"]
```

Existing DRYML attributes and methods win name collisions. For example, a
constructor parameter named `args`, `kwargs`, `build`, or `stable_hash` is
always available as `cdef.parameters["args"]`,
`cdef.parameters["kwargs"]`, and so on, rather than replacing the framework
member. Values returned by direct access and the mapping are canonical values;
they do not materialize runtime objects or resolve optional backends.

V2's `.args` and `.kwargs` are compatibility accessors, not semantic inspection
or identity surfaces. At materialization time they resolve the current class
signature and project the persisted names and values into a call. They can
therefore import a backend or raise a current-signature error. Materialization
does not rerun preparation or reapply defaults.

## Identity Versions And Legacy Data

V1 CDefs created in the supported `dryml.core` namespace retain their stored raw `cls`/`args`/`kwargs` identity, hash, paths,
and materialization behavior. V1 and V2 CDefs are distinct exact identities,
even when their visible constructor values look equivalent. DRYML reads V1
records but does not map, migrate, or silently substitute them with V2 records.

V1 symbolic class references remain inspectable without class resolution. A
V1 pickle that embeds a raw live class retains the ordinary Python import
requirement of that pickle; it is not made import-free. V1 omitted defaults are
also legacy raw-call behavior: DRYML does not reconstruct a historical default
value that was never stored.

The `dryml.core2` to `dryml.core` promotion is a clean persisted-identity break.
Pickles, symbolic references, hashes, and Store roots naming `dryml.core2` are
not decoded, aliased, or migrated. This V1/V2 behavior does not make
pre-promotion `dryml.core2` data compatible.

## Definition Mode

DRYML can capture constructor calls as definitions instead of building runtime objects.

```python
from dryml.core import definition_mode

with definition_mode():
    layer_def = Layer(width=64)
```

Inside definition mode, class construction returns a `Definition` rather than a live object. This is useful for composing object graphs declaratively.

## Definition-Only Orchestration

Strict `orchestrator` mode keeps these operations available without resolving
optional-backend classes or returning live Objects:

- create and inspect `Definition` and V1/V2 `ConcreteDefinition` values
- inspect V2 semantic parameters, hash identities, and traverse CDef graphs
- build selectors, references, requested worlds, and materialization plans
- run structural/exact definition and occurrence queries
- hydrate stored CDefs and validate, reconcile, or rebuild derived indexes
- update aliases and main references when the target is a CDef

Strict orchestration blocks these operations before constructor, cache return,
class resolution/projection, user hook, serialization, state I/O, or Store
mutation:

- direct live construction and `Definition.build()`
- Repo load, load-or-build, live cache access, and object-yielding query access
- materialization-plan execution and canonical runtime reconstruction
- Object, Repo, Store, and Zip save/restore paths
- public `fresh` and `load_or_build` core object-mode selection

`dryml.runtime.materialization_scope("warn" | "off")` is a trusted advanced
override for one admitted operation. It retains one generation lease through
nested and lazy work. It does not restore accelerator visibility, change the
public `definition` floor, bypass Store publication safety, or create a dispatch
facility.

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

Saving and restoring are live-Object operations and therefore reject in strict
orchestration.

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
- [Sessions](session.md)
- [World And Runtime](world_runtime.md)
- [Models API](models.md)
- [Artifacts API](artifacts.md)
