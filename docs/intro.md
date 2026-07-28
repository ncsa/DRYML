# Introduction

Status: draft.

DRYML is an object-centric framework for building, saving, loading, querying, and executing machine-learning workflows. Its core idea is that ML components should have stable, inspectable definitions separate from their runtime state.

Instead of treating models, datasets, training jobs, and metrics as unrelated Python objects, DRYML treats them as a graph of typed objects with reproducible construction metadata and separately persisted state.

## What DRYML Gives You

DRYML is designed around a few recurring needs:

- Reconstruct an object from a stable definition.
- Save and load object graphs without manually wiring every dependency.
- Query stored objects by their construction metadata.
- Preserve object identity separately from heavy runtime state.
- Move between ML backends while keeping shared concepts like tensor specs and datasets explicit.
- Track runtime context requirements for CPU, GPU, memory, and backend compatibility.

## The Central Lifecycle

The basic lifecycle is:

```text
Definition
    -> ConcreteDefinition
    -> Object
    -> Repo save
    -> Store state
    -> Repo query/load
    -> Object
```

`Definition` records how to construct something. `ConcreteDefinition` is the fully resolved, stable form. `Object` is the live runtime instance. `Repo` and `Store` are responsible for persistence and lookup.

## Minimal Example

```python
from dryml.core import Object, Repo


class MyThing(Object):
    def __init__(self, value):
        super().__init__()
        self.value = value


repo = Repo()
obj = MyThing(5, repo=repo)

repo.save_object(obj)
same_obj = repo.load(obj.definition)
```

The important point is that `obj.definition` is not a pickle of the object. It is the stable identity and construction description used by the repo.

## Major API Areas

### Session Runtime

`dryml.session` is the common runtime entry point. Fresh DRYML intentionally uses
unchecked Python mode, preserving inherited process and framework behavior.
`session.manage()` opts a current process into checked resource management;
`session.set_mode("orchestrator")` opts into checked worker planning without a
current workload allocation. The session guide explains immutable snapshots,
framework hooks, reset/restart boundaries, and advanced low-level alternatives:
[Sessions](session.md).

Core object system:

- `Object`
- `Serializable`
- `Definition`
- `ConcreteDefinition`
- `Repo`
- `DirStore`
- `ZipStore`

Query and persistence:

- stored, cached, known, and nested query domains
- definition result sets
- object result sets
- occurrence result sets
- aliases and main definitions

ML workflow APIs:

- `TensorSpec`
- `Dataset`
- `Model`
- `TrainFunction`
- `Experiment`
- `Artifact`

Runtime APIs:

- `dryml.session`
- advanced contexts
- resource specs
- backend compatibility checks

## What DRYML Is Not

DRYML is not only a serialization library. Object persistence is part of the system, but the larger goal is to preserve object identity, graph structure, queryability, and execution constraints.

DRYML is also not a single-backend ML framework. TensorFlow, PyTorch, JAX, NumPy, sklearn, and other integrations should plug into shared abstractions rather than replacing them.

## Where To Go Next

Read [Objects and Definitions](objects_and_defs.md) next. Most DRYML behavior depends on understanding the difference between a definition, a concrete definition, and a live object.
