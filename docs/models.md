# Models API

Status: draft.

The Models API provides DRYML object abstractions for model composition, training, evaluation, and backend integration. Models are DRYML objects and methods, so they can participate in object graphs, datasets, repos, and saved workflows.

## Core Types

Important public types:

- `Model`
- `AutoEncoder`
- `TrainFunction`
- `TrainState`
- `Experiment`

Backend packages add specialized wrappers for TensorFlow, PyTorch, sklearn, XGBoost, and other frameworks.

## Model As Method

The base `Model` is a `dryml.methods.Method`, which means it can be used in
dataset mapping and Method pipelines. Import authoring APIs from
`dryml.methods`; the legacy `dryml.code` Method imports are removed.

```python
from dryml.data import Map

predictions = Map(dataset, model)
```

The method interface supports direct selection and pure output-spec inference.
Spec-aware callers such as `Map` select one local implementation from the input
spec; this does not change the model's eager/learning Method state. If a model
has an explicit `output_spec`, it propagates batching information from the input
spec.

## Output Specs

Models can infer output specs from input specs when the wrapper has pure
framework metadata. DRYML never calls a model with fabricated data or changes
train/eval mode to infer a spec. Supply `output_spec` for custom or opaque
models that have no supported metadata route.

```python
from dryml.core import TensorSpec
from dryml.models import Model

model = Model(output_spec=TensorSpec("float32", shape=(10,)))
```

When the input spec is batched and the output spec is unbatched, DRYML can batch the output spec automatically. A non-dropping data batch has dynamic batch-size metadata because its final batch may be shorter than the requested size. TensorFlow and PyTorch selected element calls add and remove a one-item batch axis; selected batched calls do not add a second axis, while ordinary direct calls retain the wrapper's raw model behavior.

## AutoEncoder

`AutoEncoder` composes an encoder model and decoder model.

```python
from dryml.models import AutoEncoder

autoencoder = AutoEncoder(encoder=encoder_model, decoder=decoder_model)
```

Calling the autoencoder applies encoder then decoder. Output-spec inference follows the same composition.

## Training Functions

`TrainFunction` represents training behavior as a DRYML method. Backend-specific training functions implement the details for TensorFlow, PyTorch, sklearn, or other systems.

Training functions should update model state and training metadata while keeping stable construction identity separate from runtime results.

## Experiments

`Experiment` is a serializable object intended to group model, data, training configuration, and results.

A typical experiment graph might include:

- model
- training dataset
- validation dataset
- training function
- metrics
- artifacts

Because this graph is made of DRYML objects, it can be saved, queried, loaded, and reused.

## Backend Wrappers

Backend wrappers adapt external model objects to DRYML semantics.

Examples include:

- TensorFlow wrappers and training functions
- PyTorch wrappers and training functions
- sklearn model wrappers
- XGBoost model wrappers

Backend wrappers should keep external runtime state in object state and keep stable configuration in definitions.

## Sequential Layer Factories

TensorFlow Keras and PyTorch `Sequential` models require explicit layer
factories. Import `F` from `dryml` (or `dryml.core`); it is the same
`FactorySpec` class and records an inert, call-shaped construction recipe. The
backend resolves short layer names only when it constructs the model.

```python
from dryml import F
from dryml.models.tf import Sequential

model = Sequential(layer_defs=[
    F("Flatten"),
    F("Dense", 32, activation="relu"),
    F("Dense", 10),
])
```

The outer `layer_defs` container may be a list or tuple, but every element must
be an `F(...)` or `FactorySpec(...)` value. Bare strings and tuple/list layer
shorthand are rejected. Factory construction preserves supplied positional and
keyword arguments without inspecting target signatures, inserting defaults, or
persisting a backend namespace; target resolution and constructor errors remain
visible when the Sequential model is constructed.

This is deliberately different from Object construction, whose canonical CDef
uses bound constructor parameters and declared defaults. Factory identity is the
supplied call recipe: `F("Dense", 32)` is not normalized into
`F("Dense", units=32)` and does not capture omitted defaults. A later backend
default change can therefore alter construction from the unchanged declaration.
`FactorySpec.coerce(...)` remains an independently invoked conversion utility;
Sequential never performs that conversion implicitly.

For migration only, an old layer declaration such as
`[("Dense", 32, {"activation": "relu"})]` becomes
`[F("Dense", 32, activation="relu")]`. The former is no longer accepted by
Sequential; the outer layer sequence remains unchanged.

## Train State

`TrainState` records coarse training lifecycle state. Use it to distinguish untrained, trained, and related phases where supported by the training API.

## Common Pattern

```python
from dryml.core import Repo

repo = Repo()

# model, dataset, and trainer are DRYML objects.
experiment = Experiment(
    model=model,
    train_data=train_dataset,
    train_fn=train_fn,
    repo=repo,
)

repo.save_object(experiment)
```

Exact constructor signatures vary by model and experiment class. Prefer backend-specific docs and docstrings for detailed parameters.

## Common Pitfalls

- Do not put trained weights in definitions.
- Keep backend handles out of stable identity unless they are intentionally part of configuration.
- Make input/output specs explicit when automatic inference is ambiguous.
- Use contexts when backend execution requires specific resources.

## Related Docs

- [Methods](methods.md)
- [Tensor Specs](tensor_specs.md)
- [Data API](data.md)
- [Contexts](context.md)
- [Artifacts API](artifacts.md)
