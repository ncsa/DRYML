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

The base `Model` is a `Method`, which means it can be used in dataset mapping and method pipelines.

```python
from dryml.data import Map

predictions = Map(dataset, model)
```

The method interface supports output-spec inference. If a model has an explicit `output_spec`, it can propagate batching information from the input spec.

## Output Specs

Models can infer output specs from input specs.

```python
from dryml.core2 import TensorSpec
from dryml.models import Model

model = Model(output_spec=TensorSpec("float32", shape=(10,)))
```

When the input spec is batched and the output spec is unbatched, DRYML can batch the output spec automatically.

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

## Backend Progression

Start with the maintained
[sklearn models tutorial](../examples/notebooks/models_experiments_and_metrics.ipynb),
which uses `RegressionModel`, `BasicTraining`, `Experiment`, and explicit metric
evaluation on small local arrays. The next two tutorials reuse that same
current model/experiment contract for definition variants and bounded local
search.

For backend-specific progression, use the public APIs exported by
[`dryml.models.tf`](../src/dryml/models/tf/__init__.py), including `Model`,
`Sequential`, and `BasicTraining`, or
[`dryml.models.torch`](../src/dryml/models/torch/__init__.py), including
`Model`, `Sequential`, `Optimizer`, and `Training`. Install the matching `tf`
or `torch` extra and follow that backend's runtime requirements. These are
maintained backend paths, not legacy compute-context or `Trainable` APIs.

## Train State

`TrainState` records coarse training lifecycle state. Use it to distinguish untrained, trained, and related phases where supported by the training API.

## Common Pattern

```python
from dryml.core2 import Repo

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
- Declare resource requirements through worlds, then run backend work through an active runtime allocation or dispatch worker.

## Related Docs

- [Tensor Specs](tensor_specs.md)
- [Data API](data.md)
- [Worlds and Runtime](world_runtime.md)
- [Dispatch](dispatch.md)
- [Artifacts API](artifacts.md)
