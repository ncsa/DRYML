# Models API

Status: draft.

The Models API provides DRYML object abstractions for model composition, training, evaluation, and backend integration. Models are DRYML objects and methods, so they can participate in object graphs, datasets, repos, and saved workflows.

## Core Types

Important public types:

- `Model`
- `AutoEncoder`
- `TrainFunction`
- `TrainCapability`
- `TrainResumeMode`
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

`TrainFunction` represents training behavior as a DRYML method. Backend-specific training functions implement the details for TensorFlow, PyTorch, sklearn, or other systems. Its definition also supplies the deterministic output declaration used by `Experiment.train`.

Training functions are non-resumable by default. A trainer that can checkpoint every mutable component may advertise `TrainCapability.exact(...)` and call `TrainFunction.checkpoint(...)` at managed safe points. Unsupported or partially checkpointed pipelines must remain non-resumable and require an explicit rerun after interruption.

TensorFlow `BasicTraining` provides exact managed resume at completed epoch
boundaries. Its checkpoint contains Keras model variables, optimizer slots and
iterations, DRYML epoch/step progress, the target epoch, trainer/model identity,
and the TensorFlow version. A resumed invocation rebuilds fresh model and
optimizer objects, restores that state, and continues at the next epoch against
the originally pinned cache record. Managed callbacks are invocation-scoped;
Keras callbacks remain part of the trainer definition. The complete Experiment
must configure an optimizer on the trainer, in `compile_kwargs`, or in the
Experiment capabilities before the managed runtime advertises exact resume.

The exact TensorFlow capability is deliberately conservative. DRYML shuffle,
trainer-defined Keras callbacks (including `BasicEarlyStoppingTraining`), custom
positional `fit` arguments, and custom `fit_kwargs` disable resume because their
within-epoch cursor, RNG, callback, worker, or prefetch state is not included in
the epoch checkpoint. Such configurations still retain direct Keras behavior,
but interrupted managed work requires an explicit rerun. Managed graceful stop
is supported for an exact pipeline and completes the epoch prefix at the next
safe boundary.

PyTorch `Training` provides the same exact managed guarantee at completed epoch
boundaries for its built-in single-process, unshuffled loop. A checkpoint
contains the complete DRYML model state, optimizer state, DRYML epoch/step
progress, target epoch, trainer/model identity, Torch version, and Python,
NumPy, Torch CPU, and applicable CUDA RNG state. Resume hydrates a fresh model,
rebinds a fresh optimizer to that model, restores all checkpointed state, and
continues against the originally pinned cache record. Managed graceful stop is
supported at an epoch boundary. Shuffle, trainer metrics, explicit stateful loss
objects, custom training loops, Experiment-supplied optimizer/loss/metric state,
and multi-rank managed publication are not exact-capable. DRYML does not expose
worker/prefetch stages in this loop; a custom loop adding them remains
non-resumable until it declares and implements their complete state contract.

sklearn `BasicTraining` treats the estimator's blocking `fit()` call as opaque.
It supports managed completion and completed-result reuse, but interruption has
no checkpoint and a later normal call raises until the caller requests an
explicit rerun. A future incremental trainer may advertise its own exact
capability only when it can restore the estimator and every input cursor.

## Experiments

`Experiment` is a logical recipe containing non-materializing model, trainer, train, validation, and test-data edges. Managed training requires completed `CachedDataset` inputs; it resolves their exact active records once and never computes their sources. Completed NumPy-sequence and Parquet records are iterated directly without implicit adaptation.

A typical experiment graph might include:

- model
- training dataset
- validation dataset
- training function
- metrics
- artifacts

Because this graph is made of DRYML objects, it can be saved, queried, loaded, and reused. Trained weights are immutable `StoredStateRecord` products owned and selected by the Experiment's `train` realization, not mutable Experiment state. `experiment.trained_model(store=...)` verifies that product and hydrates a fresh uncached model instance.

An Experiment normally snapshots the selected Store's ordinary model directory before training. The snapshot manifest participates in record identity, so changed initial bytes make an existing train result stale. Fine-tuning selects prior state explicitly without embedding a Store-owned record in identity:

```python
fine_tune = Experiment(
    model,
    train_fn,
    train_data=completed_cache,
    model_state=prior_experiment.train.result,
)
fine_tune.train(store=store)
model_for_inference = fine_tune.trained_model(store=store)
```

Pending exact-capability work retains its original cache and model-state bindings across resume even if a producer's active realization changes. A normal call reuses a completed train result only while current logical inputs and the current ordinary-state snapshot still match; otherwise use `experiment.train.rerun(...)` explicitly.

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

`TrainState` remains a concrete compatibility view for direct in-memory trainers and is included in generic trainer checkpoints. Managed lifecycle status and result authority come from `experiment.train.status(...)`, `results(...)`, and realization history.

## Common Pattern

```python
# model and trainer are DRYML objects; train_cache is a completed CachedDataset.
experiment = Experiment(
    model=model,
    train_data=train_cache,
    train_fn=train_fn,
)
experiment.train(store=store)
trained = experiment.trained_model(store=store)
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
