# Models API

Status: draft.

The Models API provides DRYML object abstractions for model composition, training, evaluation, and backend integration. Models are DRYML objects and methods, so they can participate in object graphs, datasets, repos, and saved workflows.

## Core Types

Important public types:

- `Model`
- `ParameterCounts`
- `MeasurementUnavailableError`
- `AutoEncoder`
- `TrainFunction`
- `TrainState`
- `Experiment`
- `ExperimentData`
- `ExperimentDataError`
- `TrainingObservation`
- `model_parameter_counts`
- `parameter_counts_from_parameters`

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

Prepared TensorFlow and PyTorch models declare their native input and output
backends. A dense Dataset or inference-data boundary can therefore retain one
visible CPU host-copy handoff before invocation instead of relying on an eager
wrapper conversion. This applies only to data/inference values. TensorFlow/Torch
model-component composition never inserts a bridge: mixed-framework
`AutoEncoder` preparation fails before source acquisition. Same-framework
AutoEncoder calls remain native and retain connected gradients. TensorFlow tape
participation cannot generally be detected, so invoking Dataset preprocessing as
a differentiable or compiled body is unsupported.

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

### Measurements

`ParameterCounts(total, trainable)` reports scalar native parameter counts, not
parameter tensors, buffers, or optimizer slots. A native parameter shared by
multiple wrappers is counted once by object identity. A frozen parameter counts
in `total`; it contributes to `trainable` only when at least one participating
model path exposes that same native object as effectively trainable. Counts are
available directly from `dryml.tf.measurements.parameter_counts(native_model)`
and `dryml.torch.measurements.parameter_counts(native_model)`, without a DRYML
runtime or wrapper. `Model.parameter_counts()` uses the existing DRYML graph
traversal for composites and applies the same native identity deduplication.

`model_parameter_counts(model, *, repo=None)` measures a DRYML model or
single-backend composite through its declared native parameter objects. It returns
`ParameterCounts` and raises `MeasurementUnavailableError` for unbuilt/lazy or
unknown-shape parameters, and `TypeError` for unsupported or mixed backends. It
does not call, build, compile, save, or otherwise mutate a model; an optional
`Repo` is used only to traverse a composite graph.

`parameter_counts_from_parameters(parameters, trainable_parameters)` returns
identity-deduplicated `ParameterCounts` from native parameter iterables. It raises
`MeasurementUnavailableError` when a shape cannot be read without execution and
does not initialize a backend or mutate its inputs. A parameter present only on a
trainable path is included in both totals.

`MeasurementUnavailableError` means a native model is unbuilt, lazy, or has an
unknown parameter shape. It is not equivalent to a valid built zero-parameter
model, which returns `ParameterCounts(0, 0)`. Measurements read metadata only:
they do not invoke a forward pass, build/compile a model, start a runtime, or
save state.

`dryml.models.measurements.dataset_size(dataset)` returns declared effective
Dataset cardinality after selection/subsetting and unbatching but before trainer
batching or epoch repetition. It returns `Cardinality.finite(n)`,
`Cardinality.UNKNOWN`, or `Cardinality.INFINITE` without opening a cursor;
unknown and infinite inputs are never scanned to manufacture a count. Unbatching
an opaque natively batched source is unknown unless that source explicitly
declares its example cardinality; its batch count is never reported as examples.

### Retained Accounting

`TrainState` retains `examples_seen`, a weighted loss numerator/denominator,
the next unprocessed batch position, an invocation target epoch, a bounded
pending epoch-postlude fact, and an immutable pending safe-point observation
alongside model/optimizer progress. Supplied
TensorFlow and Torch trainers advance these facts only after a successful
optimizer update, so failed updates and evaluation do not add exposure. Repeated
epochs add exposure rather than changing effective Dataset size. The reported
observation loss is weighted by actual batch examples, including short final
batches. A restored unfinished invocation completes its retained target rather
than adding the configured epoch count again; a later call after completion is a
fresh invocation.

Its named persistence state rejects unknown fields and validates finite loss
facts, exact nonnegative counters, lifecycle enums, pending-postlude coherence,
and pending-observation type before installing restored progress. Historical
three-slot `(epoch, step, phase)` payloads remain supported with zero/`None`
defaults for U8 fields.

TensorFlow `fit`, explicit TensorFlow loops, and Torch loops validate all DRYML
callbacks before training work and invoke them only after this retained state is
truthful. Keras uses DRYML-owned train-step facts for the actual completed-update
mean loss and example count, before native user callbacks. Explicit TF and Torch
accept only mean-reduced supplied losses; sum, unreduced, and indeterminate loss
contracts fail before model or optimizer mutation. Restoration reopens
deterministic prepared data and skips `next_batch` without reapplying completed
updates. An update completing an epoch is normalized to the next epoch/batch-zero
position before its callbacks. Explicit TensorFlow and Torch retain validation
results before progress, so a progress failure does not repeat validation. Keras
safe-point recovery is deliberately narrower: a DRYML safe-point callback cannot
be combined with native Keras callbacks or validation, because their generic
event/log replay cannot be truthful. Ordinary uninterrupted Keras callback and
validation behavior remains native. Keras also requires exactly one optimizer
update per batch callback (`steps_per_execution=1`, no gradient accumulation)
and accounts for static regularizers and dynamic `add_loss` objectives in the
exact differentiated scalar before advancing retained state. Setup
failures that retain no update clear their new target, while Keras EarlyStopping
completes its accepted shortened target. sklearn's one-shot trainer accepts the
common callback keyword but rejects intermediate safe points before fitting
because it has no per-update boundary.

After a mid-epoch restore, native step/window observations remain available, but
the trainers do not present the resumed suffix's loss or metric aggregate as a
full-epoch metric.

Training prepares declared x/y Dataset specs once in the trainer's inspectable
`method_graph()`. Its producer/consumer specs and direct dense conversion edges
are retained as Method-owned graph facts; each yielded value only executes those
selected edges before native differentiation. Validation uses the same prepared
edge contract. No per-yield route selection, implicit
cross-backend generator conversion, or Dataset execution inside a tape/backward
body is supported.

Keras retained loss is the scalar objective differentiated for each accepted
completed update, including built-in regularizers and dynamic `add_loss`
contributions reported by nested layers. To keep its example-weighted aggregation
truthful, supplied Keras `class_weight`, `sample_weight`, and `loss_weights` are
rejected rather than reported with a different denominator. Mean-reduced
unweighted losses are supported. Unknown-but-finite Keras streams may complete
normally without DRYML safe-point callbacks; callbacks require a declared finite
deterministic batch count, and infinite streams require an explicit finite bound
before training starts.

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

`Experiment.train(managed=...)` is a resumable managed operation whose successful
result is the exact terminal `StateRef`, not the trainer's backend return. It adds
one invocation-local observer ahead of caller `ManagedConfig.callbacks` without
changing the caller's configuration or callback list. The observer uses this fixed
order for every training safe point and for the terminal state:

1. The managed lifecycle publishes and associates the exact Experiment checkpoint.
2. Experiment writes or reopens its retry-stable pending `ExperimentData` row.
3. It binds every Artifact recipe's `this` to that checkpoint, then evaluates or
   recovers Artifacts in configured order, publishing each completed result.
4. It publishes the row's `completed` status. An Artifact failure stops later
   Artifacts and attempts to record `failed` status with completed predecessors.
5. Caller observers run in their original order; a requested interruption is then
   decided by managed lifecycle handling.

`this.model` and `this.test_data` therefore come from the exact saved Experiment
checkpoint. `test_data` must be an exact saved Dataset reference; a missing value
or invalid recipe binding fails rather than falling back to training or validation
data. Empty Artifact configuration still creates a completed facts-only row.

Before changing training phase, model, progress, or checkpoint state,
`Experiment.train` preflights every inert Artifact recipe. Active roots may only
be `this`; the framework validates the symbolic checkpoint binding, concrete
Artifact definition, and a resumable managed `compute` operation with no required
ordinary arguments and a final `StateRef` receipt. This validation neither builds
an Artifact nor loads or computes model/Dataset payloads. Fully resolved recipes
with no `this` root are retained unchanged.

Pending observations retain their occurrence key, UTC-millisecond timestamp,
trajectory predecessor, and terminal marker in `TrainState` before checkpoint
publication. Retrying the same checkpoint updates that row instead of duplicating
it; revisiting the same StateRef through a later trajectory creates a distinct row.
History remains readable if a referenced checkpoint payload later becomes
unavailable, but explicitly loading that checkpoint still fails.

For completed `Value` Artifacts, history scalar columns retain finite Python
scalars and lossless zero-dimensional native values. Conversion is bounded and
lazy (`item()` and, where needed, zero-dimensional `numpy()`); non-scalar values
are omitted from scalar columns while their Artifact StateRef remains available.

## Experiment History

`ExperimentData` is an ordinary persisted Object containing checkpoint history for
one default-policy projected `Experiment` `ObjectRef`. Its constructor accepts only
that non-materializing `Ref[ObjectRef]` subject; it does not own an Experiment,
model, or Dataset payload. Exact checkpoint and Artifact `StateRef` values remain
in rows, so reading history never restores those referenced payloads.
The closed reference codec also retains supported frozen dense constructor arrays
losslessly, without opening the referenced checkpoint payload.

`ExperimentData.find(experiment, repo=..., store=...)` returns a fresh current
history object or `None` only when the subject has no history identity. Corrupt,
ambiguous, incomplete, or alias-less published authority raises an authority/load
error. `get_or_create(...)` selects one writable Store, creates the first empty
snapshot through the Store-local `experiment_data_current_v1` CAS alias, and may
recover only one valid empty alias-less initial snapshot. It never chooses an
arbitrary matching query result.

`add_row(...)` validates an exact checkpoint and optional predecessor projection,
then returns an opaque row key. Supplying a callback occurrence key makes identical
retries idempotent; an omitted key generates a UUID and has no retry-identity
guarantee. `update_row(...)` merges ordered expected Artifact inputs/results and
supported scalar cells. It rejects conflicting immutable facts, replacement of a
completed result, reserved scalar names, and completed status without every expected
result. Pending or failed rows may become completed only after every expected result
is present.

Call `publish(repo=..., store=...)` after local row changes. It fresh-loads the
current history, reapplies idempotent changes, saves an immutable snapshot, and
CAS-advances the selected Store alias, retrying stale writers at most eight times.
Failed publication preserves local queued changes and can leave a safe immutable
orphan snapshot. `data` lazily imports pandas and returns a recursively detached
DataFrame: missing cells are `pd.NA`, explicit nulls are `None`, and arbitrarily
large integer facts remain integers. The authoritative payload is closed
`experiment_data.json` format `dryml-experiment-data` version 1, not pandas pickle
or pandas JSON inference.

Use `ExperimentData.scalar_column(artifact, field=None)` when turning a configured
Artifact name and optional top-level scalar field into a history column. It escapes
backslashes and dots in components and rejects built-in history-field collisions.

`ExperimentDataError` is raised before local mutation when an association,
reference, row fact, scalar, status transition, or closed v1 codec payload violates
the history contract. It has no Store side effect. `TrainingObservation` is the
immutable typed safe-point value retained inside `TrainState`; its
`training_loss` property returns the window's weighted mean or `None` when no
successful update contributed. Constructing either type does not invoke a trainer,
load a checkpoint, or import an optional ML backend.

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
    checkpoint_every_steps=32,
    repo=repo,
)

repo.save_object(experiment)
```

`Experiment(..., checkpoint_every_steps=...)` accepts a positive exact integer
for intermediate optimizer-step checkpoints, or `None` (the default) for
terminal-only checkpointing. The terminal checkpoint and its Artifact evaluation
always occur. Invalid cadence values fail during construction before model or
training state mutation. Exact constructor signatures vary by model and
experiment class. Prefer backend-specific docs and docstrings for detailed
parameters.

## Evaluation Results

Metric factories return inert Artifact Folds rather than training-history scalars.
Their Dataset and Model inputs are retained as non-materializing references, so a
direct managed call or supported same-host worker materializes them only for
compute. The completed metric's exact StateRef restores its lightweight result
without reloading that model or Dataset; recomputation still requires the selected
input authority. Factories neither start a backend, download data, nor provision a
Ray target.

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
