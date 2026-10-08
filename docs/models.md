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

Backend packages add specialized wrappers for TensorFlow, PyTorch, sklearn, XGBoost, and other frameworks. The JAX/Flax NNX/Optax APIs described below are **experimental** and may change based on use.

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
Dataset example cardinality from the final Dataset persisted by an Experiment.
It returns `Cardinality.finite(n)`,
`Cardinality.UNKNOWN`, or `Cardinality.INFINITE` without opening a cursor;
unknown and infinite inputs are never scanned to manufacture a count. It delegates
to `Dataset.example_cardinality()`, so an opaque native batch count is never
relabeled as an example count.

### Retained Accounting

`TrainState` retains `examples_seen`, a weighted observation-loss window, a
separate recovery-stable weighted epoch-loss accumulator,
the next unprocessed batch position, an invocation target epoch, a bounded
pending epoch-postlude fact with finite completed-epoch metrics, and an immutable pending safe-point observation
alongside model/optimizer progress. Supplied
TensorFlow, Torch, and experimental JAX trainers advance these facts only after a successful
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
defaults for later lifecycle fields.

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
safe-point recovery is deliberately narrow: arbitrary native callbacks cannot be
combined with a DRYML safe-point callback because their generic event/log replay
cannot be truthful. The dedicated saved early-stopping adapter below supports its
own validation postlude without broadening that callback guarantee. Ordinary
uninterrupted Keras callback and validation behavior remains native. Keras also requires exactly one optimizer
update per batch callback (`steps_per_execution=1`, no gradient accumulation)
and accounts for static regularizers and dynamic `add_loss` objectives in the
exact differentiated scalar before advancing retained state. Setup
failures that retain no update clear their new target, while supported saved
early-stopping trainers complete an accepted shortened target. sklearn's one-shot trainer accepts the
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

TensorFlow `BasicTraining` supplies Keras a native `tf.data.Dataset` built from
that once-prepared Dataset plan. It never applies another batch or shuffle and
only repeats when Keras requires a known finite `steps_per_epoch`; exact source
exhaustion remains visible rather than being masked by repetition. Explicit
TensorFlow and Torch loops consume closeable native prepared-batch cursors, and
Torch exposes an `IterableDataset`/single-process DataLoader bridge for native
wrappers that require one. In all cases adapter reads, including native
read-ahead, do not advance retained progress: post-update trainer hooks own that
state.

Experimental JAX training follows the same retained preparation and cardinality
authority. It consumes the authored batches through a closeable JAX iterator;
its JITted transition does not own Dataset selection, batching, or progress.

Keras retained loss is the scalar objective differentiated for each accepted
completed update, including built-in regularizers and dynamic `add_loss`
contributions reported by nested layers. To keep its example-weighted aggregation
truthful, supplied Keras `class_weight`, `sample_weight`, and `loss_weights` are
rejected rather than reported with a different denominator. Mean-reduced
unweighted losses are supported. Unknown-but-finite Keras streams may complete
normally without DRYML safe-point callbacks; callbacks require a declared finite
deterministic batch count, and infinite streams require an explicit finite bound
before training starts.

### Managed Early Stopping

Keras `BasicEarlyStoppingTraining` and Torch/JAX `EarlyStoppingTraining` are saved
specialized training behaviors, not invocation telemetry callbacks. All three
accept `monitor`, nonnegative exact-integer `patience`, `mode="min"` or
`mode="max"`, finite nonnegative `min_delta`, and `restore_best_weights`.
`min_delta` is an absolute threshold and equality is not improvement. The first
completed epoch establishes the best value; each later non-improving completed
epoch increments `wait`, and training stops when `wait > patience`. Thus
`patience=0` stops after the first completed non-improving epoch.

The monitor is read only from a truthful completed-epoch postlude. The supported
training monitor is `loss`; validation monitors use a nonempty `val_` prefix,
including `val_loss`, and require a validation Dataset.
A validation monitor without validation data, a missing completed metric, a
nonfinite value, or invalid configuration fails explicitly rather than silently
shortening training. Arbitrary `validation_freq` scheduling remains deferred and
is not claimed by this integration.

The TrainFunction checkpoint retains the best metric, wait count, best completed
epoch, last processed postlude, and accepted shortened target. When restoration is
enabled, it also retains a bounded best Model snapshot. A restored pending
validation/decision postlude reuses its completed
metrics and processes that epoch idempotently, without repeating an optimizer
update or incrementing `wait` twice. A genuinely new invocation resets these
invocation facts.

With `restore_best_weights=False`, the stopping epoch's Model and Optimizer state
remain installed. With it enabled, DRYML restores only Model parameters and
Model-owned mutable weights/buffers from the best completed epoch, exactly once.
Optimizer slots and progression remain at the stopping epoch; JAX Model RNG also
remains at the stopping epoch. This is intentionally a mixed-point graph, not a
full training rewind. `Experiment.train` publishes a new terminal checkpoint
after optional restoration and returns that exact StateRef; managed completion
and history associate the same reference. Publication, association, or history
failure propagates and does not report a terminal receipt.

Keras uses a DRYML-owned epoch adapter for this behavior, so managed safe points
and its supported validation postlude no longer reject
`BasicEarlyStoppingTraining`. This does not make arbitrary saved native Keras
callbacks replayable. Invocation telemetry remains a separate concern.

### Invocation Telemetry

`Experiment.train(callbacks=None, observer_strict=False, managed=None)` accepts a
bounded collection of reporting-only observers for that invocation. This is a
separate lane from `ManagedConfig.callbacks`: managed callbacks run strictly after
a durable checkpoint has been published and associated, while training telemetry
observes native training events and is not a checkpoint, Artifact, history row, or
completion authority.

Keras `BasicTraining` and `BasicEarlyStoppingTraining` accept local
`tf.keras.callbacks.Callback` instances and deliver the ordinary native callback
methods with Keras's native log mappings. DRYML accounting and saved behavior hooks
run before invocation telemetry at their truthful native boundaries. Native
callbacks saved on a Keras trainer remain behavior-capable configuration and still
have the existing managed-replay restrictions; they are not silently reclassified
as recoverable telemetry. Callers must supply reporting-only invocation callbacks.
DRYML rejects known built-in behavior controls, including Keras early stopping,
learning-rate scheduling, termination, checkpoint, backup, and EMA-swap callbacks;
`BasicEarlyStoppingTraining` remains the supported saved early-stop lane. This
closed classification is not a semantic sandbox and cannot prove that an arbitrary
custom callback is reporting-only.

Torch and experimental JAX training use a smaller host-side contract because their
DRYML-owned loops have no universal native callback protocol. Each observer is a
callable receiving one mapping. `event` is `train_batch_end` or `epoch_end`;
accepted-update mappings include zero-based `epoch` and `batch`, retained `step`
and `examples_seen`, and host scalar `loss`/available metrics. Epoch mappings carry
the completed epoch position, metrics, and `stopped` decision. Delivery occurs
outside backward/JIT and only after the accepted update, coordinated owner state,
DRYML safe-point hooks, and progress reporting are truthful. Pending-postlude
recovery does not replay a missed external event.

Local calls may supply live backend-compatible instances. Core Execute calls must
instead supply only `F(...)`/`FactorySpec(...)` entries; the admitted worker builds
fresh observer instances from that configuration. A live callback/client is
rejected by Execute capture before worker training. Factory configuration is
transported for the invocation only and is never inserted into the Experiment or
TrainFunction definition, checkpoint, early-stopping state, or managed ordinary-
argument digest. Changing or omitting observers, or changing `observer_strict`, on
a compatible unfinished invocation therefore retains the same managed attempt and
accepted update position. Factory values should identify how the worker constructs
the observer; credentials and live service clients belong in its selected runtime,
not in the configuration.

An ordinary observer `Exception` emits one bounded `RuntimeWarning` per observer
and training continues by default. `observer_strict=True` propagates the original
failure after any already accepted update and checkpoint remain authoritative; it
does not report terminal completion or roll work back. Interruption and cancellation
are never downgraded to telemetry warnings. At most 64 observers are accepted, no
background delivery queue is created, and an observer's optional `close()` is
called once in reverse construction order after normal completion, managed early
stop, failure, or interruption. Cleanup never masks an already escaping workload
failure. External events remain best-effort and nonauthoritative: worker loss can
drop or duplicate them, and DRYML does not inspect a callback to prove it is
reporting-only. Mutation, early stopping, scheduling, or other training behavior
must use a saved behavior integration instead.

### Dataset-Owned Training Input

Supplied trainers consume one canonical Dataset yielding `(inputs, targets)`.
Use `dryml.data.as_supervised(...)` to select source fields, and apply `Take`,
`Shuffle`, `Batch`, `Repeat`, and related operators to that Dataset before
constructing the Experiment. TensorFlow/Keras, explicit TensorFlow, and Torch
require both branches to be explicitly batched; use `Batch(dataset, 1)` when a
singleton update is intended. They preserve authored selection, order, and batch
boundaries and validate the actual examples in every accepted batch. sklearn
materializes either canonical examples or authored batches once for `fit`, records
one successful-fit transition, and exposes no optimizer safe points. Its retained
exposure is the number of submitted examples (including the actual size of each
authored batch), not the number of Dataset yields or optimizer updates.

Trainer constructors no longer accept `batch_size`, `num_examples`, `shuffle`,
`shuffle_seed`, `shuffle_buffer_size`, `x_path`, or `y_path`. Saved definitions
containing these retired fields fail during definition projection before source,
model, optimizer, or Store mutation. Rebuild the trainer and move those controls
to its Dataset; there is no legacy migration mode.

The retained `training_preparation` exposes the canonical input/target consumer
specs and selected handoff edges for TensorFlow, Torch, and experimental JAX.
This is inspection evidence only; accepted updates and checkpoints remain owned
by `TrainState` and the managed Experiment lifecycle.

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
checkpoint. `test_data` is a materializing Dataset slot: it accepts deterministic
stateless definitions as well as saved reference authority, and Artifact recipes
receive its exact checkpoint projection. A missing value or invalid recipe binding
fails rather than falling back to training or validation data. Empty Artifact
configuration still creates a completed facts-only row.

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
Embedded references use core's closed `dryml-reference-json` codec, which retains
supported canonical Definition values and frozen dense constructor arrays
losslessly without opening the referenced checkpoint payload.

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

### Experimental JAX Models

`dryml.models.jax.Model` is an **experimental** functional wrapper. Its
constructor records separate explicit `F(...)`/`FactorySpec` `init_fn` and
`apply_fn`, authored initializer arguments/keywords, an exact integer `seed`,
and an explicit `output_spec`; constructor spelling may change in a future
release. After JAX runtime admission, the initializer receives a distinct
initialization key followed by the authored arguments and returns exactly
`(parameters, mutable_state)`. The retained model RNG is split from that key.
The apply function receives `(parameters, mutable_state, rng_state, input_tree,
training_bool)` and returns exactly `(predictions, candidate_mutable_state,
candidate_next_rng)`. It does not update parameters or optimizer slots.

Public raw, selected-batched, selected-element, `Map`, and same-backend
`AutoEncoder` calls return only predictions. They evaluate a state snapshot and
discard candidate mutable/RNG changes, including after a failure or retry.
DRYML never runs a fabricated forward call to infer the explicit output
specification. Functional state supports plain dict/list/tuple trees of
`jax.Array` leaves; malformed candidate topology, dtype, shape, or typed-key
continuations fail before installation. `trainable_mask` is retained definition
configuration and is reapplied to restored current parameters. Training preserves
every false-mask parameter leaf exactly after the Optax candidate, including a
parameter-dependent transformation such as decoupled weight decay. Functional
models with shared parameter leaves are supported for inference, measurement, and
persistence, but are rejected before optimizer binding because this experimental
trainer does not claim shared-gradient semantics. Parameter measurement excludes
mutable state, RNG, and optimizer slots.

`dryml.models.jax.Optimizer` is also **experimental**. Its constructor records
only an Optax `F(...)` recipe and does not import Optax, create a transformation,
or allocate slots. The training-facing `bind(model)` seam reconstructs the recipe and
initializes slots on first training use against current model parameters. It
retains factory identity and parameter-template evidence, rejects incompatible
later binding before update, and restores slots only after the matching model
template is bound. Live transformations are never persisted. Inference-only
Model construction and loading therefore do not require Optax. Experimental
`TrainFunction` owns only behavior-continuation state; concrete `Training`
supplies the loop. `pure_training_transition(...)` returns candidates without
installing them so a future loop can coordinate owners at its eager commit
boundary.

`dryml.models.jax.NNXModel` (also `FlaxModel`) is the first-class experimental
Flax NNX adapter sharing that candidate seam. Its explicit module factory and
authored dimensions/configuration are rebuilt during load, must not provide the
reserved `rngs` keyword, and receive wrapper-owned `nnx.Rngs` only after runtime
admission. It partitions `Param` variables from remaining mutable/module-RNG
state, not a live NNX GraphDef or device object. The separate external model RNG
is split away from construction and remains distinct from module RNG streams.
Public prediction clones module state and discards candidate BatchNorm, dropout,
and external-RNG changes rather than mutating authoritative state.

`dryml.models.jax.Training` is the matching **experimental** Dataset-owned
trainer for functional `Model` and `NNXModel`. Construct it with an experimental
`Optimizer`, a definition-compatible scalar mean-loss callable (or an `F(...)`
factory returning one), and a nonnegative epoch count. It accepts no selection,
batching, shuffle, path, or trainer-owned data controls: its Experiment must
provide canonical explicitly batched `(inputs, targets)` Datasets. The trainer
plans its JAX handoffs once, then consumes the closeable JAX-native prepared
iterator without applying another batch or shuffle. Finite sources retain short
final batches and account their actual example count; infinite sources require an
authored finite bound. Unknown finite sources are supported only without DRYML
safe-point callbacks.

Zero requested epochs validate the Experiment's declared Dataset/callback inputs
and complete the retained invocation lifecycle without preparing or opening a
source, applying a model, importing/binding Optax, allocating slots, or updating
owners. Prepared `Batch`, `Map`, and `Unbatch` pipelines forward their logical
epoch to an enclosed seed-aware `Take(...)` boundary; `Take` retains strict
exhaustion, fixed-prefix, and zero-acquisition behavior. Ordinary non-epoch-aware
Dataset pipelines retain their normal fresh traversal per epoch.

Training computes a pure JITted parameter-only gradient/Optax candidate, waits
for its arrays, validates model parameters, mutable state, RNG, and optimizer
slots, then eagerly installs all owner state and TrainState accounting together.
Callbacks and managed checkpoints run only after that accepted transition, so a
callback/publication failure retains the truthful update. An interruption during
the bounded transition repairs the prior Model, Optimizer, TrainFunction, and
progress state before it propagates. Resume reopens the same logical Dataset epoch
and skips only accepted yielded batches; a seed-aware `GeneratorDataset` under
`Take` therefore retains its selected epoch seed while resuming. Validation is
snapshot-only: candidate mutable state and RNG are always discarded. JAX Training
does not yet support general trainer metrics or arbitrary native callbacks. Its
invocation telemetry is limited to the host-side mapping contract above. Its
saved `EarlyStoppingTraining` specialization supports only completed
training/validation loss facts under the managed contract above; no broader
callback or metric API is implied.

Every JAX owner writes a versioned host-array envelope and validates its complete
runtime topology, path structure, shared-leaf topology, dtype, shape, typed-key
implementation, and factory identity before installation. Parameters alone are
counted by `Model.parameter_counts()` and `dryml.jax.measurements`. A failed
graph restore is invalidated by the normal Store restore boundary and must be
loaded again as a fresh exact graph.

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
