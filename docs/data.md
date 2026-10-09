# Data API

Status: draft.

The DRYML Data API provides reusable, repo-backed dataset objects and dataset transformations. Datasets are normal DRYML objects, so they can be saved, queried, composed, and used as part of larger object graphs.

## Dataset Contract

`Dataset` is an abstract iterable dataset type. Every concrete Dataset subclass
must implement `__iter__`; legacy `__len__` remains optional because cardinality
can be unknown. The supported source, mapped, and structural dataset classes
implement iteration and remain constructible.

Important expectations:

- A dataset should be re-iterable.
- `iter(dataset)` should produce a fresh iterator.
- `dataset.spec` describes one yielded element.
- `yield_cardinality()` returns the declared number of yielded values as a
  finite, unknown, or infinite `Cardinality`. It normalizes legacy integer and
  `Cardinality` `__len__` declarations without opening a cursor.
- `example_cardinality()` returns an example total only when declared TensorSpec
  batch metadata proves it. Unbatched values have one example per yield; fixed
  uniform batches multiply yield cardinality; dynamic batches remain unknown.
  A zero-yield dataset has zero examples regardless of batch metadata.
- `examples_in(value)` validates one runtime TensorSpec tree and returns its
  exact positive example count. Batched leaves must expose non-rank-zero,
  nonempty matching leading dimensions and honor fixed batch declarations;
  coherent unbatched trees count as one example.
- Dataset composition propagates an example total only when it can prove one:
  `Batch` retains source examples (and removes a dropped remainder), `Unbatch`
  turns proved batched examples into yields, and `Take`/`Skip` use declared yield
  ranges. `Shuffle` retains only full-membership totals, `Zip` requires aligned
  branch yield and example totals, and `Chain` requires compatible batch
  semantics. `Map` is unknown by default; pass `preserves_examples=True` only
  when every result preserves its input count, which is checked at iteration.
- `peek()` returns one element without permanently consuming the dataset.

These count methods do not open, peek, scan, or otherwise consume a Dataset,
and they import no optional tensor backend. Generic values without a complete
TensorSpec tree retain normal iteration behavior but have unknown metadata
example cardinality and cannot be runtime-counted. Yield positions, cursor
positions, and `Take(source, n)` remain yield-based: `Take` still requests
exactly `n` source yields and reports exhaustion rather than silently clipping
to a short source declaration.

`dryml.artifacts.CachedDataset` implements this same contract after completion.
Its persisted output spec is the codec's actual NumPy-backed `SpecTree`, so
`Map`, `Batch`, `Unbatch`, cursors, and other ordinary Dataset consumers do not
branch on cache type or codec. A new or progress-only cache has no consumable
spec; only a completed StateRef restores an iterable Dataset.

### Cursors and Exact Bounds

`dataset.iterator()` creates an independent closeable `DatasetCursor`. Its
`position` counts consumed yields, `skip(n)` requires a nonnegative exact
integer and either advances exactly `n` values or raises `DatasetExhaustedError`
with requested and actual counts, and `close()` releases its owned iterator.
Dataset objects do not retain a shared cursor. Array and NPY-file sources use
equivalent private indexed advancement, so skipping does not read discarded rows
or files.

`Take(source, n)` requires a nonnegative exact integer and has finite
cardinality `n`. It yields exactly `n` values or raises `DatasetExhaustedError`
after its available prefix; `Take(source, 0)` does not open the source. The
older `Skip(source, n)` remains forgiving when a source ends before its prefix.
For declared seed-aware `GeneratorDataset` sources, `Take` also carries an
optional logical `epoch`; `Repeat(Take(...))` advances selected epochs by default
without changing its strict yield count. `fixed_prefix=True` intentionally reuses
one epoch. Opaque generator factories receive no new replay or seed requirement.
A completed `CachedDataset` instead persists one selected realization: repeating
that cache replays its stored values and does not advance the source's epoch seed.

### Prepared Stream Graphs

`dataset.method_graph()` returns an inert `MethodGraph` view for a Dataset
pipeline. It does not open, peek, select from, or otherwise consume a source.
Call `graph.learn(strategy="local")` before `graph.iterator()`; preparation
records qualified local selections from declared specs and opens no sources.
`graph.iterator()` then returns an independent closeable graph cursor. Optional
positional specs supplied to `learn` are assertions of the graph's declared
source specs, and `output_spec` is an assertion of its declared result spec.
Mismatched assertions, unsupported strategies, unqualified operators, and
Method-selection failures raise before source execution.

The initial prepared subset is `Map`, `Batch`, `Unbatch`, `Zip`, and `Chain`.
It preserves their ordinary order and cardinality rules: Map emits one result
per input, Batch retains short final batches unless `drop_remainder=True`,
Unbatch emits split items in order, Zip stops at the shortest source with its
existing left-to-right positional over-pull, and Chain opens/advances sources in
declaration order. Other Dataset operators keep their eager behavior but reject
graph planning explicitly.

One graph cursor owns all source and selected output iterator resources it
acquires. It closes them once in reverse acquisition order on exhaustion,
explicit close, acquisition/body failure, or a consumer body error. Reusing one
Dataset occurrence in branches opens independent cursors and independent
unknown-spec discovery buffers; it is not teeing, memoization, or deduplication.
`skip()` has the normal exact cursor contract, and reopening a graph creates a
fresh traversal rather than serializing an iterator or generator frame.

Prepared Dataset boundaries may retain one dense NumPy/TensorFlow/Torch/JAX handoff
edge when a downstream Method has no direct compatible implementation. The
adapter is applied while advancing the Dataset cursor, before native model
forward/loss/backward/tape or compiled work begins. Dataset preprocessing is
therefore data-only: it is not an end-to-end autodiff, tracing, or `tf.function`
contract. Batch and Unbatch select the existing declared-backend stack/split
operation once when available; their generic Python fallback remains equivalent
for order, short batches, cardinality, and cursor cleanup. Zip and Chain retain
the Python fallback unless compatible source-native composition is explicitly
provided by all inputs.

### Native Training Inputs

`dataset.prepare()` returns a dependency-light `PreparedDataset` for one native
training invocation. Construction plans the qualified Dataset/Method graph once
without opening a source. Its `execution_level` is `"stream"` when that qualified
plan runs and `"eager"` when an unqualified operator retains ordinary eager
Dataset iteration. Selection failures in qualified operators still fail during
preparation; eager fallback is only for unqualified Dataset operators.

`PreparedDataset.iterator()` opens an independent closeable cursor each time.
`training_batches(preparation)` applies one already selected x/y handoff per
authored batch and returns a closeable cursor. It preserves the x/y tree, dtype,
shape, batch axis, batch order, and short final batch. It checks finite declared
yield counts at exhaustion and raises `DatasetExhaustedError` for a short source.
Closing, conversion failure, source failure, or exhaustion closes the owned
cursor. Reads never advance accepted-update progress, exposure, or checkpoint
state; successful trainer updates remain their sole owners.

Backend plugins deliberately remain separate from the generic seam:

- `dryml.tf.training_data.as_training_dataset(prepared, preparation)` produces
  a native `tf.data.Dataset` without a second batch, shuffle, or repeat. Each
  native traversal reopens a DRYML cursor.
- `dryml.torch.training_data.as_training_dataset(prepared, preparation)` returns
  a native `IterableDataset`. `as_data_loader(dataset)` is available only for
  wrappers that require a loader; it fixes `num_workers=0` and `batch_size=None`
  so it neither duplicates a source nor adds a hidden batch axis. Explicit
  loops may use `iter_training_batches` directly.
- `dryml.jax.training_data.iter_training_batches(prepared, preparation)` returns
  a closeable iterator of JAX-native batches. It does not introduce JAX training
  state or compilation.

Importing `dryml.data.native` or any training-data plugin imports neither its
heavy framework nor the unsupported historical `dryml.data.tf` and
`dryml.data.torch` wrappers. Framework imports occur only when a native adapter
is requested or a retained handoff executes.

## Source Datasets

Common source dataset classes:

- `GeneratorDataset`
- `ArrayDataset`
- `NpyFileDataset`
- `TFDSAdapter`
- `TorchDatasetAdapter`

`TFDSAdapter` loads a selected TFDS split and delivers either native TensorFlow
values or NumPy values. `data_dir` is an optional string local TFDS root; `None`
uses TFDS's normal default. `download` is an exact boolean and defaults to
`True` for compatibility, so ordinary use can cause TFDS download/network and
filesystem preparation side effects. Set `download=False` with a prepared
`data_dir` to require local-only loading: a missing split then fails through
TFDS rather than downloading. `download_config` accepts a TFDS `DownloadConfig`
object and is forwarded only when supplied, so callers can control TFDS
preparation behavior. Adapter construction imports TFDS and propagates its
import, split, local filesystem, and download failures; NumPy delivery avoids
importing DRYML's TensorFlow spec backend.

`GeneratorDataset` accepts an optional explicit `example_count` for opaque
or dynamically batched sources. It also supports a declared `seed_aware=True`
factory protocol: a base `seed` and versioned epoch derivation supply a deterministic
`seed` keyword to `iterator_for_epoch(epoch)`, allowing direct epoch reopening
without traversing earlier epochs.

The historical modules `dryml.data.tf.dataset` and
`dryml.data.torch.dataset` are unsupported legacy APIs. They are not current
exports and are not compatible with this Dataset contract.

Example:

```python
import numpy as np

from dryml.core import TensorSpec
from dryml.data import ArrayDataset

dataset = ArrayDataset(
    np.arange(12, dtype="float32").reshape(3, 4),
    spec=TensorSpec("float32", shape=(4,)),
)
```

## Transforming Data

`Map` applies one Method or a pipeline of Methods to each source element. Method
types and authoring helpers are owned by `dryml.methods`, not `dryml.code`.
`Map` selects one local callable from a complete source spec before iteration;
when only the backend is missing, it may inspect one first value to select it.
That local selection does not change the Method's eager, learning, or cached
state.

When a selected Method explicitly declares `iteration_independent`, a Map cursor
may delegate discarded values to its source cursor without invoking the Method.
Otherwise it transforms the discarded prefix normally. The fast path may omit
data-dependent errors in discarded transformed values, but preserves source
counts, exhaustion, and validation for delivered values.

```python
from dryml.data import Map, Scale

scaled = Map(dataset, Scale(0.5))
```

Common transformation methods:

- `Pipe`: compose methods.
- `Project`: project nested structures.
- `Select`: select values by path.
- `Cast`: change dtype.
- `Flatten`: flatten tensor-like values.
- `Scale`: multiply/shift values.
- `ArgMax`: compute argmax along an axis.

## Numerical Reduction Methods

`Diff`, `Abs`, `Squared`, and `Equal` are reusable native numerical Methods.
`Diff`, `Abs`, and `Squared` promote signed `int8`/`int16`/`int32`/`int64` and
`float32`/`float64` inputs to float64 before arithmetic; they reject boolean
arithmetic, broadcasting, mixed backends, non-finite inputs, unsigned/complex,
object, sparse, and ragged values. `Equal` accepts supported numeric or boolean
equal-shaped values and returns a native boolean tensor.

`ArrayMean(axis=None)` and `ArrayQuantile(q, axis=None)` reduce one bounded
native array. Their axis declarations accept `None`, one integer, or a unique
tuple of integers. Mean accepts booleans and returns float64; quantile rejects
booleans, preserves tuple request order and duplicates, and uses native linear
interpolation. Empty selected populations, invalid axes, non-finite values, and
unsupported dtypes raise rather than changing population semantics.

`MeanInitial`, `MeanUpdate`, and `MeanFinalize` are the reusable declared
sum/count program behind the named mean factory. `ReservoirInitial`,
`ReservoirUpdate`, and `ReservoirQuantile` are the corresponding bounded
Algorithm R program. They keep carry tensors in NumPy, Torch CPU, or TensorFlow
CPU through initialization, transitions, and finalization. They do not collect
the Dataset or use a global random generator. Mean carry is a finite float64
sum with a nonnegative scalar int64 count; transitions reject changed carry
shape/dtype, non-finite intermediate sums, and count wrap before returning a
next carry. Reservoir carry is float64 storage plus scalar int64 population and
draw counters and a two-limb int64 key. Fill consumes no draw; each attempted
post-fill draw, including a rejection, advances the counter once. Invalid
population/draw metadata and int64 wrap fail before arithmetic or sampling.

The evaluation factories in `dryml.metrics` build their source projections from
these Data Methods: `Project(prediction=Pipe(Select(x), model),
target=Select(y))`, followed by an explicit prediction/target handoff to
independent host NumPy storage, then `Diff` and either `Abs` or `Squared` for
regression. Classification factories use the same host boundary and require
caller-supplied label Methods;
`ArgMax` remains an explicit label conversion rather than an implicit classifier
policy. This terminal evaluation policy may copy one concrete accelerator result
to the host; it does not make general mixed-backend Method inputs or accelerator
handoffs valid.

`Fold` retains its Dataset through `Ref[AutoRef]`. The declaration, its CDef, and
an exact completed Fold state retain the selected reference rather than an owned
Dataset payload. The Dataset is materialized only by `Fold.compute()` through its
selected managed Repo; a result-only `StateRef` remains readable after that input
is unavailable, while a later fresh rerun fails normally. During compatible
recovery Fold applies its saved processed-yield count to a new closeable cursor;
it never persists or transfers the prior iterator. Exact skipping detects a source
that now ends before the saved position. Positional continuation assumes a
re-iterable source but does not claim that a stochastic suffix equals the suffix
from the interrupted traversal. Saved EOF progress needs no new source cursor.
CachedDataset applies the same positional meaning to retained cache progress: the
completed prefix remains exact, while a resumed stochastic source may provide a
fresh suffix after the saved yield count.

## Structural Operations

Structural dataset nodes change iteration structure rather than individual values.

- `Batch`
- `Unbatch`
- `Take`
- `Skip`
- `Shuffle`
- `Repeat`

For bounded custom synchronous behavior, subclass `StreamDataset` and declare a
class-level `dryml.methods.StreamNode`. The declaration names ordered
`IteratorPort` inputs/output, pure element-spec and cardinality transforms,
deterministic pull policy, and an exact maximum buffered-item count. Its
implementation receives borrowed input iterators and returns an iterator; a list
returned by an element `Map` Method remains one element and is not implicitly
expanded into stream outputs. Custom declarations are process-local trusted code,
not a generator serializer, async scheduler, key join, whole-stream collector,
or JIT interface.

Example:

```python
from dryml.data import Batch, Take

small_batches = Take(Batch(dataset, batch_size=32), 10)
```

## Combining Datasets

`Zip` combines datasets elementwise. `Chain` concatenates datasets sequentially.

```python
from dryml.data import Zip, Chain

pairs = Zip(features, labels)
combined = Chain(train_a, train_b)
```

## Working With `(x, y)` Data

Utility functions help with common supervised-learning structures:

- `iter_xy(dataset)`
- `collect_xy(dataset)`
- `collate_xy(dataset)`
- `Collect`

`as_supervised(dataset, inputs, targets)` builds a persisted ordinary Dataset
projection yielding `(inputs, targets)`, rather than asking trainers to select
fields. A live Dataset returns a live `Map`. A soft Definition,
ConcreteDefinition, ObjectRef, StateRef, symbolic `Expr` (including `Par`), or explicit `Ref(...)`/`Mat(...)`
assertion returns an inert `Map` Definition without resolving or materializing
the source; an assertion contributes its retained target to that new materializing
Dataset graph. A scalar path selects one branch (and can retain a nested
dictionary); named/nested selections use dictionaries of
`Select.from_path(...)` leaves. Tuple/list selection trees require explicit
`Select.from_path` leaves so path and output-tree intent cannot be guessed.
`input_as_target=True` produces an autoencoder-style pair without copies or
implicit batching.

Dataset placeholders can be supplied before choosing the actual datasets:

```python
from dryml.core import Par
from dryml.data import as_supervised
from dryml.models import Experiment

experiment_template = Experiment.defn(
    model=model_template,
    train_fn=training_template,
    train_data=as_supervised(Par("train_ds"), "cart", input_as_target=True),
    test_data=as_supervised(Par("test_ds"), "cart", input_as_target=True),
)
experiment = experiment_template.sub(train_ds=train_ds, test_ds=test_ds)
```

This authors an inert graph; it does not inspect a placeholder's specification or
iterate data. Binding accepts the same source definitions/references as direct
authoring, including a concrete training definition and saved test `StateRef`.
The bound source must materialize a Dataset at the normal build boundary. Literal
selection errors still fail when `as_supervised` is called. The helper uses core
`authoring_helper` with an explicit dataset-source predicate and a shared live/inert
projection recipe; it does not opt into concrete signature normalization.

Native trainers consume this authored Dataset as-is. Put `Batch`, `Shuffle`,
`Take`, and related controls in the Dataset graph before constructing an
Experiment; trainers do not add hidden batching, selection, or shuffling.

These utilities assume an element structure where `x` and `y` can be selected by path.

## Specs And Data

Dataset specs are important because models and methods use them to infer outputs and verify structure. A dataset yielding `(x, y)` pairs should usually expose a matching spec tree.

```python
from dryml.core import TensorSpec

pair_spec = (
    TensorSpec("float32", shape=(128,)),
    TensorSpec("int64", shape=()),
)
```

## Common Pitfalls

- Dataset objects should be re-iterable unless clearly documented otherwise.
- Keep specs aligned with actual yielded values.
- Avoid embedding large data directly in definitions when a file-backed source is more appropriate.
- Use `Batch` and `Unbatch` consistently with tensor specs.

## Related Docs

- [Methods](methods.md)
- [Tensor Specs](tensor_specs.md)
- [Models API](models.md)
- [Repos and Stores](repos.md)
