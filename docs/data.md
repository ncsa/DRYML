# Data API

Status: draft.

The DRYML Data API provides reusable, repo-backed dataset objects and dataset transformations. Datasets are normal DRYML objects, so they can be saved, queried, composed, and used as part of larger object graphs.

## Dataset Contract

`Dataset` is an abstract iterable dataset type. Every concrete Dataset subclass
must implement `__iter__`; `__len__` remains optional because cardinality can be
unknown. The supported source, mapped, and structural dataset classes implement
iteration and remain constructible.

Important expectations:

- A dataset should be re-iterable.
- `iter(dataset)` should produce a fresh iterator.
- `dataset.spec` describes one yielded element.
- `len(dataset)` should return cardinality when known.
- `peek()` returns one element without permanently consuming the dataset.

## Source Datasets

Common source dataset classes:

- `GeneratorDataset`
- `ArrayDataset`
- `NpyFileDataset`
- `TFDSAdapter`
- `TorchDatasetAdapter`

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
target=Select(y))`, followed by `Diff` and either `Abs` or `Squared` for
regression. Classification factories require caller-supplied label Methods;
`ArgMax` remains an explicit conversion rather than an implicit classifier
policy.

`Fold` retains its Dataset through `Ref[AutoRef]`. The declaration, its CDef, and
an exact completed Fold state retain the selected reference rather than an owned
Dataset payload. The Dataset is materialized only by `Fold.compute()` through its
selected managed Repo; a result-only `StateRef` remains readable after that input
is unavailable, while a later compute/rerun fails normally.

## Structural Operations

Structural dataset nodes change iteration structure rather than individual values.

- `Batch`
- `Unbatch`
- `Take`
- `Skip`
- `Shuffle`
- `Repeat`

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
