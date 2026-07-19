# Data API

Status: draft.

The DRYML Data API provides reusable, repo-backed dataset objects and dataset transformations. Datasets are normal DRYML objects, so they can be saved, queried, composed, and used as part of larger object graphs.

## Dataset Contract

`Dataset` is the base iterable dataset type.

Important expectations:

- A dataset should be re-iterable.
- `iter(dataset)` should produce a fresh iterator.
- `dataset.spec` describes one yielded element.
- `len(dataset)` should return cardinality when known.
- `peek()` returns one element without permanently consuming the dataset.

`CachedDataset` follows the same contract after a completed compatible
realization is active. Before that point, iteration and `peek()` raise rather
than implicitly computing dependencies. Use `cached.view(repo=...)` or
`cached.view(store=...)` when ambient repository selection would be ambiguous.
Use `cached.tensorflow_view(...)` or `cached.torch_view(...)` for explicit lazy
framework tensor views over a completed NumPy or Parquet representation. View
construction and `support()` do not import the optional framework; iteration
does.

## Source Datasets

Common source dataset classes:

- `GeneratorDataset`
- `ArrayDataset`
- `NpyFileDataset`
- `TFDSAdapter`
- `TorchDatasetAdapter`

Example:

```python
import numpy as np

from dryml.core2 import TensorSpec
from dryml.data import ArrayDataset

dataset = ArrayDataset(
    np.arange(12, dtype="float32").reshape(3, 4),
    spec=TensorSpec("float32", shape=(4,)),
)
```

## Transforming Data

`Map` applies one method or a pipeline of methods to each source element.

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
from dryml.core2 import TensorSpec

pair_spec = (
    TensorSpec("float32", shape=(128,)),
    TensorSpec("int64", shape=()),
)
```

CachedDataset copies this lightweight element spec, source cardinality, and
source-order declaration into its own definition while retaining the source as
a non-materializing definition reference. Empty sources therefore require
declared metadata but can produce a valid empty cache.

## Resume Capability

`dataset_resume_capability(dataset_or_definition)` inspects a pipeline without
constructing it and reports `exact`, `replay`, or `none`. Exact continuation
requires a durable source cursor and a checkpoint/restore contract for every
stateful stage. `open_resumable_dataset(...)` is the corresponding exact cursor
protocol used during cache computation. Replay is diagnostic only and is never
silently substituted for exact resume.

## Common Pitfalls

- Dataset objects should be re-iterable unless clearly documented otherwise.
- Keep specs aligned with actual yielded values.
- Avoid embedding large data directly in definitions when a file-backed source is more appropriate.
- Use `Batch` and `Unbatch` consistently with tensor specs.

## Related Docs

- [Tensor Specs](tensor_specs.md)
- [Models API](models.md)
- [Repos and Stores](repos.md)
