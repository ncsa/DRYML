# Ragged Tensors

DRYML supports packed NumPy ragged values and native TensorFlow RaggedTensors
through TensorSpec, selected/known-spec Methods, composed Method IR, and ordinary
or prepared Dataset pipelines. Ragged layout is an explicit storage contract:
Dynamic dimensions on a dense tensor still describe a rectangular value.

## Batch Variable-Length Samples

```python
import numpy as np
from dryml.core.tensor_spec import Dynamic, TensorSpec
from dryml.data import Batch, Cast, GeneratorDataset, Map, Scale, Unbatch

def samples():
    return iter(np.ones((length, 5), dtype="float32") for length in (3, 0, 7))

source = GeneratorDataset(
    samples, cardinality=3,
    spec=TensorSpec("float32", shape=(Dynamic, 5), backend="numpy"),
)
batched = Batch(source, 2, ragged=True)
assert batched.spec.layout.value == "ragged"
assert batched.spec.ragged_rank == 1
assert batched.spec.full_shape == (Dynamic, Dynamic, 5)

mapped = Map(batched, Cast("float64"), Scale(mean=1, std=2, dtype="float64"),
             preserves_examples=True)
graph = mapped.method_graph()
graph.learn()
with graph.iterator() as cursor:
    assert [len(value) for value in cursor] == [2, 1]

assert [value.shape for value in Unbatch(mapped)] == [(3, 5), (0, 5), (7, 5)]
```

`ragged=True` partitions dense sample axes through the last Dynamic axis.
Fixed-shape leaves, such as scalar labels or fixed-width targets, still stack as
dense tensors. Opaque Python leaves retain their list batching contract. A dense
source with `shape=(10, Dynamic)` consequently produces two partition levels and
full shape `(B, 10, L)`, while `shape=(Dynamic, 5)` needs only one.

Existing ragged samples automatically gain one outer partition with ordinary
`Batch(source, size)`; no extra flag is needed. Each source must be logically
unbatched and ragged-producing leaves must declare a NumPy or TensorFlow backend.
Planning is metadata-only; optional TensorFlow activation occurs at runtime.
Dense collation without `ragged=True` continues to reject unequal shapes.

## Physical Partitions And Logical Batch

`ragged_rank` counts row-partition levels in the **full physical tensor**. It is
independent of whether the outermost axis is interpreted as an example batch.
Partitioned axes immediately follow that outermost axis; deeper variable axes
require intervening uniform axes to be partitioned too.

For a batched value with full shape `(B, L, 5)`, declare:

```python
spec = TensorSpec(
    "float32", shape=(Dynamic, 5), batch=Dynamic, backend="numpy",
    layout="ragged", ragged_rank=1, row_splits_dtype="int64",
)
```

Rank must be positive and smaller than known full rank; row splits use int32 or
int64. A None rank/dtype leaves metadata unconstrained, but adding or removing
ragged batch partitions requires a known rank. Explicit ragged batching of dense
samples uses int64 offsets.

`as_tensor_spec(value, batched=True)` **interprets** an existing outer axis and
does not add a partition. In contrast, `with_batch()`/`batch_spec_tree()` **add** a
batch to an unbatched ragged sample and increase its partition rank. Updating an
already batched spec's batch size leaves rank unchanged.

Unbatch removes the outer partition: rank one produces dense rows; greater ranks
produce ragged rows with one fewer level. A row can remain a ragged value even if
its partitions happen to be uniform; no implicit densification occurs.

Empty rows are valid examples and preserve their dense trailing shape. A batch
containing only empty rows still has its positive outer example population.
Dataset example-count checks reject an empty **batch**, not an empty row.
Ragged row lengths do not change example counts or `preserves_examples` semantics.

## NumPy Reference Value

`dryml.numpy.RaggedTensor` is an ordinary runtime value, separate from the
Serializable `NumpyArray` storage container. Its constructor takes a packed
`flat_values` ndarray and outer-to-inner `nested_row_splits` arrays:

```python
from dryml.numpy import RaggedTensor

value = RaggedTensor(
    np.arange(15, dtype="float32").reshape(3, 5),
    (np.array([0, 2, 2, 3], dtype="int64"),),
)
assert value.shape == (3, None, 5)
assert value[1].shape == (0, 5)
```

Partitions must be one-dimensional, start at zero, be nondecreasing, use one
common int32/int64 dtype, and end at the next level's row population (or packed
value population for the innermost level). Malformed input raises TypeError or
ValueError before a usable value is published. The constructor validates owned
copies; ordinary access exposes read-only views. Flat values cannot have object
dtype. This is read-only ownership, not isolation against deliberately altered
NumPy buffer owners.

`RaggedTensor.stack(items, ragged_rank=..., row_splits_dtype=...)` adds an outer
partition, supporting dense or already-ragged inputs with matching dtype and dense
tail shape. `with_flat_values()` replaces packed values while retaining offsets;
`astype()` preserves partitions during dtype conversion. These operations allocate
owned buffers. The reference implementation favors explicit validation and ownership
over zero-copy mutation; it does not define a Store codec or implicit array conversion.
Trusted pickle round trips reconstruct through the same validation and read-only
ownership boundary; pickle remains trusted executable input.

## Method And Backend Contracts

Ragged values are tensor leaves, not structural Python lists or opaque object
leaves. Runtime admission checks dtype, layout, shape, physical partition rank and
declared offset dtype. Fixed uniform widths are proved from partition metadata.
Known-spec Methods admit changing row lengths and batch sizes without retaining
values or partitions in their preparation metadata. Explicit raw-output specs can
validate ragged or dense outputs independently.
When a ragged source leaves offset dtype unspecified, Batch preserves the common
actual dtype within each group and rejects mixed offset dtypes rather than narrowing.

Cast and scalar Scale preserve partitions on NumPy and TensorFlow. Pipe, Project,
Select and authored Methods/Models can carry ragged intermediate values and return
dense outputs. Models still need an explicit or pure inferred output contract;
support for ragged inputs is the authored native model's responsibility.

Flatten and ArgMax reject ragged inference until their ragged-axis semantics are
defined. Spec-only fake samples cannot invent row partitions. Automatic cross-backend
handoffs remain dense-only; native Torch/JAX ragged collation is not provided.
[Padding](padding.md) explicitly produces dense values and optional geometry masks,
with fixed or maximum targets, left/right placement and opt-in truncation. These
operations are not silently introduced by batching or Method selection.

The TensorFlow adapter retains native partition rank/dtype. Eager inference reads
row splits on the host to prove uniform widths, which can synchronize a device;
flat numeric payloads are not copied for inference. TypeSpecs and traced values
use their declared static shape metadata instead. Supported shape-preserving
Methods can execute with a `tf.RaggedTensorSpec` input signature; Dataset iterator
execution is still owned Python cursor execution, not a TensorFlow graph compiler.
