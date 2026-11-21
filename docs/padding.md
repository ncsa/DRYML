# Padding And Validity Masks

`dryml.data.Padding` is a pure Method that turns dense or supported ragged inputs
into padded dense tensors. It can return the matching boolean validity mask in
the same invocation. Ordinary Map, Method composition and prepared stream graphs
share the operation and output contracts.

## Basic Use

```python
import numpy as np
from dryml.core.tensor_spec import Dynamic, TensorSpec
from dryml.data import Batch, GeneratorDataset, Map, Padding

def samples():
    return iter(np.ones((length, 5), dtype="float32") for length in (0, 2, 4))

source = GeneratorDataset(samples, cardinality=3,
    spec=TensorSpec("float32", shape=(Dynamic, 5), backend="numpy"))
ragged = Batch(source, 2, ragged=True)
padded = Map(ragged, Padding(5, return_mask=True), preserves_examples=True)

graph = padded.method_graph()
graph.learn()
with graph.iterator() as cursor:
    first = next(cursor)
assert first["values"].shape == (2, 5, 5)
assert first["mask"].shape == (2, 5)
assert first["mask"].tolist() == [
    [False, False, False, False, False],
    [True, True, False, False, False],
]
```

`Padding(5)` returns only a tensor. `Padding(5, return_mask=True)` returns
`{"values": tensor, "mask": bool_tensor}`. The dictionary is an ordinary SpecTree
and runtime structure: Select/Project can extract or rename fields, and an authored
Model can consume both. Masks are explicit outputs, not automatically attached
Keras masks or sample weights.

Fixed targets can also be applied to individual dense samples before ordinary
Batch. For batch-maximum targets, form a ragged batch first so the transform sees
all row lengths in that batch.

## Axes And Targets

`axis=0` refers to the first **sample** axis, excluding any logical batch axis.
Negative axes follow sample rank as well. For `(B, L, 5)`, `axis=0` pads L and
leaves B and the feature width unchanged. Padding never creates examples by
padding the batch axis.

`length` accepts a nonnegative integer, None for the maximum in the current input,
or a tuple matching a tuple of axes. A scalar target applies to every selected
axis. For example:

```python
Padding(128)                                  # fixed sequence length
Padding()                                     # current input/batch maximum
Padding((128, 64), axis=(0, 1))                # two fixed sample axes
Padding((None, 64), axis=(0, 1), return_mask=True)
```

Maximum targets examine this invocation's geometry, not future Dataset values.
Dense dimensions are already rectangular, so a maximum target leaves that
dimension unchanged. Ragged targets use the maximum row length at each selected
partition level. Every variable ragged sample axis must be selected; unselected
partitioned axes must be uniform. A fixed prefix such as the 10 in `(B, 10, L)`
can remain unselected while padding L with `axis=1`.

`side="right"` appends padding; `side="left"` prepends it independently within
each row. For example, two values padded to length four have masks
`[True, True, False, False]` and `[False, False, True, True]`, respectively.
All selected axes use the same side.

## Mask Meaning And Shape

True marks an original **retained position**; False marks an artificial padded
position. Mask generation uses sizes and row partitions, never comparison with
the fill value. Original zeros, NaNs, and False boolean values remain valid.

The mask contains the batch axis, if present, plus the sample prefix through the
last selected padding axis. Unchanged trailing feature dimensions are omitted:

| Values shape | Padded sample axes | Mask shape |
| --- | --- | --- |
| `(B, T, 5)` | `0` | `(B, T)` |
| `(B, H, W, C)` | `(0, 1)` | `(B, H, W)` |
| `(T, 5)` without batch metadata | `0` | `(T,)` |
| `(B, T, F)` with F also padded | `(0, 1)` | `(B, T, F)` |

Empty rows have entirely false masks. All-empty ragged batches padded to their
maximum produce a zero-width dimension; fixed targets instead produce filled
values and false masks. Zero-sized trailing features do not erase valid prefix
positions. Dataset counting still distinguishes an empty sample from an empty
batch and rejects zero-example batches under its existing contract.

Dense input positions are all originally present. Padding cannot recover an
external padding mask from an already-padded dense tensor without external length
or partition evidence.

## Truncation And Failure

Oversized input is rejected by default. Use `truncate=True` to discard the right
tail with right padding, or the left head with left padding. Masks mark retained
data, not discarded data. Targets and maxima are resolved over the whole original
input before truncation, including nested rows later removed by another axis.

Malformed options fail during declaration. Inference rejects unknown rank,
invalid/aliased sample axes, unsupported layouts/backends, missing ragged rank,
unselected variable ragged axes, and statically provable oversize conflicts.
Dynamic conflicts fail during execution. NumPy/Torch/JAX use ValueError for size
conflicts; TensorFlow graph assertions may raise native InvalidArgumentError.
Native casting/allocation errors propagate, and Dataset cursors close their
children through the ordinary failure path. Input buffers are not changed.

Fill values are Python scalars cast according to native rules while preserving
the input dtype. Choose a value compatible with that dtype; no output dtype is
inferred from the fill value.

## Specs And Backends

Fixed targets give fixed output dimensions. Maximum targets retain a declared
fixed dimension or Dynamic. Outputs have dense layout with ragged metadata
cleared. Masks have bool dtype, the same backend/batch metadata, and the prefix
of the value's sample-axis names. Output inference is metadata-only and evaluates
no sources or optional frameworks. Use an explicit input spec for reusable Dynamic
calls; ordinary first-call numerical learning retains its exact-input contract.

NumPy and TensorFlow support ragged-to-dense padding and dense padding. Torch and
JAX provide dense native implementations. Padded values/masks can then use the
existing dense backend handoff mechanism, for example from NumPy ragged sources
through Padding into a Torch model. TensorFlow supports dynamic dense/ragged input
signatures; JAX supports shape-specialized native JIT execution. Retained numeric
values keep their native gradients on TensorFlow, Torch and JAX. IR preparation
reuse does not promise one backend executable for every concrete input shape.
