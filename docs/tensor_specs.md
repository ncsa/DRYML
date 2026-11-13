# Tensor Specs

Status: draft.

`TensorSpec` is DRYML's backend-independent description of tensor-like values. It lets datasets, models, and methods communicate expected shapes, dtypes, layouts, and batch semantics without depending on a single ML framework.

## Basic Spec

```python
from dryml.core import TensorSpec

image = TensorSpec("float32", shape=(28, 28, 1))
```

This describes one unbatched sample with dtype `float32` and sample shape `(28, 28, 1)`.

## Batch Semantics

`shape` describes one sample. `batch` describes whether there is a batch axis.

```python
from dryml.core.tensor_spec import Dynamic

batched = image.with_batch(Dynamic)

assert batched.shape == (28, 28, 1)
assert batched.full_shape == (Dynamic, 28, 28, 1)
```

Batch values:

- `None`: unbatched
- `Dynamic`: batched with unknown batch size
- integer: fixed batch size

When a Method is prepared from a known `TensorSpec`, `Dynamic` is a retained
validation allowance rather than the size observed on its first value. For
example, a `batch=Dynamic` prepared Method may accept batches of 64 and then 17;
a fixed batch or fixed sample dimension rejects drift before the selected Method
body runs. This validation remains local preparation behavior and does not infer
or execute Dataset streams.

## DTypes

DRYML normalizes dtype values through its core dtype system. Users can usually pass simple strings such as `"float32"`, `"int64"`, or backend dtype objects when supported. NumPy Unicode dtypes, regardless of their storage width, normalize to the canonical `"string"` dtype and convert back to `np.str_`. NumPy object dtypes retain the distinct `"object"` kind; byte strings are not implicitly decoded. Numerical reductions and classification label validation still reject object arrays.

The normalized dtype is stored on `TensorSpec.dtype`.

## Opaque Python Values

Use the canonical dtype name `"object"` with `layout="python"` to declare an
arbitrary Python value. DRYML records no Python class, fields, geometry, array
representation, or content fingerprint:

```python
from dryml.core import TensorSpec
from dryml.core.tensor_spec import Dynamic

object_spec = TensorSpec("object", layout="python")
object_batch_spec = object_spec.with_batch(Dynamic)
assert object_spec.is_python_object
```

An unbatched leaf accepts any value unchanged, including `None`, a mesh, a dict,
a list, or a tensor. A batched leaf accepts a list or tuple of arbitrary values;
only its outer length is validated against fixed or `Dynamic` batch metadata.
It does not require homogeneous Python classes or inspect the items. A list
inside an unbatched object leaf is one opaque value, not a structural list.
Outer spec-tree dict/tuple/list declarations still define ordinary structure.

Opaque sample shape is `()` (or unknown rank), and there is no tensor backend or
sparse/ragged metadata. `Batch` produces lists of the original references;
`Unbatch` yields those same references. No array coercion or optional tensor
framework is needed. Synthetic samples cannot be generated from an opaque spec.

For a NumPy object array, use the existing dense layout and NumPy backend:

```python
numpy_object_batch = TensorSpec("object", shape=(), batch=Dynamic, backend="numpy")
```

Here the outer array supplies a tensor batch axis, while its entries contain
arbitrary Python references. `"numpy"` is a backend, not a layout. Canonical
object dtype maps back to `np.object_` through `DType.np()`. Numerical reductions
and implicit cross-framework tensor handoffs still require supported numerical
dtypes.

`as_tensor_spec(value)` also infers unrecognized Python leaves as opaque specs.
It preserves standard dict/list/tuple/namedtuple structure, recognizes numerical
leaves through their existing adapters, and ignores opaque leaves when checking
backend consistency. Native adapter or dtype failures are errors, not opaque
fallbacks. Inferred metadata retains no Python object or class.

With `batched=True`, a nonempty list/tuple of opaque noncontainer leaves is one
object batch. `detect_spec_tree(samples, SpecHint(batch_mode="batched"))` merges
different outer lengths to `Dynamic`. Empty or object-valued structural containers
are ambiguous: supply an explicit object spec to keep them opaque. An explicit
spec also avoids sampling a generator during construction.

Object specs work for Method inputs, outputs, and intermediate graph edges.
First-call learning can infer opaque leaves; known-spec preparation accepts an
explicit declaration before values exist. The conversion Method owns
interpretation of object content and reports its numerical output with an
ordinary fixed-shape `TensorSpec`.

## Shape And Rank

Useful properties:

- `rank`: rank of one sample
- `full_rank`: rank including batch, when present
- `shape`: sample shape
- `full_shape`: batch plus sample shape, when batched
- `batched`: whether the spec has a batch axis

Unknown rank is represented by `shape=None`. Dynamic dimensions are represented by `Dynamic`.

### Dynamic Dense Dimensions And Variable Batches

Dynamic axes are supported in dense sample shapes independently of the batch
size. These declarations have known full rank three:

| Full shape contract | Sample shape | Batch |
| --- | --- | --- |
| `(B, 10, L)` | `(10, Dynamic)` | `Dynamic` |
| `(B, L, 5)` | `(Dynamic, 5)` | `Dynamic` |
| `(B, 10, 5)` | `(10, 5)` | `Dynamic` |
| `(B, L, F)` | `(Dynamic, Dynamic)` | `Dynamic` |

Each yielded or passed dense tensor remains rectangular. Sizes may differ
between calls or Dataset yields; unequal-length rows within one tensor require
a separate ragged or padded representation. A repeated `Dynamic` is not a named
symbol and does not establish equality between axes or different tensors.

```python
import numpy as np
from dryml.core.tensor_spec import Dynamic, TensorSpec
from dryml.data import Flatten

inputs = TensorSpec("float32", shape=(Dynamic, 5), batch=Dynamic, backend="numpy")
flatten = Flatten()
outputs = flatten.infer_output_spec(inputs)
assert outputs.shape == (Dynamic,)
flatten.learn(inputs, output_spec=outputs)
assert flatten(np.ones((4, 12, 5), dtype="float32")).shape == (4, 60)
assert flatten(np.ones((1, 7, 5), dtype="float32")).shape == (1, 35)
```

Known-spec Method preparation retains these wildcards, validates fixed dimensions,
rank, dtype, layout and backend on each call, and reuses its selected invoker.
An explicit `output_spec` checks the raw result as well. In contrast, `learn()`
without a spec captures the first numerical call's exact physical shape; observing
one value does not establish which dimensions may vary.

Shape inference follows each Method's operation: Cast and Scale preserve sample
dimensions; ArgMax removes its selected sample axis and axis name; Flatten
multiplies fixed sizes and otherwise retains a Dynamic width. A known zero factor
proves a flattened width of zero even beside Dynamic dimensions. Flatten preserves
a single sample-axis name and uses an unnamed axis when collapsing several axes.
Native arrays carry physical axes rather than DRYML labels, so known-spec validation
interprets their axis names from the declaration. Explicit TensorSpec arguments
must still carry compatible labels of their own.

`merge_tensor_specs` and sample inference generalize varying sizes to Dynamic
without mixing batched and unbatched declarations. Sampling can only establish
variation actually observed; use an explicit spec for other intended variability.

Logical dynamic specs do not allocate unknown-sized arrays. NumPy's concrete
backend-spec conversion still requires fixed sizes. TensorFlow, Torch and JAX
runtime values can be validated under a DRYML dynamic contract; this does not
promise a backend JIT will compile every concrete shape into the same executable.

## Layouts

`TensorSpec.layout` describes the broad storage style:

- dense
- ragged
- sparse
- python

Dense tensors are the common default. Ragged and sparse specs can carry extra layout metadata such as ragged rank, row-splits dtype, or sparse format.

## Spec Trees

Many datasets and models operate on nested structures, not one tensor. DRYML supports spec trees made from `TensorSpec`, `dict`, `tuple`, and `list`.

```python
from dryml.core import TensorSpec

sample_spec = {
    "x": TensorSpec("float32", shape=(32,)),
    "y": TensorSpec("int64", shape=()),
}
```

Useful helpers include:

- `iter_specs(spec_tree)`
- `batch_spec_tree(spec_tree)`
- `unbatch_spec_tree(spec_tree)`
- `assert_same_spec_structure(...)`

## Backend Use

Backend integrations can convert a `TensorSpec` into framework-specific shapes, signatures, arrays, or tensors. The spec itself should stay semantic and portable.

Prepared local Method graphs use a complete backend field as a declared boundary,
not as a claim about an arbitrary runtime value. Supported dense CPU NumPy,
TensorFlow, Torch, and JAX handoffs preserve the semantic spec exactly while
changing that field on the consumer side. JAX handoffs require a concrete,
fully-addressable, single-device CPU source and explicitly place the target on CPU.
They reject a dtype JAX would narrow under the active x64 configuration without
changing that configuration. GPU/multi-device transfers, tracers or abstract JAX
values, sparse/ragged/quantized tensors, object/string/complex/bfloat values, and
implicit lossy dtype changes are outside this handoff contract.

## Equality

Ordinary `TensorSpec` equality is backend-neutral semantic equality. Otherwise
matching NumPy and Torch specs can compare equal when their portable tensor
meaning agrees. Use `equal_exact(other)` when a Method cache or another exact
contract must also compare backend and every stored specification field. It
returns `False` for non-`TensorSpec` values and does not change normal equality
or hashing.

```python
from dryml.core import TensorSpec

numpy_spec = TensorSpec("float32", shape=(2,), backend="numpy")
torch_spec = TensorSpec("float32", shape=(2,), backend="torch")
assert numpy_spec == torch_spec
assert not numpy_spec.equal_exact(torch_spec)
```

Examples of backend-specific consumers:

- TensorFlow model signatures
- PyTorch batch conversion
- NumPy array datasets
- JAX input/output conventions

## Datasets And Models

Datasets expose the spec of yielded elements. Methods and models use specs to infer output structures.

```python
import numpy as np

from dryml.core import TensorSpec
from dryml.data import ArrayDataset

dataset = ArrayDataset(
    np.array([1, 2, 3], dtype="int64"),
    spec=TensorSpec("int64", shape=()),
)
print(dataset.spec)
```

Models can infer output specs. When model output specs are unbatched but inputs are batched, DRYML can propagate the batch dimension onto the output spec.

Dataset-owned native training keeps the authored `(inputs, targets)` spec tree.
`Batch` supplies batch metadata before preparation, and each trainer retains the
selected backend-specific consumer specs in `TrainingPreparation`. TensorFlow,
Torch, and experimental JAX adapters preserve the same semantic dtype, shape,
structure, and batch meaning; they do not infer trainer-owned batch controls.

## Common Pitfalls

- Do not include the batch dimension in `shape`; use `batch` instead.
- Use `Dynamic` for unknown dimensions, not `None` inside a shape.
- Use `shape=None` only when rank itself is unknown.
- Keep spec trees structurally consistent across inputs and outputs.
- Treat backend-specific dtype behavior as integration-specific.

## Related Docs

- [Data API](data.md)
- [Methods](methods.md)
- [Models API](models.md)
- [Contexts](context.md)
