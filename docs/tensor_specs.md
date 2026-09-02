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

## DTypes

DRYML normalizes dtype values through its core dtype system. Users can usually pass simple strings such as `"float32"`, `"int64"`, or backend dtype objects when supported.

The normalized dtype is stored on `TensorSpec.dtype`.

## Shape And Rank

Useful properties:

- `rank`: rank of one sample
- `full_rank`: rank including batch, when present
- `shape`: sample shape
- `full_shape`: batch plus sample shape, when batched
- `batched`: whether the spec has a batch axis

Unknown rank is represented by `shape=None`. Dynamic dimensions are represented by `Dynamic`.

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
