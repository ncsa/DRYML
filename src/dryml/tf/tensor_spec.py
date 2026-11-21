"""Lazy TensorFlow signature conversion and leaf-local metadata inference.

TensorFlow is imported only when a leaf needs it. During function tracing,
exact built-in TypeSpecs with ordinary metadata are inspected as Python without
AutoGraph conversion; tree traversal and custom value/TypeSpec hooks keep their
normal conversion context. No inferred results or tensor values are cached.
"""

from typing import Any
from dryml.core.tensor_spec import Dynamic, Layout, TensorSpec, SpecTree
from dryml.core.utils.recurse import map_leaves
from .dtype import dtype


def dims_to_tf(shape):
    if shape is None:
        return None
    out = []
    for d in shape:
        if d is Dynamic:
            out.append(None)
        else:
            out.append(int(d))
    return tuple(out)


def _tensor_spec_tf(self, *, include_batch: bool = True, name: str | None = None):
    import tensorflow as tf

    if not include_batch and self.batched:
        return _tensor_spec_tf(self.without_batch(), include_batch=True, name=name)
    shape = self.framework_shape(include_batch=include_batch)
    tf_shape = dims_to_tf(shape)
    tf_dtype = self.dtype.tf()

    if self.layout is Layout.DENSE:
        return tf.TensorSpec(shape=tf_shape, dtype=tf_dtype, name=name)

    if self.layout is Layout.RAGGED:
        ragged_rank = self.ragged_rank
        if ragged_rank is None:
            raise ValueError("TensorSpec.ragged_rank is required for TensorFlow ragged conversion.")
        row_splits_dtype = (
            tf.int64 if self.row_splits_dtype is None else self.row_splits_dtype.tf()
        )
        return tf.RaggedTensorSpec(
            shape=tf_shape,
            dtype=tf_dtype,
            ragged_rank=ragged_rank,
            row_splits_dtype=row_splits_dtype,
        )

    if self.layout is Layout.SPARSE:
        return tf.SparseTensorSpec(shape=tf_shape, dtype=tf_dtype)

    raise TypeError(f"Unsupported TensorFlow layout: {self.layout}")


def output_signature(spec_tree: SpecTree, *, flexible_batch: bool = True):
    """Convert a DRYML SpecTree into a tf.data output_signature tree."""
    def leaf_to_tf(spec: TensorSpec):
        # The final DRYML batch may be smaller than the nominal batch size.
        if flexible_batch and spec.batched and spec.shape is not None:
            spec = spec.with_batch(batch=Dynamic)
        return spec.tf()

    return map_leaves(spec_tree, leaf_to_tf, pred=lambda x: isinstance(x, TensorSpec))


def _tf_shape_to_dryml(shape: Any) -> tuple[int | object, ...] | None:
    """
    Convert a tf.TensorShape-like object to DRYML shape form.

    Unknown rank -> None
    Unknown dim  -> Dynamic
    """
    try:
        dims = shape.as_list()
    except ValueError:
        return None

    out = []
    for d in dims:
        out.append(Dynamic if d is None else int(d))
    return tuple(out)


def _split_batch(
    shape: tuple[int | object, ...] | None,
    *,
    batched: bool,
) -> tuple[tuple[int | object, ...] | None, int | object | None]:
    if not batched:
        return shape, None

    if shape is None:
        raise ValueError(
            "Cannot set batched=True when the TensorFlow shape has unknown rank."
        )

    if len(shape) == 0:
        raise ValueError(
            "Cannot set batched=True for a rank-0 TensorFlow spec/value."
        )

    return shape[1:], shape[0]


def _spec_from_tf_metadata(
    x: Any,
    value: Any,
    *,
    batched: bool,
    batch_axis_name: str | None,
) -> TensorSpec:
    """Inspect one resolved TypeSpec, retaining eager-only row-split inference."""
    import tensorflow as tf

    if isinstance(x, tf.RaggedTensorSpec):
        shape = _tf_shape_to_dryml(x.shape)
        if isinstance(value, tf.RaggedTensor) and tf.executing_eagerly() and shape is not None:
            import numpy as np
            shape = list(shape)
            for axis, splits in enumerate(value.nested_row_splits, start=1):
                lengths = np.diff(splits.numpy())
                if len(lengths) and np.all(lengths == lengths[0]):
                    shape[axis] = int(lengths[0])
            shape = tuple(shape)
        sample_shape, batch = _split_batch(shape, batched=batched)

        return TensorSpec(
            dtype=dtype(x.dtype),
            shape=sample_shape,
            batch=batch,
            layout=Layout.RAGGED,
            batch_axis_name=batch_axis_name if batch is not None else None,
            ragged_rank=int(x.ragged_rank),
            row_splits_dtype=dtype(x.row_splits_dtype),
            backend="tf",
        )

    if isinstance(x, tf.SparseTensorSpec):
        shape = _tf_shape_to_dryml(x.shape)
        sample_shape, batch = _split_batch(shape, batched=batched)

        return TensorSpec(
            dtype=dtype(x.dtype),
            shape=sample_shape,
            batch=batch,
            layout=Layout.SPARSE,
            batch_axis_name=batch_axis_name if batch is not None else None,
            sparse_format="tf_sparse",
            backend="tf",
        )

    if isinstance(x, tf.TensorSpec):
        shape = _tf_shape_to_dryml(x.shape)
        sample_shape, batch = _split_batch(shape, batched=batched)

        return TensorSpec(
            dtype=dtype(x.dtype),
            shape=sample_shape,
            batch=batch,
            layout=Layout.DENSE,
            batch_axis_name=batch_axis_name if batch is not None else None,
            backend="tf",
        )

    raise TypeError(f"Unsupported TensorFlow spec/value type: {type(x).__name__}")


def as_tensor_spec(
    x: SpecTree,
    *,
    batched: bool = False,
    batch_axis_name: str | None = "batch",
) -> SpecTree:
    """
    Infer metadata for a TensorFlow value/TypeSpec or a plain tree of them.

    Args:
        x: Dense, sparse or RaggedTensor values/TypeSpecs, possibly in a tree.
        batched: Interpret the leading physical axis as batch without changing
            ragged_rank, which always counts physical partition levels.
        batch_axis_name: Optional logical batch label.

    Returns:
        A SpecTree preserving layout, dtype and ragged partition metadata.
        Eager ragged values additionally prove uniform widths from row splits;
        TypeSpecs and traced values retain their declared static shape facts.
        During TensorFlow function tracing, exact built-in TypeSpecs with ordinary
        metadata are inspected without AutoGraph rewriting. Custom value
        conversion and TypeSpec/metadata subclasses retain their ordinary
        conversion context.

    Raises:
        TypeError: For unsupported values, TypeSpecs or dtypes.
        ValueError: For scalar/unknown-rank batching, invalid ragged metadata or
            unrecognized canonical dtype names.

    Side Effects:
        Imports TensorFlow only when visiting a leaf; empty trees remain passive.
        Eager ragged shape inference copies row splits to the host and may
        synchronize the device; numeric flat values are not copied or retained.
        No tensor values or inferred results are cached.
    """
    def leaf_to_spec(x: Any) -> TensorSpec:
        import tensorflow as tf

        value = x
        if not isinstance(x, tf.TypeSpec):
            x = tf.type_spec_from_value(x)

        # Custom conversion hooks run above; subclass metadata stays convertible.
        if (tf.inside_function()
                and type(x) in (tf.TensorSpec, tf.RaggedTensorSpec, tf.SparseTensorSpec)
                and type(x.shape) is tf.TensorShape
                and type(x.dtype) is tf.dtypes.DType
                and type(batched) is bool
                and (batch_axis_name is None or type(batch_axis_name) is str)):
            if (type(x) is not tf.RaggedTensorSpec
                    or (type(x.ragged_rank) is int
                        and type(x.row_splits_dtype) is tf.dtypes.DType)):
                return tf.autograph.experimental.do_not_convert(_spec_from_tf_metadata)(
                    x, value, batched=batched, batch_axis_name=batch_axis_name)
        return _spec_from_tf_metadata(x, value, batched=batched, batch_axis_name=batch_axis_name)

    return map_leaves(x, leaf_to_spec)
