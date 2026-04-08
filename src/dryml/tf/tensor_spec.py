import tensorflow as tf
from typing import Any
from dryml.core2.tensor_spec import Dynamic, Layout, TensorSpec, SpecTree, map_tree_leaves
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
    assume_batched: bool,
) -> tuple[tuple[int | object, ...] | None, int | object | None]:
    if not assume_batched:
        return shape, None

    if shape is None:
        raise ValueError(
            "Cannot set assume_batched=True when the TensorFlow shape has unknown rank."
        )

    if len(shape) == 0:
        raise ValueError(
            "Cannot set assume_batched=True for a rank-0 TensorFlow spec/value."
        )

    return shape[1:], shape[0]


def tensor_spec(
    x: SpecTree,
    *,
    assume_batched: bool = False,
    batch_axis_name: str | None = "batch",
) -> SpecTree:
    """
    Convert a TensorFlow TypeSpec or TensorFlow value to a DRYML TensorSpec.

    Parameters
    ----------
    assume_batched:
        TensorFlow specs do not intrinsically identify a "batch axis".
        If True, interpret the leading axis as batch.
    """
    def leaf_to_spec(x: Any) -> TensorSpec:

        if not isinstance(x, tf.TypeSpec):
            x = tf.type_spec_from_value(x)

        if isinstance(x, tf.RaggedTensorSpec):
            shape = _tf_shape_to_dryml(x.shape)
            sample_shape, batch = _split_batch(shape, assume_batched=assume_batched)

            return TensorSpec(
                dtype=dtype(x.dtype),
                shape=sample_shape,
                batch=batch,
                layout=Layout.RAGGED,
                batch_axis_name=batch_axis_name if batch is not None else None,
                ragged_rank=int(x.ragged_rank),
                row_splits_dtype=dtype(x.row_splits_dtype),
            )

        if isinstance(x, tf.SparseTensorSpec):
            shape = _tf_shape_to_dryml(x.shape)
            sample_shape, batch = _split_batch(shape, assume_batched=assume_batched)

            return TensorSpec(
                dtype=dtype(x.dtype),
                shape=sample_shape,
                batch=batch,
                layout=Layout.SPARSE,
                batch_axis_name=batch_axis_name if batch is not None else None,
                sparse_format="tf_sparse",
            )

        if isinstance(x, tf.TensorSpec):
            shape = _tf_shape_to_dryml(x.shape)
            sample_shape, batch = _split_batch(shape, assume_batched=assume_batched)

            return TensorSpec(
                dtype=dtype(x.dtype),
                shape=sample_shape,
                batch=batch,
                layout=Layout.DENSE,
                batch_axis_name=batch_axis_name if batch is not None else None,
            )

        raise TypeError(f"Unsupported TensorFlow spec/value type: {type(x).__name__}")

    return map_tree_leaves(x, leaf_to_spec)
