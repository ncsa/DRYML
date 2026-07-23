from typing import Any

from dryml.core.tensor_spec import Dynamic, Layout, TensorSpec
from dryml.core.utils.recurse import map_leaves
from .dtype import dtype


def _tensor_spec_jax(
    self,
    *,
    include_batch: bool = True,
    sharding=None,
    weak_type: bool = False,
):
    from dryml.runtime import import_configured_framework
    jax = import_configured_framework("jax")

    if self.layout is not Layout.DENSE:
        raise TypeError("Default TensorSpec.jax() only supports dense tensors.")

    shape = self.framework_shape(include_batch=include_batch)
    if shape is None:
        raise ValueError("Cannot convert unknown-rank TensorSpec to jax.ShapeDtypeStruct.")

    jax_shape = []
    for d in shape:
        if d is Dynamic:
            raise ValueError(
                "Dynamic dimensions cannot be represented by plain jax.ShapeDtypeStruct."
            )
        jax_shape.append(int(d))

    return jax.ShapeDtypeStruct(
        tuple(jax_shape),
        self.dtype.jax(),
        sharding=sharding,
        weak_type=weak_type,
    )


def _shape_to_dryml(shape: Any) -> tuple[int, ...] | None:
    if shape is None:
        return None
    return tuple(int(d) for d in shape)


def _split_batch(
    shape: tuple[int, ...] | None,
    *,
    batched: bool,
) -> tuple[tuple[int, ...] | None, int | None]:
    if not batched:
        return shape, None

    if shape is None:
        raise ValueError(
            "Cannot set batched=True when the JAX shape has unknown rank."
        )

    if len(shape) == 0:
        raise ValueError(
            "Cannot set batched=True for a rank-0 JAX value/spec."
        )

    return shape[1:], shape[0]


def as_tensor_spec(
    x: Any,
    *,
    batched: bool = False,
    batch_axis_name: str | None = "batch",
) -> TensorSpec:
    """
    Convert a jax.ShapeDtypeStruct or JAX array-like object to a DRYML TensorSpec.

    This treats JAX specs/arrays as dense tensor metadata.
    """


    def leaf_to_spec(x: Any) -> TensorSpec:
        if not hasattr(x, "shape") or not hasattr(x, "dtype"):
            raise TypeError(
                "dryml.jax.as_tensor_spec(x) expects a jax.ShapeDtypeStruct-like "
                "or JAX array-like object with .shape and .dtype."
            )

        full_shape = _shape_to_dryml(x.shape)
        sample_shape, batch = _split_batch(full_shape, batched=batched)

        return TensorSpec(
            dtype=dtype(x.dtype),
            shape=sample_shape,
            batch=batch,
            layout=Layout.DENSE,
            batch_axis_name=batch_axis_name if batch is not None else None,
            backend = "jax",
        )

    return map_leaves(x, leaf_to_spec)
