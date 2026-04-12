from __future__ import annotations
from dataclasses import dataclass
from dryml.core2.tensor_spec import TensorSpec, Dynamic, Layout
from dryml.core2.dtype import normalize_dtype
from dryml.core2.utils.recurse import map_leaves
from .spec import TorchTensorSpec
from .dtype import dtype

def _tensor_spec_torch(
    self,
    *,
    include_batch: bool = True,
    device: str | None = None,
    requires_grad: bool | None = None,
):
    import torch
    from .spec import TorchTensorSpec

    shape = self.framework_shape(include_batch=include_batch)

    if self.layout is Layout.DENSE:
        torch_layout = torch.strided
    elif self.layout is Layout.SPARSE:
        fmt = self.sparse_format
        if fmt == "coo":
            torch_layout = torch.sparse_coo
        elif fmt == "csr":
            torch_layout = torch.sparse_csr
        else:
            raise ValueError("PyTorch sparse conversion requires sparse_format='coo' or 'csr'.")
    else:
        raise TypeError("PyTorch conversion does not support ragged layout.")

    return TorchTensorSpec(
        shape=shape,
        dtype=self.dtype.torch(),
        layout=torch_layout,
        device=device,
        requires_grad=requires_grad,
    )


def _shape_to_dryml(shape: Any) -> tuple[int | object, ...] | None:
    if shape is None:
        return None

    out = []
    for d in shape:
        if d is Dynamic:
            out.append(Dynamic)
        else:
            out.append(int(d))
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
            "Cannot set assume_batched=True when the PyTorch shape has unknown rank."
        )

    if len(shape) == 0:
        raise ValueError(
            "Cannot set assume_batched=True for a rank-0 PyTorch tensor/spec."
        )

    return shape[1:], shape[0]


def _layout_from_torch_layout(layout: Any) -> tuple[Layout, str | None]:
    s = str(layout)

    if s == "torch.strided":
        return Layout.DENSE, None
    if s == "torch.sparse_coo":
        return Layout.SPARSE, "coo"
    if s == "torch.sparse_csr":
        return Layout.SPARSE, "csr"
    if s == "torch.sparse_csc":
        return Layout.SPARSE, "csc"
    if s == "torch.sparse_bsr":
        return Layout.SPARSE, "bsr"
    if s == "torch.sparse_bsc":
        return Layout.SPARSE, "bsc"

    raise TypeError(f"Unsupported PyTorch layout: {layout!r}")


def dtype(x: Any) -> DType:
    """
    Convert a torch.dtype, torch.Tensor, or TorchTensorSpec to a DRYML DType.
    """
    if hasattr(x, "dtype"):
        x = x.dtype

    if isinstance(x, str):
        name = x.removeprefix("torch.")
        return normalize_dtype(name)

    s = str(x)
    if not s.startswith("torch."):
        raise TypeError(f"Unsupported PyTorch dtype-like object: {x!r}")

    return normalize_dtype(s.removeprefix("torch."))


def tensor_spec(
    x: Any,
    *,
    assume_batched: bool = False,
    batch_axis_name: str | None = "batch",
) -> TensorSpec:
    """
    Convert a torch.Tensor or DRYML TorchTensorSpec to a DRYML TensorSpec.

    Parameters
    ----------
    assume_batched:
        PyTorch tensors carry shape, dtype, and layout, but not batch semantics.
        If True, interpret the leading axis as batch.
    """


    def leaf_to_spec(x: Any) -> TensorSpec:

        if isinstance(x, TorchTensorSpec):
            full_shape = _shape_to_dryml(x.shape)
            out_dtype = dtype(x.dtype)
            layout, sparse_format = _layout_from_torch_layout(x.layout)
        else:
            if not hasattr(x, "shape") or not hasattr(x, "dtype"):
                raise TypeError(
                    "dryml.torch.tensor_spec(x) expects a torch.Tensor-like object "
                    "or TorchTensorSpec."
                )

            full_shape = _shape_to_dryml(x.shape)
            out_dtype = dtype(x.dtype)

            layout_obj = getattr(x, "layout", None)
            if layout_obj is None:
                raise TypeError(
                    f"Cannot determine PyTorch layout for object of type {type(x).__name__}."
                )

            layout, sparse_format = _layout_from_torch_layout(layout_obj)

        sample_shape, batch = _split_batch(full_shape, assume_batched=assume_batched)

        return TensorSpec(
            dtype=out_dtype,
            shape=sample_shape,
            batch=batch,
            layout=layout,
            batch_axis_name=batch_axis_name if batch is not None else None,
            sparse_format=sparse_format,
            backend = "torch",
        )

    return map_leaves(x, leaf_to_spec)
