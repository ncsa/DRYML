from __future__ import annotations
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class TorchTensorSpec:
    """
    DRYML-owned PyTorch-side tensor spec adapter.

    PyTorch has native dtype/layout objects, but no native public TensorSpec
    analogous to tf.TensorSpec or jax.ShapeDtypeStruct.
    """
    shape: tuple[int | object, ...] | None
    dtype: Any
    layout: Any
    device: Any | None = None
    requires_grad: bool | None = None
