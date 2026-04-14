from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import torch


@dataclass(frozen=True, slots=True)
class TorchTensorSpec:
    """
    DRYML-owned PyTorch-side tensor spec adapter.

    PyTorch has native dtype/layout objects, but no native public TensorSpec
    analogous to tf.TensorSpec or jax.ShapeDtypeStruct.
    """
    shape: tuple[int | object, ...] | None
    dtype: Any
    layout: Any | None = None
    device: Any | None = None
    requires_grad: bool | None = None


    def __post_init__(self):
        if self.layout is None:
            object.__setattr__(self, "layout", torch.strided)

