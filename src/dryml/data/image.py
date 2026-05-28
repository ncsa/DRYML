from __future__ import annotations

from dryml.code import traits
from dryml.core2.tensor_spec import SpecTree, map_spec_tree
from dryml.data.transforms import ElementwiseTransform


class ImageNormalize(ElementwiseTransform):
    """Convert image tensors to float32 in the [0, 1] range."""

    def infer_output_spec(self, input_spec: SpecTree) -> SpecTree:
        return map_spec_tree(input_spec, lambda spec: spec.with_dtype("float32"))

    @traits(backend="numpy")
    def numpy_call(self, image):
        import numpy as np

        return image.astype(np.float32) / 255.0

    @traits(backend="tf")
    def tf_call(self, image):
        import tensorflow as tf

        return tf.cast(image, tf.float32) / 255.0

    @traits(backend="torch")
    def torch_call(self, image):
        import torch

        return image.to(torch.float32) / 255.0


__all__ = ["ImageNormalize"]
