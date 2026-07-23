from __future__ import annotations

from dryml.core.methods import Method, traits
from dryml.core.tensor_spec import SpecTree, map_spec_tree


class ImageNormalize(Method):
    """Convert image tensors to float32 in the [0, 1] range."""

    def infer_output_spec(self, input_spec: SpecTree) -> SpecTree:
        return map_spec_tree(input_spec, lambda spec: spec.with_dtype("float32"))

    @traits(backend="numpy")
    def numpy_call(self, image):
        import numpy as np

        return image.astype(np.float32) / 255.0

    @traits(backend="tf")
    def tf_call(self, image):
        from dryml.runtime import import_configured_framework
        tf = import_configured_framework("tensorflow")

        return tf.cast(image, tf.float32) / 255.0

    @traits(backend="torch")
    def torch_call(self, image):
        from dryml.runtime import import_configured_framework
        torch = import_configured_framework("torch")

        return image.to(torch.float32) / 255.0


__all__ = ["ImageNormalize"]
