from dryml.code import traits
from dryml.core2.dtype import normalize_dtype
from dryml.core2.tensor_spec import SpecTree, map_spec_tree

from .base import ElementwiseTransform

# Element wise 'simple' transformations
# map, select, rename, cast, normalize

class Select(ElementwiseTransform):
    def __init__(self, *idxs):
        self.idxs = idxs

    def __call__(self, x):
        result = x
        for idx in self.idxs:
            result = result[idx]
        return result

    def infer_output_spec(self, input_spec: SpecTree) -> SpecTree:
        result = input_spec
        for idx in self.idxs:
            result = result[idx]
        return result


class Pipe(ElementwiseTransform):
    def __init__(self, *transforms):
        if not transforms:
            raise ValueError("Pipe requires at least one transform.")
        self.transforms = transforms

    def __call__(self, x):
        result = x
        for transform in self.transforms:
            result = transform(result)
        return result

    def infer_output_spec(self, input_spec: SpecTree) -> SpecTree:
        spec = input_spec
        for transform in self.transforms:
            spec = transform.infer_output_spec(spec)
        return spec

    def bind_first(self, first_value, *, input_spec=None):
        bound_transforms = []
        value = first_value
        spec = input_spec

        for transform in self.transforms:
            if hasattr(transform, "bind_first"):
                bound, value = transform.bind_first(value, input_spec=spec)
            else:
                bound = transform
                value = bound(value)
            bound_transforms.append(bound)

            if spec is not None and hasattr(transform, "infer_output_spec"):
                spec = transform.infer_output_spec(spec)
            else:
                spec = None

        def bound_pipe(x):
            result = x
            for transform in bound_transforms:
                result = transform(result)
            return result

        return bound_pipe, value


class Cast(ElementwiseTransform):
    def __init__(self, dtype):
        self.dtype = normalize_dtype(dtype)

    def infer_output_spec(self, input_spec: SpecTree) -> SpecTree:
        return map_spec_tree(input_spec, lambda spec: spec.with_dtype(self.dtype))

    @traits(backend="numpy")
    def numpy_call(self, x):
        return x.astype(self.dtype.np())

    @traits(backend="tf")
    def tf_call(self, x):
        import tensorflow as tf
        return tf.cast(x, self.dtype.tf())

    @traits(backend="torch")
    def torch_call(self, x):
        return x.to(self.dtype.torch())
