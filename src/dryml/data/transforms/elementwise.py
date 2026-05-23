from dryml.code import Method, backend_impl

# Element wise 'simple' transformations
# map, select, rename, cast, normalize

class Select(Method):
    def __init__(self, idxs):
        self.idxs = idxs

    def __call__(self, x):
        result = x
        for idx in self.idxs:
            result = result[idx]
        return result


class Cast(Method):
    def __init__(self, dtype):
        self.dtype = dtype

    @backend_impl('numpy')
    def numpy_call(self, x):
        return x.astype(self.dtype.np())

    @backend_impl('tf')
    def tf_call(self, x):
        import tensorflow as tf
        return tf.cast(x, self.dtype.tf())

    @backend_impl('torch')
    def torch_call(self, x):
        return x.to(self.dtype.torch())
