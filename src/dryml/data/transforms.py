from dryml.data.util import func_source_extract
from dryml.code import Method, MultiFrameworkMethod
import numpy as np


class FuncTransform(Method):
    @staticmethod
    def from_function(
            func,
            func_args=(),
            func_kwargs={},
            framework=None):
        return FuncTransform(
            func_source_extract(func),
            func_args=func_args,
            func_kwargs=func_kwargs,
            framework=framework,
            **kwargs)

    def __init__(
            self,
            func_code,
            func_args=(),
            func_kwargs={}):
        # Save any arguments which will be passed after the
        # data to the function
        self.args = func_args
        self.kwargs = func_kwargs
        self.framework = framework

        # Evaluate passed function code
        lcls = {}
        exec(func_code, globals(), lcls)

        # Check for function definition
        if len(lcls) == 0:
            raise ValueError("Code defines no objects!")
        if len(lcls) > 1:
            raise ValueError("Code defines more than one object!")

        # Get newly defined object
        func = list(lcls.values())[0]

        if not callable(func):
            raise ValueError(
                "Function code doesn't contain a function definition!")

        self.func = func

    def __call__(self, x):
        return self.func(x, *self.args, **self.kwargs)

class BestCat(MultiFrameworkMethod):
    def numpy_call(self, x):
        return np.argmax(x, axis=-1)

    def tf_call(self, x):
        import tensorflow as tf
        return tf.argmax(x, axis=-1)

    def torch_call(self, x):
        import torch
        return torch.argmax(x, dim=-1)


class Flatten(MultiFrameworkMethod):
    def numpy_call(self, x):
        #return x.reshape([x.shape[0], -1])
        return x.flatten()

    def tf_call(self, x):
        import tensorflow as tf
        #return tf.reshape(x, [tf.shape(x)[0], -1])
        return tf.reshape(x, [-1])

    def torch_eval(self, x):
        import torch
        #return torch.reshape(x, (torch.shape[0], -1))
        return torch.reshape(x, (-1,))


class Transpose(MultiFrameworkMethod):
    def __init__(self, axes=None):
        self.axes = axes

    def numpy_call(self, x):
        # Move axes up by one
        #new_axes = [0]
        #for i in self.axes:
        #    new_axes.append(i+1)
        #new_axes = tuple(new_axes)
        return np.transpose(x, self.axes)

    def tf_call(self, x):
        import tensorflow as tf
        #new_axes = [0]
        #for i in self.axes:
        #    new_axes.append(i+1)
        #new_axes = tuple(new_axes)
        return tf.transpose(x, self.axes)

    def torch_eval(self, x):
        #new_axes = [0]
        #for i in self.axes:
        #    new_axes.append(i+1)
        #new_axes = tuple(new_axes)
        return x.permute(*self.axes)


class Cast(MultiFrameworkMethod):
    def __init__(self, dtype='float32'):
        self.dtype = dtype

    def numpy_call(self, x):
        np_dtype = getattr(np, self.dtype)
        return x.astype(np_dtype)

    def tf_call(self, x):
        import tensorflow as tf
        tf_dtype = getattr(tf, self.dtype)
        return tf.cast(tf_dtype)

    def torch_call(self, x):
        import torch
        torch_dtype = getattr(torch, self.dtype)
        return x.to(torch_dtype)


class Select(Method):
    def __init__(self, idxs):
        self.idxs = idxs

    def __call__(self, x):
        result = x
        for idx in self.idxs:
            result = result[idx]
        return result
