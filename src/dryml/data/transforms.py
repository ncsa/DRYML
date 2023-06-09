from dryml.models import Trainable
from dryml.data.dataset import Dataset
from dryml.data.util import nestize, function_inspection, \
    promote_function, func_source_extract
from dryml.data.numpy_dataset import NumpyDataset
import numpy as np
from typing import Callable


class StaticTransform(Trainable):
    def __init__(self, mode='X'):
        self.train_state = Trainable.trained
        if mode not in ['all', 'X', 'Y']:
            raise ValueError(f"mode '{mode}' not supported.")
        self.mode = mode

    def train(self, *args, train_spec=None, **kwargs):
        pass

    def applier(
            self,
            data: Dataset,
            func: Callable,
            func_args=(),
            func_kwargs={}):
        # Apply function to data
        #
        # Args:
        #  data: Dataset to apply function to.
        #  func: Function to apply.
        #  func_args: Additional arguments to pass to func.
        #  func_kwargs: Additional keyword arguments to pass to func.

        func_inspect = function_inspection(func)

        if func_inspect["n_args"] == 0:
            raise ValueError(
                "Function must take at least one explicit argument! "
                "function signature: " + func_inspect["signature"])

        if self.mode == 'all':
            if func_inspect["n_args"] == 1:
                # promote to two arguments
                func = promote_function(func)
            elif func_inspect["n_args"] > 2:
                raise ValueError(
                    "Function must take at most two explicit arguments! "
                    "function signature: " + func_inspect["signature"])
            return data.apply(
                func,
                func_args=func_args,
                func_kwargs=func_kwargs)
        elif self.mode == 'X':
            if func_inspect["n_args"] > 1:
                raise ValueError(
                    "Function must take at most one explicit argument! "
                    "function signature: " + func_inspect["signature"])
            return data.apply_X(
                func,
                func_args=func_args,
                func_kwargs=func_kwargs)
        elif self.mode == 'Y':
            if func_inspect["n_args"] > 1:
                raise ValueError(
                    "Function must take at most one explicit argument! "
                    "function signature: " + func_inspect["signature"])
            return data.apply_Y(
                func,
                func_args=func_args,
                func_kwargs=func_kwargs)
        else:
            raise RuntimeError("Unknown mode!")

    def numpy_eval(self, data, *args, **kwargs):
        raise NotImplementedError()

    def eval(self, data: Dataset, *args, **kwargs):
        raise NotImplementedError()


class FrameworkTransform(StaticTransform):
    # A version of StaticTransform with different function
    # definitions for each framework.

    def eval(self, data: Dataset, *args, **kwargs):
        # Special case for Tensorflow datasets
        try:
            import dryml.data.tf
            if type(data) is dryml.data.tf.TFDataset:
                if hasattr(self, 'tf_eval'):
                    return self.tf_eval(data, *args, **kwargs)
        except ImportError:
            pass

        # Special case for Torch datasets
        try:
            import dryml.data.torch
            if type(data) is dryml.data.torch.TorchDataset:
                if hasattr(self, 'torch_eval'):
                    return self.torch_eval(data, *args, **kwargs)
        except ImportError:
            pass

        # Fallback, try to cast first to numpy then apply.
        if type(data) is not NumpyDataset:
            data = data.numpy()

        return self.numpy_eval(data, *args, **kwargs)


class BestCat(FrameworkTransform):
    def numpy_eval(self, data, *args, **kwargs):
        return self.applier(
            data,
            lambda x: np.argmax(x, axis=-1))

    def tf_eval(self, data, *args, **kwargs):
        import tensorflow as tf
        return self.applier(
            data,
            lambda x: tf.argmax(x, axis=-1))

    def torch_eval(self, data, *args, **kwargs):
        import torch
        return self.applier(
            data,
            lambda x: torch.argmax(x, dim=-1))


class Flatten(FrameworkTransform):
    def numpy_eval(self, data, *args, **kwargs):
        if data.batched:
            return self.applier(
                data,
                lambda x: x.reshape([x.shape[0], -1]))
        else:
            return self.applier(
                data,
                lambda x: x.flatten())

    def tf_eval(self, data, *args, **kwargs):
        import tensorflow as tf
        if data.batched:
            return self.applier(
                data,
                lambda x: tf.reshape(x, [tf.shape(x)[0], -1]))
        else:
            return self.applier(
                data,
                lambda x: tf.reshape(x, [-1]))

    def torch_eval(self, data, *args, **kwargs):
        import torch
        if data.batched:
            return self.applier(
                data,
                lambda x: torch.reshape(x, (torch.shape[0], -1)))
        else:
            return self.applier(
                data,
                lambda x: torch.reshape(x, (-1,)))


class Transpose(FrameworkTransform):
    def __init__(self, axes=None):
        self.train_state = Trainable.trained
        self.axes = axes

    def numpy_eval(self, data, *args, **kwargs):
        if data.batched:
            # Move axes up by one
            new_axes = []
            for i in self.axes:
                new_axes.append(i+1)

            return self.applier(
                data,
                lambda x: np.transpose(x, [0]+new_axes))
        else:
            return self.applier(
                data,
                lambda x: np.transpose(x, self.axes))

    def tf_eval(self, data, *args, **kwargs):
        import tensorflow as tf
        if data.batched:
            new_axes = []
            for i in self.axes:
                new_axes.append(i+1)
            new_axes = tuple(new_axes)

            return self.applier(
                data,
                lambda x: tf.transpose(x, (0,)+new_axes))
        else:
            return self.applier(
                data,
                lambda x: tf.transpose(x, self.axes))

    def torch_eval(self, data, *args, **kwargs):
        if data.batched:
            new_axes = []
            for i in self.axes:
                new_axes.append(i+1)

            return self.applier(
                data,
                lambda x: x.permute(*new_axes))
        else:
            return self.applier(
                data,
                lambda x: x.permute(*self.axes))


class Cast(FrameworkTransform):
    def __init__(self, dtype='float32'):
        self.dtype = dtype

    def numpy_eval(self, data, *args, **kwargs):
        np_dtype = getattr(np, self.dtype)

        def caster(x):
            return x.astype(np_dtype)

        return self.applier(
            data,
            nestize(caster))

    def tf_eval(self, data, *args, **kwargs):
        import tensorflow as tf
        tf_dtype = getattr(tf, self.dtype)

        def caster(x):
            return tf.cast(x, tf_dtype)

        return self.applier(
            data,
            nestize(caster))

    def torch_eval(self, data, *args, **kwargs):
        import torch
        torch_dtype = getattr(torch, self.dtype)

        def caster(x):
            return x.to(torch_dtype)

        return self.applier(
            data,
            nestize(caster))


class FuncTransform(StaticTransform):
    @classmethod
    def from_function(
            cls,
            func,
            func_args=(),
            func_kwargs={},
            framework=None):
        return FuncTransform(
            func_source_extract(func),
            func_args=func_args,
            func_kwargs=func_kwargs,
            framework=framework)

    def __init__(
            self,
            func_code,
            func_args=(),
            func_kwargs={},
            framework=None):
        # Save any arguments which will be passed after the
        # data to the function
        self.args = func_args
        self.kwargs = func_kwargs
        if framework is not None:
            if framework not in ['tf', 'torch', 'numpy']:
                raise ValueError(
                    "Framework must be one of 'tf', 'torch', 'numpy' or None!")
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
        self.train_state = Trainable.trained

    def eval(self, data: Dataset, *args, **kwargs):
        if self.framework is not None:
            if self.framework == 'tf':
                data = data.tf()
            elif self.framework == 'torch':
                data = data.torch()
            elif self.framework == 'numpy':
                data = data.numpy()
        return self.applier(
            data,
            self.func,
            func_args=self.args,
            func_kwargs=self.kwargs)
