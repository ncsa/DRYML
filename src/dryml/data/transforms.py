from dryml.models import DryTrainable
from dryml.data.dataset import Dataset
from dryml.data.util import nestize
from dryml.data.numpy_dataset import NumpyDataset
import numpy as np
from typing import Callable


class StaticTransform(DryTrainable):
    def __init__(self, mode='X'):
        self.train_state = DryTrainable.trained
        if mode not in ['all', 'X', 'Y']:
            raise ValueError(f"mode '{mode}' not supported.")
        self.mode = mode

    def train(self, *args, train_spec=None, **kwargs):
        pass

    def applier(self, data: Dataset, func: Callable):
        if self.mode == 'all':
            return data.apply(func)
        elif self.mode == 'X':
            return data.apply_X(func)
        elif self.mode == 'Y':
            return data.apply_Y(func)
        else:
            raise RuntimeError("Unknown mode!")

    def numpy_eval(self, data: NumpyDataset):
        raise NotImplementedError()

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
        if type(data) is NumpyDataset:
            return self.numpy_eval(data, *args, **kwargs)


class BestCat(StaticTransform):
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


class Flatten(StaticTransform):
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


class Transpose(StaticTransform):
    def __init__(self, axes=None):
        self.train_state = DryTrainable.trained
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


class Cast(StaticTransform):
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
