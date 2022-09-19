from dryml.data.dataset import Dataset
from dryml.data.util import nested_batcher, nested_unbatcher, \
    nested_flatten
from dryml.utils import is_iterator
from dryml.data.util import taker, skiper
import numpy as np
from typing import Callable


class NumpyDataset(Dataset):
    """
    A Numpy based dataset based on a list of numpy elements
    """

    def __init__(
            self, data, indexed=False,
            supervised=False, batch_size=None, size=None):

        if type(data) is np.ndarray or type(data) is tuple:
            data_size = len(data)
            if type(data) is tuple:
                size_set = set(map(lambda d: len(d), nested_flatten(data)))
                if len(size_set) > 1:
                    raise ValueError(
                        "nested elements have different numbers of elements!")
                data_size = size_set.pop()
            super().__init__(
                indexed=indexed, supervised=supervised,
                batch_size=data_size)

            self._data_gen = lambda: [data]
            self.size = 1

        elif callable(data):
            # We have a method which is supposed to yield
            # A generator.
            super().__init__(
                indexed=indexed, supervised=supervised,
                batch_size=batch_size)
            self._data_gen = data
            if size is None:
                size = np.nan
            self.size = size

        elif is_iterator(data):
            # Can't use consumable iterator
            raise TypeError(
                "Can't use a consumable like an iterator or generator. "
                "Pass a callable which produces it instead.")
        else:
            df_test = False
            try:
                import pandas as pd
                if type(data) is pd.core.frame.DataFrame:
                    df_test = True
            except ImportError:
                pass
            if df_test:
                if indexed is False:
                    super().__init__(
                        indexed=indexed, supervised=supervised,
                        batch_size=len(data))
                    self._data_gen = lambda: [data.to_numpy()]
                    self.size = 1
                elif indexed is True:
                    super().__init__(
                        indexed=indexed, supervised=supervised,
                        batch_size=len(data))
                    self._data_gen = lambda: [(data.index.to_numpy(),
                                              data.to_numpy())]
                    self.size = 1
            elif type(data) is list:
                data_size = len(data)
                super().__init__(
                    indexed=indexed, supervised=supervised,
                    batch_size=batch_size)

                self._data_gen = lambda: data
                if size is None:
                    self.size = data_size
                else:
                    if size != data_size:
                        ValueError("Detected incorrect dataset size")
                    self.size = size
            else:
                super().__init__(
                    indexed=indexed, supervised=supervised,
                    batch_size=batch_size)

                self._data_gen = lambda: data
                if size is None:
                    self.size = np.nan
                else:
                    self.size = size

    def as_indexed(self, start=0) -> Dataset:
        """
        If not already indexed, return a version of this dataset
        which is indexed.
        """
        if self.indexed:
            return self
        else:
            if not self.batched:
                def enumerate_dataset(gen_func, start=0):
                    i = start
                    it = iter(gen_func())
                    while True:
                        try:
                            d = next(it)
                        except StopIteration:
                            return
                        yield (i, d)
                        i += 1

                return NumpyDataset(
                    lambda: enumerate_dataset(self.data_gen, start=start),
                    indexed=True,
                    supervised=self.supervised,
                    batch_size=self.batch_size,
                    size=self.size)
            else:
                def enumerate_dataset(gen_func, start=0):
                    it = iter(gen_func())
                    i = start
                    while True:
                        try:
                            d = next(it)
                        except StopIteration:
                            return
                        from dryml.data.util import get_data_batch_size
                        batch_size = get_data_batch_size(d)
                        idx = np.array(list(range(i, i+batch_size)))
                        i += batch_size
                        yield (idx, d)

                return NumpyDataset(
                    lambda: enumerate_dataset(self.data_gen, start=start),
                    indexed=True,
                    supervised=self.supervised,
                    batch_size=self.batch_size,
                    size=self.size)

    @property
    def data_gen(self):
        """
        Get a function which returns a generator which yields the dataset
        """
        return self._data_gen

    def data(self):
        """
        Get the internal dataset
        """
        return self.data_gen()

    def batch(self, batch_size=32, drop_remainder=True) -> Dataset:
        """
        Batch this data
        """
        if self.batched:
            if self.batch_size != batch_size:
                return self.unbatch().batch(batch_size=batch_size)
            else:
                return self
        else:
            return NumpyDataset(
                lambda: nested_batcher(
                     self.data_gen,
                     batch_size,
                     lambda e: np.stack(e, axis=0),
                     drop_remainder=drop_remainder),
                indexed=self.indexed,
                supervised=self.supervised,
                batch_size=batch_size,
                size=self.size)

    def unbatch(self) -> Dataset:
        """
        Unbatch this data
        """
        if not self.batched:
            return self
        else:
            return NumpyDataset(
                lambda: nested_unbatcher(self.data_gen),
                indexed=self.indexed,
                supervised=self.supervised,
                size=self.size)

    def map(self, func: Callable = None) -> Dataset:
        """
        Map a function across all elements of a dataset
        """

        return NumpyDataset(
            lambda: map(func, self),
            indexed=self.indexed,
            supervised=self.supervised,
            batch_size=self.batch_size,
            size=self.size)

    def take(self, n):
        """
        Take only a specific number of examples
        """

        new_size = self.size
        if new_size is np.nan:
            new_size = n
        elif new_size is np.inf:
            new_size = n
        else:
            if new_size > n:
                new_size = n

        return NumpyDataset(
            lambda: taker(self.data_gen, n),
            indexed=self.indexed,
            supervised=self.supervised,
            batch_size=self.batch_size,
            size=new_size,
            )

    def skip(self, n):
        """
        Skip a specific number of examples
        """

        new_size = self.size
        if new_size is not np.nan and new_size is not np.inf:
            if n > new_size:
                new_size = 0
            else:
                new_size -= n

        return NumpyDataset(
            lambda: skiper(self.data_gen, n),
            indexed=self.indexed,
            supervised=self.supervised,
            batch_size=self.batch_size,
            size=new_size)

    def __len__(self):
        """
        Get length of dataset. Will return Infinite if infinite,
        and unknown if it can't be determined.
        """
        return self.size

    def numpy(self):
        return self

    def tf(self):
        from dryml.data.tf import TFDataset
        import dryml.data.util as util
        import tensorflow as tf

        # Heuristic to determine output_signature
        peek_data = self.peek()

        def get_numpy_array_spec(e):
            e_tens = tf.constant(e)
            e_spec = tf.TensorSpec.from_tensor(e_tens)
            if self.batched:
                # We need to remove the first shape number
                # Since this data is batched
                list_shape = list(e_spec.shape)
                list_shape[0] = None
                e_spec = tf.TensorSpec(
                    tf.TensorShape(list_shape),
                    e_spec.dtype)
            return e_spec
        peek_data_signature = util.nested_apply(
            peek_data, get_numpy_array_spec)

        # Create tf dataset
        dataset = tf.data.Dataset.from_generator(
            lambda: self.data_gen(),
            output_signature=peek_data_signature)

        return TFDataset(
            dataset,
            indexed=self.indexed,
            supervised=self.supervised,
            batch_size=self.batch_size,
            size=self.size)

    def torch(self):
        from dryml.data.torch import TorchIterableDatasetWrapper, TorchDataset
        import torch

        dataset = self.map_el(lambda el: torch.tensor(el))

        # Create torch dataset
        ds = TorchIterableDatasetWrapper(
            dataset.data_gen)

        return TorchDataset(
            ds,
            indexed=self.indexed,
            supervised=self.supervised,
            batch_size=self.batch_size,
            size=self.size)
