from dryml.data.dry_data import DryData, NotIndexedError, \
    NotSupervisedError
from dryml.data.util import nested_batcher, nested_unbatcher
import numpy as np
from typing import Callable


class NumpyDataset(DryData):
    """
    A Numpy based dataset based on a list of numpy elements
    """

    def __init__(
            self, data, indexed=False,
            supervised=False, batch_size=None):

        if type(data) is np.ndarray:
            super().__init__(
                indexed=indexed, supervised=supervised,
                batch_size=len(data))

            self.data_gen = [data]
            return

        try:
            import pandas as pd
            if type(data) is pd.core.frame.DataFrame:
                if indexed is False:
                    super().__init__(
                        indexed=indexed, supervised=supervised,
                        batch_size=len(data))
                    self.data_gen = [data.to_numpy()]
                elif indexed is True:
                    super().__init__(
                        indexed=indexed, supervised=supervised,
                        batch_size=len(data))
                    self.data_gen = [(data.index.to_numpy(), data.to_numpy())]
        except ImportError:
            pass

        else:
            super().__init__(
                indexed=indexed, supervised=supervised,
                batch_size=batch_size)

            self.data_gen = data

    def index(self):
        """
        If indexed, return the index of this dataset
        """
        if not self.indexed():
            raise NotIndexedError()

        return map(lambda t: t[0], self.data_gen)

    def as_indexed(self, start=0) -> DryData:
        """
        If not already indexed, return a version of this dataset
        which is indexed.
        """
        if self.indexed():
            return self
        else:
            if not self.batched():
                def enumerate_dataset(gen, start=0):
                    i = start
                    it = iter(gen)
                    while True:
                        try:
                            d = next(it)
                        except StopIteration:
                            return
                        yield (i, d)
                        i += 1

                return NumpyDataset(
                    enumerate_dataset(self.data_gen, start=start),
                    indexed=True,
                    supervised=self.supervised(),
                    batch_size=self.batch_size())
            else:
                def enumerate_dataset(gen, start=0):
                    it = iter(gen)
                    i = start
                    while True:
                        try:
                            d = next(it)
                        except StopIteration:
                            return
                        batch_size = len(d)
                        idx = np.array(list(range(i, i+batch_size)))
                        i += batch_size
                        yield (idx, d)

                return NumpyDataset(
                    enumerate_dataset(self.data_gen, start=start),
                    indexed=True,
                    supervised=self.supervised(),
                    batch_size=self.batch_size())

    def as_not_indexed(self):
        """
        Strip index from dataset
        """
        if not self.indexed():
            return self
        else:
            return NumpyDataset(
                map(lambda t: t[1], self.data_gen),
                indexed=False,
                supervised=self.supervised(),
                batch_size=self.batch_size())

    def as_not_supervised(self) -> DryData:
        """
        Strip supervised targets
        """

        if not self.supervised():
            return self
        else:
            if self.indexed():
                return NumpyDataset(
                    map(lambda i, xy: (i, xy[0]), self.data_gen),
                    indexed=self.indexed(),
                    supervised=False,
                    batch_size=self.batch_size())
            else:
                return NumpyDataset(
                    map(lambda x, y: x, self.data_gen),
                    indexed=self.indexed(),
                    supervised=False,
                    batch_size=self.batch_size())

    def intersect(self) -> DryData:
        """
        Intersect this dataset with another
        """
        raise NotImplementedError()

    def data(self):
        """
        Get the internal dataset
        """
        return self.data_gen

    def batch(self, batch_size=32) -> DryData:
        """
        Batch this data
        """
        if self.batched():
            return self
        else:
            return NumpyDataset(
                nested_batcher(self.data_gen, batch_size),
                indexed=self.indexed(),
                supervised=self.supervised(),
                batch_size=batch_size)

    def unbatch(self) -> DryData:
        """
        Unbatch this data
        """
        if not self.batched():
            return self
        else:
            return NumpyDataset(
                nested_unbatcher(self.data_gen),
                indexed=self.indexed(),
                supervised=self.supervised())

    def apply_X(self, func: Callable = None) -> DryData:
        """
        Apply a function to the X component of DryData
        """

        if self.indexed():
            if self.supervised():
                return NumpyDataset(
                    map(lambda i, xy: (i, (func(xy[0]), xy[1])),
                        self.data_gen),
                    indexed=self.indexed(),
                    supervised=self.supervised(),
                    batch_size=self.batch_size())
            else:
                return NumpyDataset(
                    map(lambda i, x: (i, func(x)),
                        self.data_gen),
                    indexed=self.indexed(),
                    supervised=self.supervised(),
                    batch_size=self.batch_size())
        else:
            if self.supervised():
                return NumpyDataset(
                    map(lambda x, y: (func(x), y),
                        self.data_gen),
                    indexed=self.indexed(),
                    supervised=self.supervised(),
                    batch_size=self.batch_size())
            else:
                return NumpyDataset(
                    map(lambda x: func(x),
                        self.data_gen),
                    indexed=self.indexed(),
                    supervised=self.supervised(),
                    batch_size=self.batch_size())

    def apply_Y(self, func=None) -> DryData:
        """
        Apply a function to the Y component of DryData
        """

        if not self.supervised():
            raise NotSupervisedError(
                "Can't apply a function to the Y component of "
                "non supervised dataset")

        if self.indexed():
            return NumpyDataset(
                map(lambda i, xy: (i, (xy[0], func(xy[1]))),
                    self.data_gen),
                indexed=self.indexed(),
                supervised=self.supervised(),
                batch_size=self.batch_size())
        else:
            return NumpyDataset(
                map(lambda x, y: (x, func(y)),
                    self.data_gen),
                indexed=self.indexed(),
                supervised=self.supervised(),
                batch_size=self.batch_size())

    def apply(self, func=None) -> DryData:
        """
        Apply a function to (X, Y)
        """

        if not self.supervised():
            raise NotSupervisedError(
                "Can't apply a function to the Y component of "
                "non supervised dataset")

        if self.indexed():
            return NumpyDataset(
                map(lambda i, xy: (i, func(*xy)),
                    self.data_gen),
                indexed=self.indexed(),
                supervised=self.supervised(),
                batch_size=self.batch_size())
        else:
            return NumpyDataset(
                map(lambda x, y: func(x, y),
                    self.data_gen),
                indexed=self.indexed(),
                supervised=self.supervised(),
                batch_size=self.batch_size())

    def __iter__(self):
        """
        Create iterator
        """

        return iter(self.data())

    def take(self, n):
        """
        Take only a specific number of examples
        """

        def taker(gen, n):
            i = 0
            it = iter(gen)
            while i < n:
                try:
                    yield next(it)
                    i += 1
                except StopIteration:
                    return
            return

        return NumpyDataset(
            taker(self.data_gen, n),
            indexed=self.indexed(),
            supervised=self.supervised(),
            batch_size=self.batch_size())

    def skip(self, n):
        """
        Skip a specific number of examples
        """

        def skiper(gen, n):
            i = 0
            it = iter(gen)
            while i < n:
                try:
                    next(it)
                    i += 1
                except StopIteration:
                    return
            while True:
                try:
                    yield next(it)
                except StopIteration:
                    return

        return NumpyDataset(
            skiper(self.data_gen, n),
            indexed=self.indexed(),
            supervised=self.supervised(),
            batch_size=self.batch_size())

    def __len__(self):
        """
        Get length of dataset. Will return Infinite if infinite,
        and unknown if it can't be determined.
        """
        raise NotImplementedError()

    def numpy(self):
        return self
