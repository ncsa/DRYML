from __future__ import annotations
from typing import Callable


class NotIndexedError():
    pass


class NotSupervisedError():
    pass


class DryData():
    """
    A Simple wrapper class to house Dry Level data operations
    """

    def __init__(self, indexed=False, supervised=False,
                 batch_size=None):
        self._indexed = indexed
        self._supervised = supervised
        self._batch_size = batch_size

    @property
    def indexed(self) -> bool:
        """
        Indicate whether this dataset is indexed.
        """
        return self._indexed

    def index(self):
        """
        If indexed, return the index of this dataset
        """
        raise NotImplementedError()

    def as_indexed(self, start=0) -> DryData:
        """
        If not already indexed, return a version of this dataset
        which is indexed.
        """
        raise NotImplementedError()

    def as_not_indexed(self) -> DryData:
        """
        Strip index from dataset
        """
        raise NotImplementedError()

    @property
    def supervised(self) -> bool:
        """
        Indicate whether this dataset is supervised (has targets as well)
        """
        return self._supervised

    def as_not_supervised(self) -> DryData:
        """
        Strip supervised targets
        """
        raise NotImplementedError()

    def intersect(self) -> DryData:
        """
        Intersect this dataset with another
        """
        raise NotImplementedError()

    def data(self):
        """
        Return the backing dataset
        """
        raise NotImplementedError()

    @property
    def batched(self) -> bool:
        """
        Indicate whether this data has been batched
        """
        if self._batch_size is not None:
            return True
        else:
            return False

    @property
    def batch_size(self):
        """
        Get the batch size
        """

        return self._batch_size

    def batch(self, batch_size=32) -> DryData:
        """
        Batch this data
        """
        raise NotImplementedError()

    def unbatch(self) -> DryData:
        """
        Unbatch this data
        """
        raise NotImplementedError()

    def apply_X(self, func: Callable = None) -> DryData:
        """
        Apply a function to the X component of DryData
        """
        raise NotImplementedError()

    def apply_Y(self, func: Callable = None) -> DryData:
        """
        Apply a function to the Y component of DryData
        """
        raise NotImplementedError()

    def apply(self, func: Callable = None) -> DryData:
        """
        Apply a function to (X, Y)
        """
        raise NotImplementedError()

    def __iter__(self):
        """
        Create an iterator
        """
        raise NotImplementedError()

    def take(self, n):
        """
        Take only a specific number of examples
        """
        raise NotImplementedError()

    def skip(self, n):
        """
        Skip a specific number of examples
        """
        raise NotImplementedError()

    def __len__(self):
        """
        Get length of dataset. Will return Infinite if infinite,
        and unknown if it can't be determined.
        """
        raise NotImplementedError()

    def numpy(self):
        """
        Create a Numpy Dataset from this dataset.
        """

        raise NotImplementedError()

    def collect(self):
        """
        Collect all data from the dataset
        """

        result = []
        for el in self:
            result.append(el)
        return result

    def peek(self):
        """
        Get the first element to have a look
        """

        return self.take(1).collect()[0]
