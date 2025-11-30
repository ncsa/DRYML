from copy import deepcopy
from dryml.core2.object import Object
from dryml.core2.utils.stable_hash import _stable_leaf_bytes


class TestClassA(Object):
    def __init__(self, A):
        self.A = A


class DeepcopyAware:
    def __init__(self, val):
        self.val = val
        self.counter = 0

    def __deepcopy__(self, memo):
        cls = type(self)
        new = cls.__new__(cls)
        memo[id(self)] = new
        for k, v in self.__dict__.items():
            setattr(new, k, deepcopy(v, memo))
        new.counter += 1
        return new

    def __stable_leaf_bytes__(self):
        return _stable_leaf_bytes(self.val)
