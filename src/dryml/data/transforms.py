from dryml.models import DryTrainable
from dryml.data.dry_data import DryData
import numpy as np


class BestCat(DryTrainable):
    def __init__(self):
        self.train_state = DryTrainable.trained

    def train(self, *args, train_spec=None, **kwargs):
        pass

    def eval(self, data: DryData, *args, **kwargs):
        if data.batched:
            data = data.batch()
        return data.numpy().apply_X(
            func=lambda x: np.argmax(x, axis=-1))


def flattener(x):
    return x.reshape([x.shape[0], -1])


class Flatten(DryTrainable):
    def __init__(self):
        self.train_state = DryTrainable.trained

    def train(self, *args, train_spec=None, **kwargs):
        pass

    def eval(self, data: DryData, *args, **kwargs):
        if not data.batched:
            data = data.batch()
        return data.numpy().apply_X(
            func=flattener)
