from dryml import Object, Meta
from dryml.models import Trainable, Component
from dryml.models import TrainFunction as BaseTrainFunction
from dryml.data import Dataset


class Wrapper(Object):
    __dry_compute_context__ = 'tf'

    @Meta.collect_args
    @Meta.collect_kwargs
    def __init__(self, cls, *args, **kwargs):
        if type(cls) is not type:
            raise TypeError(
                f"Expected first argument to be type. Got {type(cls)}")
        self.cls = cls
        self.args = args
        self.kwargs = kwargs
        self.obj = None

    def compute_prepare_imp(self):
        self.obj = self.cls(*self.args, **self.kwargs)

    def compute_cleanup_imp(self):
        del self.obj
        self.obj = None


class TrainFunction(BaseTrainFunction):
    __dry_compute_context__ = 'tf'


class Model(Component):
    __dry_compute_context__ = 'tf'


class Trainable(Trainable):
    __dry_compute_context__ = 'tf'

    def eval(self, data: Dataset, *args, eval_batch_size=32, **kwargs):
        def eval_func(X):
            return self.model(X, *args, **kwargs)
        if data.batched:
            # We can execute the method directly on the data
            return data.tf() \
                       .apply_X(func=eval_func)
        else:
            # We first need to batch the data, then unbatch to leave
            # The dataset character unchanged.
            return data.tf() \
                       .batch(batch_size=eval_batch_size) \
                       .apply_X(func=eval_func) \
                       .unbatch()
