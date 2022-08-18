from dryml import DryObject
from dryml.models import DryTrainable, DryComponent
from dryml.models import TrainFunction as BaseTrainFunction
from dryml.data import DryData


class ObjectWrapper(DryObject):
    __dry_compute_context__ = 'tf'

    def __init__(self, obj_cls, obj_args=(), obj_kwargs={}):
        self.obj_args = obj_args
        self.obj_kwargs = obj_kwargs
        self.obj_cls = obj_cls
        self.obj = None

    def compute_prepare_imp(self):
        self.obj = self.obj_cls(*self.obj_args, **self.obj_kwargs)

    def compute_cleanup_imp(self):
        del self.obj
        self.obj = None


class TrainFunction(BaseTrainFunction):
    __dry_compute_context__ = 'tf'


class Model(DryComponent):
    __dry_compute_context__ = 'tf'


class Trainable(DryTrainable):
    __dry_compute_context__ = 'tf'

    def __init__(self):
        pass

    def eval(self, data: DryData, *args, eval_batch_size=32, **kwargs):
        if data.batched:
            # We can execute the method directly on the data
            return data.tf().apply_X(
                func=lambda X: self.model(X, *args, **kwargs))
        else:
            # We first need to batch the data, then unbatch to leave
            # The dataset character unchanged.
            return data.tf().batch(batch_size=eval_batch_size) \
                       .apply_X(
                            func=lambda X: self.model(X, *args, **kwargs)) \
                       .unbatch()
