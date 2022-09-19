from dryml.object import Object
from dryml.context import cls_method_compute


@cls_method_compute('__call__')
class Component(Object):
    """
    A Type for an ML component
    """

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()


class TrainFunction(Component):
    def __init__(self):
        self.train_args = ()
        self.train_kwargs = {}

    def __call__(
            self, trainable, train_data,
            train_spec=None, train_callbacks=[]):
        raise NotImplementedError("method must be implemented in a subclass")
