from dryml.object import Object
from dryml.data.dataset import Dataset
from dryml.context import cls_method_compute


@cls_method_compute('eval')
class DryModel(Object):
    def eval(self, X: Dataset, *args, **kwargs):
        raise NotImplementedError()
