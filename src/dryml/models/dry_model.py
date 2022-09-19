from dryml.object import Object
from dryml.data.dry_data import DryData
from dryml.context import cls_method_compute


@cls_method_compute('eval')
class DryModel(Object):
    def eval(self, X: DryData, *args, **kwargs):
        raise NotImplementedError()
