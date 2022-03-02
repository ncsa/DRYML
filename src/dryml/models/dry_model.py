from dryml.dry_object import DryObject
from dryml.data.dry_data import DryData
from dryml.context import cls_method_compute


@cls_method_compute('eval')
class DryModel(DryObject):
    def eval(self, X: DryData, *args, **kwargs):
        raise NotImplementedError()
