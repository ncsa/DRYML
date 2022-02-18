from dryml.dry_object import DryObject
from dryml.data.dry_data import DryData


class DryModel(DryObject):
    def eval(self, X: DryData, *args, **kwargs):
        raise NotImplementedError()
