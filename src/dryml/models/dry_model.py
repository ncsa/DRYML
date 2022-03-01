from dryml.dry_object import DryObject
from dryml.data.dry_data import DryData
from dryml.context import compute


class DryModel(DryObject):
    @compute
    def eval(self, X: DryData, *args, **kwargs):
        raise NotImplementedError()
