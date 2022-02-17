from dryml.dry_object import DryObject


class DryModel(DryObject):
    def eval(self, X, *args, **kwargs):
        raise NotImplementedError()
