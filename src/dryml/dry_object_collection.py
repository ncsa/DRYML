from dryml.dry_object import DryObject


class DryObjectCollection(DryObject):
    def __init__(
            self, *args, dry_args=None,
            dry_kwargs=None, **kwargs):
        self.objects = []
