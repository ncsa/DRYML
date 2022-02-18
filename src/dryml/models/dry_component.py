from dryml.dry_object import DryObject


class DryComponent(DryObject):
    """
    A Type for an ML component
    """

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()
