from dryml.dry_object import DryObject
from dryml.context import compute


class DryComponent(DryObject):
    """
    A Type for an ML component
    """

    @compute
    def __call__(self, *args, **kwargs):
        raise NotImplementedError()
