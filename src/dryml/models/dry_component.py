from dryml.dry_object import DryObject
from dryml.context import cls_method_compute


@cls_method_compute('__call__')
class DryComponent(DryObject):
    """
    A Type for an ML component
    """

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()
