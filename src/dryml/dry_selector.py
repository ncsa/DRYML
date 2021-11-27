from dryml.dry_object import DryObject

class DrySelector(object):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(obj: DryObject):
        # Check whether an object matches this selector
        return True
