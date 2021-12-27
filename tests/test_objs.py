import dryml


class SimpleObject(dryml.DryObject):
    def __init__(self, i=0, **kwargs):
        self.i = i
        dry_kwargs = {
            'i': i
        }
        super().__init__(
            dry_kwargs=dry_kwargs,
            **kwargs
        )

    def load_object_imp(self, file) -> bool:
        return True

    def save_object_imp(self, file) -> bool:
        return True

    def version(self):
        return 2

    def __eq__(self, rhs):
        return self.i == rhs.i
