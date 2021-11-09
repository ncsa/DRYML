import dryml

class SimpleObject(dryml.DryObject):
    def __init__(self, i=0):
        self.i = i
        dry_kwargs = dryml.DryConfig({
            'i': i
        })
        super().__init__(
            dry_kwargs=dry_kwargs
        )

    def version(self):
        return 1

    def __eq__(self, rhs):
        return self.i == rhs.i
