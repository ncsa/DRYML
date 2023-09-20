import dryml


class SimpleObject(dryml.Object):
    def __init__(self, i, **kwargs):
        self.i = i

    def version(self):
        return 2

    def __eq__(self, rhs):
        return self.i == rhs.i
