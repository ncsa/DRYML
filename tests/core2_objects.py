from dryml.core2.core2 import Remember, Defer, UniqueID, Metadata, Serializable


class TestClass1(Remember):
    def __init__(self, x, *args, test=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.x = x
        self.test = test


class TestClass2(Remember):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs


class TestClass3(Remember):
    def __init__(self, *args):
        super().__init__()
        self.args = args


class TestClass4(Remember, UniqueID, Metadata):
    def __init__(self, x, *args, test=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.x = x
        self.test = test


class TestClass5(Serializable):
    def __init__(self, x, *args, test=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.x = x
        self.test = test
