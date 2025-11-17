from dryml.core2.object import Lazy, Pickleable, UniqueID, Metadata
import os
from dryml.core2.util import pickler, unpickler


class HelloObject(UniqueID):
    def __init__(self, **kwargs):
        pass

    def get_message(self):
        raise RuntimeError("Not implemented for this class")


class HelloStr(HelloObject):
    def __init__(self, msg: str = "Test", **kwargs):
        self.str_message = msg

    def get_message(self):
        return f"Hello! {self.str_message}"


class HelloInt(HelloObject):
    def __init__(self, msg: int = 1, **kwargs):
        self.int_msg = msg

    def get_message(self):
        return f"Hello! {self.int_msg}"


class TestBase(UniqueID, Metadata):
    def __init__(self, *args, base_msg: str = "base", **kwargs):
        super().__init__(*args, **kwargs)
        self.base_msg = base_msg


class TestClassA(TestBase):
    def __init__(self, *args, item=[32], **kwargs):
        super().__init__(*args, **kwargs)


class TestClassA2(TestBase):
    def __init__(self, *args, item=[32], **kwargs):
        super().__init__(*args, **kwargs)


class TestClassB(TestBase):
    def __init__(self, layers, *args, **kwargs):
        super().__init__(*args, **kwargs)


class TestNest(Lazy):
    def __init__(self, A):
        self.A = A


class TestNest2(Lazy):
    def __init__(self, A=None):
        self.A = A


class TestNest3(Lazy):
    def __init__(self, *args, **kwargs):
        pass

    def __getitem__(self, key):
        if type(key) is str:
            return self.__kwargs__[key]
        elif type(key) is int:
            return self.__args__[key]
        else:
            raise KeyError()


class TestNest4(UniqueID):
    def __init__(self, A, **kwargs):
        super().__init__(**kwargs)
        self.A = A


class TestClassF1(UniqueID):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.val = None


class TestClassC(Lazy):
    def __init__(self, A, B=None):
        self.A = A
        self.B = B


class TestClassC2(Pickleable):
    def __init__(self, C):
        self.C = C
        self.data = 0

    def set_val(self, val):
        self.data = val


class TestClass1(Lazy):
    def __init__(self, x, *args, test=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.x = x
        self.test = test


class TestClass2(Lazy):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs


class TestClass3(Lazy):
    def __init__(self, *args):
        super().__init__()
        self.args = args


class TestClass4(UniqueID, Metadata):
    def __init__(self, x, *args, test=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.x = x
        self.test = test


class TestClass5(Lazy):
    def __init__(self, x, *args, test=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.x = x
        self.test = test


class TestLazy1(Lazy):
    def __init__(self, x):
        super().__init__()
        self.x = x


