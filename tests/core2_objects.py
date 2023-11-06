from dryml.core2.object import Remember, UniqueID, Metadata, Serializable, Defer
import os
from dryml.core2.util import pickler, unpickler


class HelloObject(Serializable, UniqueID):
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


class TestBase(Serializable, UniqueID, Metadata):
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


class TestNest(Remember):
    def __init__(self, A):
        self.A = A


class TestNest2(Serializable):
    def __init__(self, A=None):
        self.A = A


class TestNest3(Serializable):
    def __init__(self, *args, **kwargs):
        pass

    def __getitem__(self, key):
        if type(key) is str:
            return self.__kwargs__[key]
        elif type(key) is int:
            return self.__args__[key]
        else:
            raise KeyError()


class TestNest4(Remember, UniqueID):
    def __init__(self, A, **kwargs):
        super().__init__(**kwargs)
        self.A = A


class TestClassF1(Remember, UniqueID):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.val = None


class TestClassC(Serializable):
    def __init__(self, A, B=None):
        self.A = A
        self.B = B


class TestClassC2(Serializable):
    def __init__(self, C):
        self.C = C
        self.data = 0

    def set_val(self, val):
        self.data = val

    def _save_to_dir_imp(self, dir: str):
        data_file = os.path.join(dir, 'data.pkl')
        with open(data_file, 'wb') as f:
            f.write(pickler(self.data))
        return True

    def _load_from_dir_imp(self, dir: str):
        data_file = os.path.join(dir, 'data.pkl')
        with open(data_file, 'rb') as f:
            self.data = unpickler(f.read())
        return True


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


class TestDefer1(Defer):
    def __init__(self, x):
        super().__init__()
        self.x = x
