import dryml
import dryml.models
import zipfile
import pickle


class HelloObject(dryml.Object):
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


class TestBase(dryml.Object):
    def __init__(self, *args, base_msg: str = "base", **kwargs):
        self.base_msg = base_msg


class TestClassA(TestBase):
    def __init__(self, *args, item=[32], **kwargs):
        pass


class TestClassA2(TestBase):
    def __init__(self, *args, item=[32], **kwargs):
        pass


class TestClassB(TestBase):
    def __init__(self, layers, *args, **kwargs):
        pass


class HelloTrainable(dryml.models.Trainable):
    def __init__(self, *args, msg="test", **kwargs):
        pass


class HelloTrainableB(dryml.models.Trainable):
    def __init__(self, *args, msg="test", **kwargs):
        pass


class HelloTrainableC(dryml.models.Trainable):
    def __init__(self, A):
        self.A = A


class HelloTrainableD(dryml.models.Trainable):
    def __init__(self, A=None):
        self.A = A


class HelloTrainableE(dryml.models.Trainable):
    @dryml.Meta.collect_args
    @dryml.Meta.collect_kwargs
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __getitem__(self, key):
        if type(key) is str:
            return self.kwargs[key]
        elif type(key) is int:
            return self.args[key]
        else:
            raise KeyError(f"{key}")


class TestNest(dryml.Object):
    def __init__(self, A):
        self.A = A


class TestNest2(dryml.Object):
    def __init__(self, A=None):
        self.A = A


class TestNest3(dryml.Object):
    @dryml.Meta.collect_args
    @dryml.Meta.collect_kwargs
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __getitem__(self, key):
        if type(key) is str:
            return self.kwargs[key]
        elif type(key) is int:
            return self.args[key]
        else:
            raise KeyError()


class TestClassC(dryml.Object):
    def __init__(self, A, B=None):
        self.A = A
        self.B = B


class TestClassC2(dryml.Object):
    def __init__(self, C):
        self.C = C
        self.data = 0

    def set_val(self, val):
        self.data = val

    def save_object_imp(self, file: zipfile.ZipFile):
        with file.open('data.pkl', 'w') as f:
            f.write(dryml.core.utils.pickler(self.data))
        return True

    def load_object_imp(self, file: zipfile.ZipFile):
        with file.open('data.pkl', 'r') as f:
            self.data = dryml.core.utils.unpickler(f.read())
        return True


class TestClassD1(dryml.Object):
    pass


class TestClassD2(TestClassD1):
    pass


class TestClassD3(TestClassD2):
    @dryml.Meta.collect_kwargs
    def __init__(self, A, **kwargs):
        assert 'dry_id' not in kwargs
        self.A = A
        self.mdl_kwargs = kwargs


class TestClassE(dryml.Object):
    def __init__(self):
        pass

    def set_val(self, val):
        self.data = val

    def save_compute_imp(self, file: zipfile.ZipFile) -> bool:
        with file.open('data.pkl', 'w') as f:
            f.write(dryml.core.utils.pickler(self.data))
        return True

    def load_compute_imp(self, file: zipfile.ZipFile) -> bool:
        with file.open('data.pkl', 'r') as f:
            self.data = dryml.core.utils.unpickler(f.read())
        return True


class TestClassF1(dryml.Object):
    def __init__(self):
        self.val = None


class TestClassG1(dryml.Object):
    def __init__(self, val):
        pass
