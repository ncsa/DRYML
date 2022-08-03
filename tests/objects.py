import dryml
import zipfile
import pickle


class HelloObject(dryml.DryObject):
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


class TestBase(dryml.DryObject):
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


class HelloTrainable(dryml.models.DryTrainable):
    def __init__(self, *args, msg="test", **kwargs):
        pass


class HelloTrainableB(dryml.models.DryTrainable):
    def __init__(self, *args, msg="test", **kwargs):
        pass


class HelloTrainableC(dryml.models.DryTrainable):
    def __init__(self, A):
        self.A = A


class HelloTrainableD(dryml.models.DryTrainable):
    def __init__(self, A=None):
        self.A = A


class TestNest(dryml.DryObject):
    def __init__(self, A):
        self.A = A


class TestNest2(dryml.DryObject):
    def __init__(self, A=None):
        self.A = A


class TestClassC(dryml.DryObject):
    def __init__(self, A, B=None):
        self.A = A
        self.B = B


class TestClassC2(dryml.DryObject):
    def __init__(self, C):
        self.C = C
        self.data = 0

    def set_val(self, val):
        self.data = val

    def save_object_imp(self, file: zipfile.ZipFile):
        with file.open('data.pkl', 'w') as f:
            f.write(dryml.utils.pickler(self.data))
        return True

    def load_object_imp(self, file: zipfile.ZipFile):
        with file.open('data.pkl', 'r') as f:
            self.data = pickle.loads(f.read())
        return True


class TestClassD1(dryml.DryObject):
    pass


class TestClassD2(TestClassD1):
    pass


class TestClassD3(TestClassD2):
    @dryml.DryMeta.collect_kwargs
    def __init__(self, A, **kwargs):
        assert 'dry_id' not in kwargs
        self.A = A
        self.mdl_kwargs = kwargs


class TestClassE(dryml.DryObject):
    def __init__(self):
        pass

    def set_val(self, val):
        self.data = val

    def save_compute_imp(self, file: zipfile.ZipFile) -> bool:
        print(f"save_compute_imp called for {__class__}")
        with file.open('data.pkl', 'w') as f:
            f.write(dryml.utils.pickler(self.data))
        return True

    def load_compute_imp(self, file: zipfile.ZipFile) -> bool:
        print(f"load_compute_imp called for {__class__}")
        with file.open('data.pkl', 'r') as f:
            self.data = pickle.loads(f.read())
        return True


class TestClassF1(dryml.DryObject):
    def __init__(self):
        self.val = None
