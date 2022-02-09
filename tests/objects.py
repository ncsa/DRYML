import dryml


class HelloObject(dryml.DryObject):
    def __init__(self, **kwargs):
        pass

    def load_object_imp(self, file) -> bool:
        return super().load_object_imp(file)

    def save_object_imp(self, file) -> bool:
        return super().save_object_imp(file)

    def get_message(self):
        raise RuntimeError("Not implemented for this class")


class HelloStr(HelloObject):
    def __init__(self, msg: str = "Test", **kwargs):
        self.str_message = msg

    def load_object_imp(self, file) -> bool:
        return True

    def save_object_imp(self, file) -> bool:
        return True

    def get_message(self):
        return f"Hello! {self.str_message}"


class HelloInt(HelloObject):
    def __init__(self, msg: int = 1, **kwargs):
        self.int_msg = msg

    def load_object_imp(self, file) -> bool:
        return True

    def save_object_imp(self, file) -> bool:
        return True

    def get_message(self):
        return f"Hello! {self.int_msg}"


class TestBase(dryml.DryObject):
    def __init__(self, *args, base_msg: str = "base", **kwargs):
        self.base_msg = base_msg

    def load_object_imp(self, file) -> bool:
        return super().load_object_imp(file)

    def save_object_imp(self, file) -> bool:
        return super().save_object_imp(file)


class TestClassA(TestBase):
    def __init__(self, *args, item=[32], **kwargs):
        pass

    def load_object_imp(self, file) -> bool:
        return True

    def save_object_imp(self, file) -> bool:
        return True


class TestClassA2(TestBase):
    def __init__(self, *args, item=[32], **kwargs):
        pass

    def load_object_imp(self, file) -> bool:
        return True

    def save_object_imp(self, file) -> bool:
        return True


class TestClassB(TestBase):
    def __init__(self, layers, *args, **kwargs):
        pass

    def load_object_imp(self, file) -> bool:
        return True

    def save_object_imp(self, file) -> bool:
        return True


class HelloComponent(dryml.DryComponent):
    def __init__(self, *args, msg="test", **kwargs):
        pass


class HelloComponentB(dryml.DryComponent):
    def __init__(self, *args, msg="test", **kwargs):
        pass
