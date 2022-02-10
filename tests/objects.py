import dryml


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


class HelloComponent(dryml.DryComponent):
    def __init__(self, *args, msg="test", **kwargs):
        pass


class HelloComponentB(dryml.DryComponent):
    def __init__(self, *args, msg="test", **kwargs):
        pass
