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

    def version(self):
        return 1

    def __eq__(self, rhs):
        return self.i == rhs.i

class HelloObject(dryml.DryObject):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_message(self):
        raise RuntimeError("Not implemented for this class")

class HelloStr(HelloObject):
    def __init__(self, msg:str="Test", **kwargs):
        self.str_message = msg
        dry_kwargs = {
            'msg': msg,
        }
        super().__init__(
            dry_kwargs=dry_kwargs,
            **kwargs
        )

    def get_message(self):
        return f"Hello! {self.str_message}"

class HelloInt(HelloObject):
    def __init__(self, msg:int=1, **kwargs):
        self.int_msg = msg
        dry_kwargs = {
            'msg': msg,
        }
        super().__init__(
            dry_kwargs=dry_kwargs,
            **kwargs
        )

    def get_message(self):
        return f"Hello! {self.int_msg}"
