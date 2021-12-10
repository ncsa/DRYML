import dryml

class HelloObject(dryml.DryObject):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_object_imp(self, file) -> bool:
        return super().load_object_imp(file)

    def save_object_imp(self, file) -> bool:
        return super().save_object_imp(file)

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

    def load_object_imp(self, file) -> bool:
        return True

    def save_object_imp(self, file) -> bool:
        return True

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

    def load_object_imp(self, file) -> bool:
        return True

    def save_object_imp(self, file) -> bool:
        return True

    def get_message(self):
        return f"Hello! {self.int_msg}"

class TestBase(dryml.DryObject):
    def __init__(self, *args, base_msg:str="base", dry_args=None, dry_kwargs=None, **kwargs):
        dry_args = dryml.utils.init_arg_list_handler(dry_args)
        dry_kwargs = dryml.utils.init_arg_dict_handler(dry_kwargs)
        self.base_msg = base_msg
        dry_kwargs['base_msg'] = base_msg
        super().__init__(
            *args,
            dry_args=dry_args,
            dry_kwargs=dry_kwargs,
            **kwargs
        )

    def load_object_imp(self, file) -> bool:
        return super().load_object_imp(file)

    def save_object_imp(self, file) -> bool:
        return super().save_object_imp(file)

class TestClassA(TestBase):
    def __init__(self, *args, item=[32], dry_args=None, dry_kwargs=None, **kwargs):
        dry_args = dryml.utils.init_arg_list_handler(dry_args)
        dry_kwargs = dryml.utils.init_arg_dict_handler(dry_kwargs)
        dry_kwargs['item'] = item
        super().__init__(
            *args,
            dry_args=dry_args,
            dry_kwargs=dry_kwargs,
            **kwargs
        )

    def load_object_imp(self, file) -> bool:
        return True

    def save_object_imp(self, file) -> bool:
        return True

class TestClassB(TestBase):
    def __init__(self, layers, *args, dry_args=None, dry_kwargs=None, **kwargs):
        dry_args = dryml.utils.init_arg_list_handler(dry_args)
        dry_kwargs = dryml.utils.init_arg_dict_handler(dry_kwargs)
        dry_args.append(layers)
        super().__init__(
            *args,
            dry_args=dry_args,
            dry_kwargs=dry_kwargs,
            **kwargs
        )

    def load_object_imp(self, file) -> bool:
        return True

    def save_object_imp(self, file) -> bool:
        return True
