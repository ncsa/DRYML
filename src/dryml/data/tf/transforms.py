from dryml.models import DryTrainable
from dryml.data.dry_data import DryData
from dryml.data.tf.dataset import TFDataset
import tensorflow as tf
import inspect


class FuncXMap(DryTrainable):
    @staticmethod
    def from_function(func, *args, **kwargs):
        return FuncXMap(inspect.getsource(func), *args, **kwargs)

    def __init__(self, func_code, *args, **kwargs):
        # Evaluate passed function code
        lcls = {}
        exec(func_code, globals(), lcls)

        # Check for function definition
        if len(lcls) == 0:
            raise ValueError("Code defines no objects!")
        if len(lcls) > 1:
            raise ValueError("Code defines more than one object!")

        # Get newly defined object
        func = list(lcls.values())[0]

        if not callable(func):
            raise ValueError(
                "Function code doesn't contain a function definition!")

        self.func = func
        self.train_state = DryTrainable.trained

    def train(self, *args, **kwargs):
        # We can't train a plain function
        pass

    def eval(self, data: DryData, *args, **kwargs):
        return data.apply_X(func=self.func)


class FuncYMap(DryTrainable):
    @staticmethod
    def from_function(func, *args, **kwargs):
        return FuncYMap(inspect.getsource(func), *args, **kwargs)

    def __init__(self, func_code, *args, **kwargs):
        # Evaluate passed function code
        lcls = {}
        exec(func_code, globals(), lcls)

        # Check for function definition
        if len(lcls) == 0:
            raise ValueError("Code defines no objects!")
        if len(lcls) > 1:
            raise ValueError("Code defines more than one object!")

        # Get newly defined object
        func = list(lcls.values())[0]

        if not callable(func):
            raise ValueError(
                "Function code doesn't contain a function definition!")

        self.func = func
        self.train_state = DryTrainable.trained

    def train(self, *args, **kwargs):
        # We can't train a plain function
        pass

    def eval(self, data: DryData, *args, **kwargs):
        return data.apply_Y(func=self.func)


class FuncMap(DryTrainable):
    @staticmethod
    def from_function(func, *args, **kwargs):
        return FuncMap(inspect.getsource(func), *args, **kwargs)

    def __init__(self, func_code, *args, **kwargs):
        # Evaluate passed function code
        lcls = {}
        exec(func_code, globals(), lcls)

        # Check for function definition
        if len(lcls) == 0:
            raise ValueError("Code defines no objects!")
        if len(lcls) > 1:
            raise ValueError("Code defines more than one object!")

        # Get newly defined object
        func = list(lcls.values())[0]

        if not callable(func):
            raise ValueError(
                "Function code doesn't contain a function definition!")

        self.func = func
        self.train_state = DryTrainable.trained

    def train(self, *args, **kwargs):
        # We can't train a plain function
        pass

    def eval(self, data: DryData, *args, **kwargs):
        return data.apply(func=self.func)


class BestCat(DryTrainable):
    def __init__(self):
        self.train_state = DryTrainable.trained

    def train(self, *args, **kwargs):
        pass

    def eval(self, data: DryData, *args, **kwargs):
        if type(data) is not TFDataset:
            raise TypeError(
                f"Dataset of type {type(data)} not currently supported.")

        return data.apply_X(
            func=lambda image: tf.argmax(image, axis=-1))
