from dryml.models import Trainable
from dryml.data.dataset import Dataset
import tensorflow as tf
import inspect


def func_source_extract(func):
    # Get the source code
    code_string = inspect.getsource(func)

    # Possibly strip leading spaces
    code_lines = code_string.split('\n')

    init_strip = code_lines[0]

    for line in code_lines[1:]:
        i = 0
        while i < len(init_strip) and \
                i < len(line) and \
                init_strip[i] == line[i]:
            i += 1
        if i == len(init_strip) or \
                i == len(line):
            continue

        init_strip = init_strip[:i]

    if len(init_strip) > 0:
        code_lines = list(map(
            lambda l: l[len(init_strip):],
            code_lines))

    return '\n'.join(code_lines)


class FuncXMap(Trainable):
    @staticmethod
    def from_function(func, *args, **kwargs):
        return FuncXMap(func_source_extract(func), *args, **kwargs)

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
        self.train_state = Trainable.trained

    def train(self, *args, train_spec=None, **kwargs):
        # We can't train a plain function
        pass

    def eval(self, data: Dataset, *args, **kwargs):
        return data.tf().apply_X(func=self.func)


class FuncYMap(Trainable):
    @staticmethod
    def from_function(func, *args, **kwargs):
        return FuncYMap(func_source_extract(func), *args, **kwargs)

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
        self.train_state = Trainable.trained

    def train(self, *args, train_spec=None, **kwargs):
        # We can't train a plain function
        pass

    def eval(self, data: Dataset, *args, **kwargs):
        return data.tf().apply_Y(func=self.func)


class FuncMap(Trainable):
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
        self.train_state = Trainable.trained

    def train(self, *args, train_spec=None, **kwargs):
        # We can't train a plain function
        pass

    def eval(self, data: Dataset, *args, **kwargs):
        return data.tf().apply(func=self.func)


class BestCat(Trainable):
    def __init__(self):
        self.train_state = Trainable.trained

    def train(self, *args, train_spec=None, **kwargs):
        pass

    def eval(self, data: Dataset, *args, **kwargs):
        if not data.batched:
            data = data.batch()
        return data.tf().apply_X(
            func=lambda x: tf.argmax(x, axis=-1))


def flattener(x):
    return tf.reshape(x, [tf.shape(x)[0], -1])


class Flatten(Trainable):
    def __init__(self):
        self.train_state = Trainable.trained

    def train(self, *args, train_spec=None, **kwargs):
        pass

    def eval(self, data: Dataset, *args, **kwargs):
        if not data.batched:
            data = data.batch()
        return data.tf().apply_X(
            func=flattener)
