from dryml.models import DryTrainable
from dryml.models import DryComponent
import tensorflow as tf
import inspect


class MapBasicSupervised(DryComponent):
    def __call__(
            self, trainable, in_ds, *args, target=True,
            index=False, **kwargs):
        # Assume a two element tuple, the first is X second is Y
        if target:
            # Do nothing
            return in_ds
        else:
            return in_ds.map(
                lambda t: t[0],
                num_parallel_calls=tf.data.AUTOTUNE)


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

    def eval(self, X, *args, **kwargs):
        return X.map(self.func, num_parallel_calls=tf.data.AUTOTUNE)
