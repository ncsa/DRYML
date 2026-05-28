from __future__ import annotations

from dryml.core2.object import Pickleable
from dryml.core2.tensor_spec import TensorSpec, as_tensor_spec
from dryml.core2.utils.general import validate_class
from dryml.data import collate_xy, match_input_batch, maybe_unbatch_output_spec, sample_from_spec_tree
from dryml.models import Model as BaseModel
from dryml.models import TrainFunction
from dryml.models.utils import (
    advance_train_state,
    prepare_training_data,
    validate_num_examples,
)

class Wrapper(Pickleable):
    """Pickle-backed wrapper for sklearn-style Python objects."""

    def __init__(self, cls, *args, **kwargs):
        self.cls = validate_class(cls)
        self.args = args
        self.kwargs = kwargs
        self.obj = self.cls(*args, **kwargs)


class Model(BaseModel, Pickleable):
    """Wrapper around an sklearn-style estimator class."""

    def __init__(self, cls, *args, output_spec=None, **kwargs):
        self.cls = validate_class(cls)
        self.estimator_args = args
        self.estimator_kwargs = kwargs
        self.obj = self.cls(*args, **kwargs)
        self.estimator = self.obj
        self.output_spec = output_spec

    def fit(self, x, y, *args, **kwargs):
        return self.estimator.fit(x, y, *args, **kwargs)

    def predict(self, x, *args, **kwargs):
        return self.estimator.predict(x, *args, **kwargs)

    def __call__(self, x, *args, **kwargs):
        return self.predict(x, *args, **kwargs)

    def infer_output_spec(self, input_spec):
        if self.output_spec is not None:
            return super().infer_output_spec(input_spec)

        try:
            sample = sample_from_spec_tree(input_spec)
            output = self(sample)
            return maybe_unbatch_output_spec(as_tensor_spec(output, batched=True), input_spec)
        except Exception:
            pass

        n_outputs = getattr(self.estimator, "n_outputs_", None)
        if n_outputs is not None:
            shape = () if int(n_outputs) == 1 else (int(n_outputs),)
            return match_input_batch(TensorSpec("float64", shape=shape, backend="numpy"), input_spec)

        raise NotImplementedError(
            f"Cannot infer output spec for {type(self).__name__}. "
            "Fit the estimator first, use a probe-compatible input spec, or pass output_spec explicitly."
        )


class ClassifierModel(Model):
    def __call__(self, x, *args, **kwargs):
        if hasattr(self.estimator, "predict_proba"):
            return self.estimator.predict_proba(x, *args, **kwargs)
        return self.predict(x, *args, **kwargs)

    def infer_output_spec(self, input_spec):
        if self.output_spec is not None:
            return super().infer_output_spec(input_spec)

        try:
            return super().infer_output_spec(input_spec)
        except NotImplementedError:
            classes = getattr(self.estimator, "classes_", None)
            if classes is None:
                raise
            if isinstance(classes, (tuple, list)):
                raise NotImplementedError("Multi-output classifier spec inference is not implemented.")
            return match_input_batch(
                TensorSpec("float64", shape=(len(classes),), backend="numpy"),
                input_spec,
            )


class RegressionModel(Model):
    pass


class BasicTraining(TrainFunction):
    """Fit an sklearn-style estimator from an Experiment's train_data."""

    def __init__(
        self,
        *,
        x_path=0,
        y_path=1,
        num_examples: int | None = None,
        shuffle: bool = False,
        shuffle_seed=None,
        shuffle_buffer_size: int | None = None,
        fit_args=(),
        fit_kwargs=None,
    ):
        validate_num_examples(num_examples)
        self.x_path = x_path
        self.y_path = y_path
        self.num_examples = num_examples
        self.shuffle = shuffle
        self.shuffle_seed = shuffle_seed
        self.shuffle_buffer_size = shuffle_buffer_size
        self.fit_args = tuple(fit_args)
        self.fit_kwargs = dict(fit_kwargs or {})

    def __call__(self, exp):
        train_data = prepare_training_data(
            exp.train_data,
            num_examples=self.num_examples,
            shuffle=self.shuffle,
            shuffle_seed=self.shuffle_seed,
            shuffle_buffer_size=self.shuffle_buffer_size,
        )
        x, y, n = collate_xy(
            train_data,
            x_path=self.x_path,
            y_path=self.y_path,
        )

        exp.model.prep_train()
        try:
            result = exp.model.fit(x, y, *self.fit_args, **self.fit_kwargs)
        finally:
            exp.model.prep_eval()

        advance_train_state(exp, epochs=1, steps=n)
        return result


Classifier = ClassifierModel
Regression = RegressionModel


__all__ = [
    "BasicTraining",
    "Classifier",
    "ClassifierModel",
    "Model",
    "Regression",
    "RegressionModel",
    "Wrapper",
]
