from __future__ import annotations

from dryml.core2.object import Pickleable
from dryml.core2.tensor_spec import iter_specs
from dryml.core2.utils.general import validate_class
from dryml.data import Select, Shuffle, Take, Unbatch
from dryml.data.collate import default_collate
from dryml.models import Model as BaseModel
from dryml.models import TrainFunction


def _has_batched_spec(dataset) -> bool:
    try:
        return any(spec.batched for spec in iter_specs(dataset.spec))
    except ValueError:
        return False


def _finite_len(dataset) -> int | None:
    try:
        cardinality = dataset.__len__()
    except Exception:
        return None

    if hasattr(cardinality, "is_finite"):
        if cardinality.is_finite:
            return cardinality.require_finite()
        return None
    return int(cardinality)


class Model(BaseModel, Pickleable):
    """Wrapper around an sklearn-style estimator class."""

    def __init__(self, cls, *args, output_spec=None, **kwargs):
        self.cls = validate_class(cls)
        self.estimator_args = args
        self.estimator_kwargs = kwargs
        self.estimator = self.cls(*args, **kwargs)
        self.output_spec = output_spec

    def fit(self, x, y, *args, **kwargs):
        return self.estimator.fit(x, y, *args, **kwargs)

    def predict(self, x, *args, **kwargs):
        return self.estimator.predict(x, *args, **kwargs)

    def __call__(self, x, *args, **kwargs):
        return self.predict(x, *args, **kwargs)


class ClassifierModel(Model):
    def __call__(self, x, *args, **kwargs):
        if hasattr(self.estimator, "predict_proba"):
            return self.estimator.predict_proba(x, *args, **kwargs)
        return self.predict(x, *args, **kwargs)


class RegressionModel(Model):
    pass


class BasicTraining(TrainFunction):
    """Fit an sklearn-style estimator from an Experiment's train_data."""

    def __init__(
        self,
        *,
        x_transform=None,
        y_transform=None,
        num_examples: int | None = None,
        shuffle: bool = False,
        shuffle_seed=None,
        shuffle_buffer_size: int | None = None,
        fit_args=(),
        fit_kwargs=None,
    ):
        if num_examples is not None and num_examples < 0:
            raise ValueError("num_examples must be non-negative or None.")
        self.x_transform = Select(0) if x_transform is None else x_transform
        self.y_transform = Select(1) if y_transform is None else y_transform
        self.num_examples = num_examples
        self.shuffle = shuffle
        self.shuffle_seed = shuffle_seed
        self.shuffle_buffer_size = shuffle_buffer_size
        self.fit_args = tuple(fit_args)
        self.fit_kwargs = dict(fit_kwargs or {})

    def __call__(self, exp):
        if exp.train_data is None:
            raise ValueError("Experiment has no train_data.")

        train_data = exp.train_data
        if _has_batched_spec(train_data):
            train_data = Unbatch(train_data)

        if self.shuffle:
            buffer_size = self.shuffle_buffer_size or _finite_len(train_data)
            if buffer_size is None:
                raise ValueError("shuffle_buffer_size is required when train_data length is unknown.")
            train_data = Shuffle(train_data, buffer_size, seed=self.shuffle_seed)

        if self.num_examples is not None:
            train_data = Take(train_data, self.num_examples)

        x_values, y_values = self._collect_xy(train_data)
        if not x_values:
            raise ValueError("Cannot train on an empty dataset.")

        x = default_collate(x_values)
        y = default_collate(y_values)

        exp.model.prep_train()
        result = exp.model.fit(x, y, *self.fit_args, **self.fit_kwargs)
        exp.model.prep_eval()

        exp.state.advance_epoch()
        exp.state.advance_step(len(x_values))
        exp.state.phase = "trained"
        return result

    def _collect_xy(self, train_data):
        x_values = []
        y_values = []

        for x, y in self._iter_xy(train_data):
            x_values.append(x)
            y_values.append(y)

        return x_values, y_values

    def _iter_xy(self, train_data):
        it = iter(train_data)
        try:
            first = next(it)
        except StopIteration:
            return

        x_impl, first_x = self.x_transform.bind_first(first, input_spec=train_data.spec)
        y_impl, first_y = self.y_transform.bind_first(first, input_spec=train_data.spec)
        yield first_x, first_y

        for item in it:
            yield x_impl(item), y_impl(item)


Classifier = ClassifierModel
Regression = RegressionModel


__all__ = [
    "BasicTraining",
    "Classifier",
    "ClassifierModel",
    "Model",
    "Regression",
    "RegressionModel",
]
