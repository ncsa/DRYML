from __future__ import annotations

import numpy as np

from dryml.core.object import Pickleable
from dryml.core.tensor_spec import (
    TensorSpec,
    iter_specs,
    match_input_batch,
)
from dryml.core.utils.general import validate_class
from dryml.data import materialize_supervised
from dryml.models import Model as BaseModel
from dryml.models import TrainFunction
from dryml.models.utils import (
    require_supervised_dataset,
)

class Wrapper(Pickleable):
    """Pickle-backed wrapper for sklearn-style Python objects."""

    def __init__(self, cls, *args, **kwargs):
        self.cls = validate_class(cls)
        self.args = args
        self.kwargs = kwargs
        self.obj = self.cls(*args, **kwargs)


class Model(BaseModel, Pickleable):
    """Wrapper around an sklearn-style estimator class.

    Output inference reads fitted estimator metadata only. An estimator without
    supported metadata requires an explicit ``output_spec`` and is never probed
    by calling ``predict`` with fabricated values.
    """

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

        n_outputs = getattr(self.estimator, "n_outputs_", None)
        if n_outputs is None:
            coefficients = getattr(self.estimator, "coef_", None)
            if coefficients is not None:
                n_outputs = 1 if len(getattr(coefficients, "shape", ())) == 1 else coefficients.shape[0]
        if n_outputs is not None:
            shape = () if int(n_outputs) == 1 else (int(n_outputs),)
            return match_input_batch(
                TensorSpec(next(iter_specs(input_spec)).dtype, shape=shape, backend="numpy"),
                input_spec,
            )

        raise NotImplementedError(
            f"Cannot infer output spec for {type(self).__name__}. "
            "Fit the estimator first when it exposes pure metadata, or pass output_spec explicitly."
        )


class ClassifierModel(Model):
    """Sklearn classifier wrapper with metadata-only output inference."""

    def __call__(self, x, *args, **kwargs):
        """Return class probabilities when the estimator supports them."""

        if hasattr(self.estimator, "predict_proba"):
            return self.estimator.predict_proba(x, *args, **kwargs)
        return self.predict(x, *args, **kwargs)

    def infer_output_spec(self, input_spec):
        """Infer class output shape and dtype from fitted estimator metadata."""

        if self.output_spec is not None:
            return super().infer_output_spec(input_spec)

        classes = getattr(self.estimator, "classes_", None)
        if classes is not None:
            if isinstance(classes, (tuple, list)):
                raise NotImplementedError("Multi-output classifier spec inference is not implemented.")
            predicts_probabilities = hasattr(self.estimator, "predict_proba")
            dtype = np.asarray(classes).dtype
            if predicts_probabilities:
                dtype = np.dtype("float64")
                tags_method = getattr(self.estimator, "__sklearn_tags__", None)
                if tags_method is not None:
                    try:
                        if tags_method().array_api_support:
                            dtype = next(iter_specs(input_spec)).dtype
                    except Exception:
                        pass
            return match_input_batch(
                TensorSpec(
                    dtype,
                    shape=(len(classes),) if predicts_probabilities else (),
                    backend="numpy",
                ),
                input_spec,
            )
        return super().infer_output_spec(input_spec)


class RegressionModel(Model):
    pass


class BasicTraining(TrainFunction):
    """Fit an sklearn-style estimator from a canonical Experiment Dataset.

    Dataset selection, ordering, and batching are authored before this trainer.
    Unbatched and authored-batch pairs are both materialized as one estimator
    input; a successful fit records one coarse transition rather than optimizer
    safe points.
    """

    supports_safe_points = False

    def __init__(
        self,
        *,
        fit_args=(),
        fit_kwargs=None,
    ):
        self.fit_args = tuple(fit_args)
        self.fit_kwargs = dict(fit_kwargs or {})

    __dryml_retired_constructor_parameters__ = (
        "batch_size", "num_examples", "shuffle", "shuffle_seed",
        "shuffle_buffer_size", "x_path", "y_path",
    )

    def __call__(self, exp, *, callbacks=()):
        """Fit the estimator once, rejecting unsupported intermediate safe points.

        Args:
            exp: Experiment supplying the estimator and training data.
            callbacks: Requested optimizer-update safe points.

        Returns:
            The wrapped estimator's ``fit`` result.

        Raises:
            NotImplementedError: If intermediate callbacks are requested because
                sklearn's one-shot fit has no truthful per-update boundary.

        Side Effects:
            Fits the estimator and advances its coarse Experiment TrainState on
            success. It does not retain per-update exposure accounting.
        """

        from dryml.models.utils import validate_training_callbacks

        callbacks = validate_training_callbacks(callbacks)
        if callbacks:
            raise NotImplementedError("sklearn training does not support intermediate safe points.")
        require_supervised_dataset(exp.train_data, batched=False)
        x, y, n = materialize_supervised(exp.train_data)

        exp.model.prep_train()
        try:
            result = exp.model.fit(x, y, *self.fit_args, **self.fit_kwargs)
        finally:
            exp.model.prep_eval()

        exp.state.record_fit(examples=n)
        exp.state.finish_epoch()
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
