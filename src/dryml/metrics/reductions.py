"""Native confusion reductions and inert model-evaluation Fold factories."""

from __future__ import annotations

from typing import Any, Literal, TypeAlias

import numpy as np

from dryml.artifacts import Fold, mean
from dryml.core import AutoRef, ConcreteDefinition, Ref, function
from dryml.core.tensor_spec import SpecTree, TensorSpec
from dryml.data import Abs, Diff, Map, Pipe, Project, Select, Squared
from dryml.data.reduction_methods import _MAX_INT64, _backend, _cast_float64, _dtype_name, _zeros
from dryml.methods import Accumulator, Method


Label: TypeAlias = int | str
F1Average: TypeAlias = Literal["binary", "micro", "macro", "weighted", "none"]
Path: TypeAlias = str | int | tuple[str | int, ...]
ReductionMode: TypeAlias = Literal["global", "coordinate"]


def _path(value: Any, path: Path) -> Any:
    """Select one declared nested value without creating a helper Method."""

    for key in path if isinstance(path, tuple) else (path,):
        value = value[key]
    return value


def _classes(classes: tuple[Label, ...]) -> tuple[Label, ...]:
    """Validate the fixed ordered classification domain before any computation."""

    if not isinstance(classes, tuple) or not classes:
        raise ValueError("classes must be a non-empty tuple.")
    kind = type(classes[0])
    if kind not in (int, str) or any(type(label) is not kind for label in classes):
        raise TypeError("classes must contain homogeneous exact int or str labels.")
    if len(set(classes)) != len(classes):
        raise ValueError("classes must not contain duplicates.")
    return classes


def _native_bool(value: object) -> bool:
    """Evaluate one native control predicate without extracting numerical payloads."""

    return bool(value)


def _stack(values: list[object], backend: str):
    """Stack native label comparisons along their class coordinate."""

    if backend == "numpy":
        return np.stack(values, axis=-1)
    if backend == "torch":
        import torch

        return torch.stack(values, dim=-1)
    import tensorflow as tf

    return tf.stack(values, axis=-1)


def _any(value: object, backend: str):
    """Reduce one native boolean tensor to its scalar existence predicate."""

    if backend == "numpy":
        return np.any(value)
    if backend == "torch":
        import torch

        return torch.any(value)
    import tensorflow as tf

    return tf.reduce_any(value)


def _all(value: object, backend: str):
    """Reduce one native boolean tensor to its scalar universal predicate."""

    if backend == "numpy":
        return np.all(value)
    if backend == "torch":
        import torch

        return torch.all(value)
    import tensorflow as tf

    return tf.reduce_all(value)


def _argmax(value: object, backend: str):
    """Return native int64 class coordinates from validated one-hot comparisons."""

    if backend == "numpy":
        return np.argmax(value, axis=-1).astype(np.int64, copy=False)
    if backend == "torch":
        import torch

        return torch.argmax(value.to(dtype=torch.int64), dim=-1).to(dtype=torch.int64)
    import tensorflow as tf

    return tf.argmax(value, axis=-1, output_type=tf.int64)


def _sum(value: object, backend: str, axis=None):
    """Reduce native integer or float values without moving them to host storage."""

    if backend == "numpy":
        return np.sum(value, axis=axis)
    if backend == "torch":
        import torch

        return torch.sum(value, dim=axis)
    import tensorflow as tf

    return tf.reduce_sum(value, axis=axis)


def _diagonal(value: object, backend: str):
    """Select the native diagonal from one square count matrix."""

    if backend == "numpy":
        return np.diagonal(value)
    if backend == "torch":
        return value.diagonal()
    import tensorflow as tf

    return tf.linalg.diag_part(value)


def _where(condition: object, left: object, right: object, backend: str):
    """Choose native values without backend conversion."""

    if backend == "numpy":
        return np.where(condition, left, right)
    if backend == "torch":
        import torch

        return torch.where(condition, left, right)
    import tensorflow as tf

    return tf.where(condition, left, right)


def _label_indexes(value: object, classes: tuple[Label, ...], backend: str, role: str):
    """Map scalar or one-dimensional native labels to fixed class indexes.

    Raises:
        TypeError: If labels use an unsupported backend or dtype.
        ValueError: If labels are not scalar/matching one-dimensional values or
            contain a value outside the configured domain.
    """

    dtype = _dtype_name(value)
    shape = tuple(value.shape)
    if len(shape) > 1 or (len(shape) == 1 and int(shape[0]) == 0):
        raise ValueError(f"{role} labels must be a non-empty scalar or one-dimensional batch.")
    if backend == "numpy":
        if not (np.issubdtype(value.dtype, np.integer) or np.issubdtype(value.dtype, np.str_)):
            raise TypeError(f"{role} labels must have an integer or string NumPy dtype.")
        if np.issubdtype(value.dtype, np.str_) and type(classes[0]) is not str:
            raise ValueError(f"{role} labels are outside the configured class domain.")
        if np.issubdtype(value.dtype, np.integer) and type(classes[0]) is not int:
            raise ValueError(f"{role} labels are outside the configured class domain.")
    elif dtype not in {"int8", "int16", "int32", "int64"}:
        raise TypeError(f"{role} labels must have a native signed integer dtype.")
    elif type(classes[0]) is not int:
        raise TypeError(f"{role} labels support integer class domains only on {backend}.")

    comparisons = _stack([value == label for label in classes], backend)
    valid = _all(_any(comparisons, backend), backend)
    if not _native_bool(valid):
        raise ValueError(f"{role} contains a label outside the configured class domain.")
    return _argmax(comparisons, backend)


def _validate_labels(observation: Any, classes: tuple[Label, ...], prediction: Path, target: Path):
    """Validate paired single-label values and return their shared backend/indexes."""

    predicted, expected = _path(observation, prediction), _path(observation, target)
    backend = _backend(predicted)
    if _backend(expected) != backend or tuple(predicted.shape) != tuple(expected.shape):
        raise ValueError("prediction and target labels require matching backend and shape.")
    return backend, _label_indexes(predicted, classes, backend, "prediction"), _label_indexes(expected, classes, backend, "target")


def _bincount(indexes: object, size: int, backend: str):
    """Count flattened class-pair indexes in a native int64 vector."""

    if backend == "numpy":
        return np.bincount(indexes.reshape(-1), minlength=size).astype(np.int64, copy=False)
    if backend == "torch":
        import torch

        return torch.bincount(indexes.reshape(-1), minlength=size).to(dtype=torch.int64)
    import tensorflow as tf

    return tf.math.bincount(tf.reshape(indexes, (-1,)), minlength=size, maxlength=size, dtype=tf.int64)


def _reshape(value: object, shape: tuple[int, ...], backend: str):
    """Reshape a native count vector without using host array conversion."""

    if backend == "numpy":
        return value.reshape(shape)
    if backend == "torch":
        return value.reshape(shape)
    import tensorflow as tf

    return tf.reshape(value, shape)


def _validate_count_matrix(matrix: object, *, require_int64: bool = False) -> str:
    """Validate a non-empty native square nonnegative integer count matrix."""

    backend = _backend(matrix)
    if len(matrix.shape) != 2 or int(matrix.shape[0]) != int(matrix.shape[1]) or int(matrix.shape[0]) == 0:
        raise ValueError("confusion matrix must be a non-empty square matrix.")
    dtype = _dtype_name(matrix)
    if dtype not in {"int8", "int16", "int32", "int64"} or (require_int64 and dtype != "int64"):
        raise TypeError("confusion matrix must use a signed integer dtype.")
    if _native_bool(_any(matrix < 0, backend)):
        raise ValueError("confusion matrix must not contain negative counts.")
    return backend


class ConfusionInitial(Method):
    """Allocate the fixed native int64 carry for one confusion reduction.

    Args:
        classes: Ordered, unique, homogeneous exact ``int`` or ``str`` labels.
        prediction: Path selecting decoded predictions from an observation.
        target: Path selecting decoded targets from an observation.

    Returns:
        A zero ``(len(classes), len(classes))`` native int64 count matrix.

    Raises:
        TypeError: If the domain or observed label backend/dtype is unsupported.
        ValueError: If labels have unsupported shape or backend alignment.

    Side Effects:
        Allocates invocation-owned carry only when Fold invokes this Method.
    """

    def __init__(self, classes: tuple[Label, ...], *, prediction: Path = "prediction", target: Path = "target") -> None:
        self.classes = _classes(classes)
        self.prediction, self.target = prediction, target

    def _runtime_selection_facts(self, args, kwargs):
        """Keep direct NumPy string-label calls outside TensorSpec's numeric adapter."""

        return None, None

    def __call__(self, observation: Any):
        """Return a fresh zero matrix using the observation's native backend."""

        backend, _, _ = _validate_labels(observation, self.classes, self.prediction, self.target)
        size = len(self.classes)
        return _zeros((size, size), backend, dtype="int64")

    def infer_output_spec(self, observation_spec: SpecTree) -> SpecTree:
        """Infer a fixed native int64 matrix without invoking label conversion."""

        prediction, target = _path(observation_spec, self.prediction), _path(observation_spec, self.target)
        if not isinstance(prediction, TensorSpec) or not isinstance(target, TensorSpec):
            raise TypeError("ConfusionInitial requires TensorSpec prediction and target labels.")
        if prediction.backend != target.backend or prediction.shape != target.shape:
            raise ValueError("prediction and target label specs must match.")
        return TensorSpec("int64", shape=(len(self.classes), len(self.classes)), backend=prediction.backend)


class ConfusionCounts(Accumulator):
    """Advance fixed-domain truth-row, prediction-column confusion counts.

    Args:
        classes: Ordered, unique, homogeneous exact ``int`` or ``str`` labels.
        prediction: Path selecting already-decoded predictions from an observation.
        target: Path selecting already-decoded targets from an observation.

    Returns:
        A successor native int64 count matrix with rows for truth and columns for
        prediction.

    Raises:
        TypeError: If labels or carry use unsupported types.
        ValueError: If labels are logits, mismatched, or outside ``classes``.
        OverflowError: If an update would exceed signed int64 count capacity.

    Side Effects:
        Does not mutate the supplied state; it returns a new native successor.
    """

    def __init__(self, classes: tuple[Label, ...], *, prediction: Path = "prediction", target: Path = "target") -> None:
        self.classes = _classes(classes)
        self.prediction, self.target = prediction, target

    def _runtime_selection_facts(self, args, kwargs):
        """Keep direct NumPy string-label calls outside TensorSpec's numeric adapter."""

        return None, None

    def __call__(self, observation: Any, state: object):
        """Validate labels before calculating one native, overflow-safe successor."""

        backend, predicted, expected = _validate_labels(observation, self.classes, self.prediction, self.target)
        if _validate_count_matrix(state, require_int64=True) != backend or int(state.shape[0]) != len(self.classes):
            raise ValueError("confusion carry must be an int64 matrix for the configured class domain.")
        increments = _bincount(expected * len(self.classes) + predicted, len(self.classes) ** 2, backend)
        increments = _reshape(increments, tuple(state.shape), backend)
        if _native_bool(_any(state > _MAX_INT64 - increments, backend)):
            raise OverflowError("confusion count update would overflow int64.")
        return state + increments

    def infer_output_spec(self, observation_spec: SpecTree, state_spec: SpecTree) -> SpecTree:
        """Require the declared fixed carry shape and return it unchanged."""

        expected = ConfusionInitial(self.classes, prediction=self.prediction, target=self.target).infer_output_spec(observation_spec)
        if state_spec != expected:
            raise ValueError("ConfusionCounts state spec must match its fixed class-domain matrix.")
        return state_spec


class AccuracyFromConfusion(Method):
    """Compute zero-safe scalar accuracy from one validated count matrix.

    Args:
        matrix: A non-empty square native signed-integer count matrix.

    Returns:
        A native float64 scalar; an all-zero matrix returns zero.

    Raises:
        TypeError: If the matrix is not a supported native signed-integer tensor.
        ValueError: If the matrix is empty, non-square, or contains negative counts.

    Side Effects:
        None. The matrix remains native and is not mutated.
    """

    def __call__(self, matrix: object):
        """Return native diagonal-over-total accuracy without changing the matrix."""

        backend = _validate_count_matrix(matrix)
        total = _sum(matrix, backend)
        if _native_bool(total == 0):
            return _zeros((), backend, dtype="float64")
        return _cast_float64(_sum(_diagonal(matrix, backend), backend), backend) / _cast_float64(total, backend)

    def infer_output_spec(self, input_spec: SpecTree) -> SpecTree:
        """Infer one native float64 scalar from a square integer matrix spec."""

        if not isinstance(input_spec, TensorSpec) or input_spec.shape is None or len(input_spec.shape) != 2:
            raise TypeError("AccuracyFromConfusion requires a square TensorSpec matrix.")
        if input_spec.shape[0] != input_spec.shape[1] or input_spec.shape[0] == 0:
            raise ValueError("AccuracyFromConfusion requires a square TensorSpec matrix.")
        if str(input_spec.dtype) not in {"int8", "int16", "int32", "int64"}:
            raise TypeError("AccuracyFromConfusion requires a signed integer TensorSpec matrix.")
        return TensorSpec("float64", shape=(), backend=input_spec.backend)


class F1FromConfusion(Method):
    """Compute binary, micro, macro, weighted, or per-class F1 from counts.

    Args:
        average: One of ``"binary"``, ``"micro"``, ``"macro"``, ``"weighted"``,
            or ``"none"``.
        positive_index: Required exact class coordinate for ``"binary"`` and
            forbidden for the other averaging modes.

    Returns:
        A native float64 scalar for aggregate modes or a class-ordered float64
        vector for ``average="none"``. Undefined per-class terms are zero.

    Raises:
        ValueError: If averaging controls, matrix counts, or binary class
            selection are invalid.
        TypeError: If the matrix is not a supported native signed-integer tensor.

    Side Effects:
        None. The supplied count matrix remains native and is not mutated.
    """

    def __init__(self, *, average: F1Average, positive_index: int | None = None) -> None:
        if average not in ("binary", "micro", "macro", "weighted", "none"):
            raise ValueError("average must be binary, micro, macro, weighted, or none.")
        if average == "binary":
            if type(positive_index) is not int or positive_index < 0:
                raise ValueError("binary F1 requires a nonnegative exact int positive_index.")
        elif positive_index is not None:
            raise ValueError("positive_index is valid only for binary F1.")
        self.average, self.positive_index = average, positive_index

    def __call__(self, matrix: object):
        """Return native F1 values using zero for unsupported per-class terms."""

        backend = _validate_count_matrix(matrix)
        classes = int(matrix.shape[0])
        if self.average == "binary" and (classes != 2 or self.positive_index >= classes):
            raise ValueError("binary F1 requires exactly two classes and a valid positive_index.")
        total = _sum(matrix, backend)
        if _native_bool(total == 0):
            return _zeros((classes,), backend, dtype="float64") if self.average == "none" else _zeros((), backend, dtype="float64")
        matrix_float = _cast_float64(matrix, backend)
        true_positive = _diagonal(matrix_float, backend)
        support = _sum(matrix_float, backend, axis=1)
        predicted = _sum(matrix_float, backend, axis=0)
        denominator = support + predicted
        safe_denominator = _where(denominator == 0, _zeros(tuple(denominator.shape), backend, dtype="float64") + 1.0, denominator, backend)
        per_class = _where(denominator == 0, _zeros(tuple(denominator.shape), backend, dtype="float64"), 2.0 * true_positive / safe_denominator, backend)
        if self.average == "none":
            return per_class
        if self.average == "binary":
            return per_class[self.positive_index]
        if self.average == "micro":
            return 2.0 * _sum(true_positive, backend) / (2.0 * _cast_float64(total, backend))
        if self.average == "macro":
            return _sum(per_class, backend) / float(classes)
        return _sum(per_class * support, backend) / _cast_float64(total, backend)

    def infer_output_spec(self, input_spec: SpecTree) -> SpecTree:
        """Infer scalar or class-vector native float64 output without counting."""

        if not isinstance(input_spec, TensorSpec) or input_spec.shape is None or len(input_spec.shape) != 2:
            raise TypeError("F1FromConfusion requires a square TensorSpec matrix.")
        if input_spec.shape[0] != input_spec.shape[1] or input_spec.shape[0] == 0:
            raise ValueError("F1FromConfusion requires a square TensorSpec matrix.")
        if str(input_spec.dtype) not in {"int8", "int16", "int32", "int64"}:
            raise TypeError("F1FromConfusion requires a signed integer TensorSpec matrix.")
        if self.average == "binary" and (
                input_spec.shape[0] != 2 or self.positive_index >= 2):
            raise ValueError("binary F1 requires exactly two classes.")
        shape = input_spec.shape[:1] if self.average == "none" else ()
        return TensorSpec("float64", shape=shape, backend=input_spec.backend)


def _concrete_evaluation_source(test_ds: object, model: object, *, x: Path, y: Path, prediction_labels: object | None = None, target_labels: object | None = None) -> ConcreteDefinition:
    """Declare and inertly concretize one complete model-evaluation Map graph."""

    prediction = Pipe.defn(Select.defn(x), model)
    if prediction_labels is not None:
        prediction = Pipe.defn(prediction, prediction_labels)
    target = Select.defn(y) if target_labels is None else Pipe.defn(Select.defn(y), target_labels)
    return Map.defn(test_ds, Project.defn(prediction=prediction, target=target)).concretize()


def _error_source(test_ds: object, model: object, *, x: Path, y: Path, error: type[Method]) -> ConcreteDefinition:
    """Declare one projected error stream and concretize it without materializing inputs."""

    evaluation = _concrete_evaluation_source(test_ds, model, x=x, y=y)
    return Map.defn(evaluation, Pipe.defn(Diff.defn(x="prediction", y="target"), error.defn())).concretize()


@function
def regressor_mae(test_ds: Ref[AutoRef], model: Ref[AutoRef], *, x: Path = "x", y: Path = "y", mode: ReductionMode) -> Fold:
    """Declare an inert streaming MAE Fold over one model evaluation graph.

    Args:
        test_ds: Non-materializing evaluation Dataset reference.
        model: Non-materializing model Method reference.
        x: Source path selecting model input.
        y: Source path selecting regression target.
        mode: Global or coordinate-wise mean population definition.

    Returns:
        An uncomputed Fold using U6's native mean carry.

    Raises:
        TypeError: If references or paths cannot form the declared Method graph.
        ValueError: If ``mode`` is not a supported mean population definition.

    Side Effects:
        None. It neither saves nor materializes either input, evaluates the
        model, or begins Dataset iteration.
    """

    return mean(_error_source(test_ds, model, x=x, y=y, error=Abs), mode=mode)


@function
def regressor_mse(test_ds: Ref[AutoRef], model: Ref[AutoRef], *, x: Path = "x", y: Path = "y", mode: ReductionMode) -> Fold:
    """Declare an inert streaming MSE Fold over one model evaluation graph.

    Args:
        test_ds: Non-materializing evaluation Dataset reference.
        model: Non-materializing model Method reference.
        x: Source path selecting model input.
        y: Source path selecting regression target.
        mode: Global or coordinate-wise mean population definition.

    Returns:
        An uncomputed Fold using U6's native mean carry.

    Raises:
        TypeError: If references or paths cannot form the declared Method graph.
        ValueError: If ``mode`` is not a supported mean population definition.

    Side Effects:
        None. It neither saves nor materializes either input, evaluates the
        model, or begins Dataset iteration.
    """

    return mean(_error_source(test_ds, model, x=x, y=y, error=Squared), mode=mode)


def _classifier_fold(test_ds: object, model: object, *, classes: tuple[Label, ...], prediction_labels: Method, target_labels: Method, x: Path, y: Path, finalize: Method | None) -> Fold:
    """Build one complete inert classified-evaluation Fold with caller label Methods."""

    domain = _classes(classes)
    source = _concrete_evaluation_source(
        test_ds, model, x=x, y=y,
        prediction_labels=prediction_labels, target_labels=target_labels,
    )
    return Fold(
        source,
        initial_state=ConfusionInitial(domain),
        accumulator=ConfusionCounts(domain),
        finalize=finalize,
    )


@function
def classifier_confusion_matrix(test_ds: Ref[AutoRef], model: Ref[AutoRef], *, classes: tuple[Label, ...], prediction_labels: Method, target_labels: Method, x: Path = "x", y: Path = "y") -> Fold:
    """Declare an inert fixed-domain confusion-matrix Fold without label guessing.

    Args:
        test_ds: Non-materializing evaluation Dataset reference.
        model: Non-materializing model Method reference.
        classes: Ordered finite domain of decoded labels.
        prediction_labels: Declared Method converting model output to labels.
        target_labels: Declared Method converting targets to labels.
        x: Source path selecting model input.
        y: Source path selecting target values.

    Returns:
        An uncomputed Fold whose result is a truth-row, prediction-column matrix.

    Raises:
        TypeError: If references, labels, or Methods are unsupported.
        ValueError: If ``classes`` is empty, mixed, or duplicated.

    Side Effects:
        None. Construction does not save or materialize inputs or decode labels.
    """

    return _classifier_fold(test_ds, model, classes=classes, prediction_labels=prediction_labels, target_labels=target_labels, x=x, y=y, finalize=None)


@function
def classifier_accuracy(test_ds: Ref[AutoRef], model: Ref[AutoRef], *, classes: tuple[Label, ...], prediction_labels: Method, target_labels: Method, x: Path = "x", y: Path = "y") -> Fold:
    """Declare an inert confusion-based accuracy Fold without a second evaluation loop.

    Args:
        test_ds: Non-materializing evaluation Dataset reference.
        model: Non-materializing model Method reference.
        classes: Ordered finite domain of decoded labels.
        prediction_labels: Declared Method converting model output to labels.
        target_labels: Declared Method converting targets to labels.
        x: Source path selecting model input.
        y: Source path selecting target values.

    Returns:
        An uncomputed Fold whose result is native scalar accuracy.

    Raises:
        TypeError: If references, labels, or Methods are unsupported.
        ValueError: If ``classes`` is empty, mixed, or duplicated.

    Side Effects:
        None. Construction does not save or materialize inputs or decode labels.
    """

    return _classifier_fold(test_ds, model, classes=classes, prediction_labels=prediction_labels, target_labels=target_labels, x=x, y=y, finalize=AccuracyFromConfusion())


@function
def classifier_f1(test_ds: Ref[AutoRef], model: Ref[AutoRef], *, classes: tuple[Label, ...], prediction_labels: Method, target_labels: Method, average: F1Average, positive_index: int | None = None, x: Path = "x", y: Path = "y") -> Fold:
    """Declare an inert confusion-based F1 Fold using explicit label conversion.

    Args:
        test_ds: Non-materializing evaluation Dataset reference.
        model: Non-materializing model Method reference.
        classes: Ordered finite domain of decoded labels.
        prediction_labels: Declared Method converting model output to labels.
        target_labels: Declared Method converting targets to labels.
        average: Binary, micro, macro, weighted, or per-class F1 mode.
        positive_index: Required positive class coordinate for binary F1 only.
        x: Source path selecting model input.
        y: Source path selecting target values.

    Returns:
        An uncomputed Fold with scalar or per-class native F1 result semantics.

    Raises:
        TypeError: If references, labels, or Methods are unsupported.
        ValueError: If the class domain or F1 averaging controls are invalid.

    Side Effects:
        None. Construction does not save or materialize inputs or decode labels.
    """

    return _classifier_fold(test_ds, model, classes=classes, prediction_labels=prediction_labels, target_labels=target_labels, x=x, y=y, finalize=F1FromConfusion(average=average, positive_index=positive_index))


__all__ = [
    "AccuracyFromConfusion", "ConfusionCounts", "ConfusionInitial", "F1FromConfusion",
    "F1Average", "Label", "classifier_accuracy", "classifier_confusion_matrix",
    "classifier_f1", "regressor_mae", "regressor_mse",
]
