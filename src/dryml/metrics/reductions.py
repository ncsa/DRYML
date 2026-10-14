"""Native confusion reductions and inert model-evaluation Fold factories."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from functools import partial
from typing import Any, Literal, TypeAlias

import numpy as np

from dryml.artifacts import Fold, mean
from dryml.core import AutoRef, ConcreteDefinition, Definition, Ref, authoring_helper
from dryml.core.backend import Backend
from dryml.core.cdef_graph import EdgeKind
from dryml.core.links import DefLink
from dryml.core.tensor_spec import BatchMode, SpecTree, TensorSpec
from dryml.core.template import Expr, _contains_template_value
from dryml.data import Abs, Diff, Map, Pipe, Project, Select, Squared
from dryml.data.reduction_methods import (
    MeanFinalize, MeanInitial, MeanUpdate, Path, ReductionMode, _MAX_INT64,
    _backend, _cast_float64, _dtype_name,
    _path, _reshape, _scalar, _where, _zeros,
)
from dryml.methods import Accumulator, ImplementationSelectionError, Method
from dryml.methods.conversion import backend_spec
from dryml.methods.signature import spec_node


Label: TypeAlias = int | str
F1Average: TypeAlias = Literal["binary", "micro", "macro", "weighted", "none"]


def _metric_factory(target):
    """Lift only supplied metric helpers into class-rooted Fold Definitions.

    Concrete calls retain the existing signature-normalized Fold construction.
    Symbolic calls author their complete Fold graph directly and defer
    value-dependent validation until Definition binding completes; this is
    deliberately not a general function-lifting facility.
    """

    return authoring_helper(
        author_definition=partial(_symbolic_metric_definition, target.__name__),
        validate_known_arguments=partial(_validate_known_metric_arguments, target.__name__),
        normalize_concrete=True,
    )(target)


def _validate_known_metric_arguments(name: str, arguments: Mapping[str, object]) -> None:
    """Validate literal metric controls while retaining symbolic dependencies."""

    if "mode" in arguments and not isinstance(arguments["mode"], Expr):
        mode = arguments["mode"]
        if mode not in ("global", "coordinate"):
            raise ValueError("mode must be 'global' or 'coordinate'.")
    if "classes" in arguments:
        _validate_known_classes(arguments["classes"])
    if name == "classifier_f1":
        _validate_known_f1_controls(
            arguments["average"], arguments["positive_index"],
        )


def _validate_known_classes(classes: object) -> None:
    """Reject fixed class-domain errors while allowing symbolic tuple members."""

    if isinstance(classes, Expr):
        return
    if not isinstance(classes, tuple) or not classes:
        _classes(classes)
    known = tuple(label for label in classes if not _contains_template_value(label))
    if known:
        kind = type(known[0])
        if kind not in (int, str) or any(type(label) is not kind for label in known):
            raise TypeError("classes must contain homogeneous exact int or str labels.")
        if len(set(known)) != len(known):
            raise ValueError("classes must not contain duplicates.")
    if len(known) == len(classes):
        _classes(classes)


def _validate_known_f1_controls(average: object, positive_index: object) -> None:
    """Validate F1 facts that do not depend on unresolved direct controls."""

    average_symbolic = isinstance(average, Expr)
    positive_symbolic = isinstance(positive_index, Expr)
    if not average_symbolic and average not in ("binary", "micro", "macro", "weighted", "none"):
        raise ValueError("average must be binary, micro, macro, weighted, or none.")
    if not positive_symbolic and positive_index is not None and (
            type(positive_index) is not int or positive_index < 0):
        raise ValueError("binary F1 requires a nonnegative exact int positive_index.")
    if average_symbolic or positive_symbolic:
        return
    if average == "binary" and positive_index is None:
        raise ValueError("binary F1 requires a nonnegative exact int positive_index.")
    if average != "binary" and positive_index is not None:
        raise ValueError("positive_index is valid only for binary F1.")


def _evaluation_pair_specs(input_spec: SpecTree) -> tuple[TensorSpec, TensorSpec]:
    """Return the prediction and target leaves from one metric evaluation spec."""

    if not isinstance(input_spec, Mapping) or tuple(input_spec) != (
            "prediction", "target"):
        raise TypeError("Metric evaluation requires prediction and target fields.")
    prediction, target = input_spec["prediction"], input_spec["target"]
    if not isinstance(prediction, TensorSpec) or not isinstance(target, TensorSpec):
        raise TypeError("Metric evaluation prediction and target must be TensorSpecs.")
    if prediction.backend is None or target.backend is None:
        raise ValueError("Metric evaluation requires declared prediction and target backends.")
    if prediction.batched != target.batched:
        raise ValueError("Metric evaluation prediction and target batch modes differ.")
    return prediction, target


def _evaluation_host_array(value: object, backend: Backend) -> np.ndarray:
    """Copy one dense metric value to independent C-contiguous NumPy storage.

    Accelerator transfer is intentional at this terminal evaluation boundary:
    training remains native while Fold reduction carry stays host-portable.
    """

    if backend is Backend.numpy:
        if not isinstance(value, (np.ndarray, np.generic)):
            raise TypeError("Metric evaluation expected a NumPy value.")
        array = np.asarray(value)
    elif backend is Backend.torch:
        import torch

        if not isinstance(value, torch.Tensor) or value.layout is not torch.strided:
            raise TypeError("Metric evaluation requires a dense strided Torch tensor.")
        array = value.detach().cpu().numpy()
    elif backend is Backend.tf:
        import tensorflow as tf

        if (
                not tf.is_tensor(value)
                or isinstance(value, (tf.RaggedTensor, tf.SparseTensor))
                or not callable(getattr(value, "numpy", None))):
            raise TypeError("Metric evaluation requires a concrete dense TensorFlow tensor.")
        array = value.numpy()
    elif backend is Backend.jax:
        import jax

        if not isinstance(value, jax.Array) or not value.is_fully_addressable:
            raise TypeError("Metric evaluation requires a concrete addressable JAX array.")
        if len(value.devices()) != 1:
            raise TypeError("Metric evaluation requires one JAX prediction device.")
        array = np.asarray(jax.device_get(value))
    else:
        raise TypeError("Metric evaluation selected an unsupported backend.")
    if array.dtype == np.dtype(object) or array.dtype.kind not in "?biufcSU":
        raise TypeError("Metric evaluation produced an unsupported host dtype.")
    return np.array(array, copy=True, order="C")


def _evaluation_host_spec(spec: TensorSpec) -> TensorSpec:
    """Return the host contract, retaining already-host semantic string dtype."""

    if spec.backend is Backend.numpy:
        return spec
    return backend_spec(spec, Backend.numpy)


class _AlignEvaluationTarget(Method):
    """Normalize one prediction/target pair at the host evaluation boundary.

    The original private name remains stable because authored Artifact graphs
    persist its import reference.
    """

    def __call__(self, item):
        """Reject direct use that bypasses the spec-planned metric handoff."""

        raise RuntimeError("Metric target alignment requires selected specification evidence.")

    def infer_output_spec(self, input_spec: SpecTree) -> SpecTree:
        """Declare both evaluation values on the checkpoint-safe NumPy backend."""

        prediction, target = _evaluation_pair_specs(input_spec)
        return {
            "prediction": _evaluation_host_spec(prediction),
            "target": _evaluation_host_spec(target),
        }

    def find_implementation(
            self, input_spec=None, *additional_input_specs, backend=None,
            batch_mode=None, output_spec=None):
        """Plan one explicit host boundary despite the mixed input pair."""

        return self._select_alignment(
            input_spec, additional_input_specs, backend=backend,
            batch_mode=batch_mode, output_spec=output_spec, prepare=False,
        )

    def _prepare_implementation(self, input_spec, *, backend, batch_mode):
        """Plan the same explicit host boundary during graph preparation."""

        return self._select_alignment(
            input_spec, (), backend=backend, batch_mode=batch_mode,
            output_spec=None, prepare=True,
        )

    def _select_alignment(
            self, input_spec, additional_input_specs, *, backend, batch_mode,
            output_spec, prepare):
        """Select a NumPy carrier while retaining exact pair specifications."""

        if additional_input_specs or input_spec is None:
            raise ImplementationSelectionError("conflict")
        try:
            prediction, target = _evaluation_pair_specs(input_spec)
            prediction_batch = (
                BatchMode.batched if prediction.batched else BatchMode.element
            )
            required_backend = None if backend is None else Backend(backend)
            required_batch = None if batch_mode is None else BatchMode(batch_mode)
            if (
                    required_backend not in (None, Backend.numpy)
                    or required_batch not in (None, prediction_batch)):
                raise ValueError("Metric evaluation selection facts conflict.")
            aligned_spec = self.infer_output_spec(input_spec)
            selected_output = aligned_spec if output_spec is None else output_spec
            input_node = spec_node(input_spec)
            output_node = spec_node(selected_output)
        except (TypeError, ValueError) as error:
            raise ImplementationSelectionError("conflict") from error

        implementation = (
            super()._prepare_implementation(
                None, backend=Backend.numpy, batch_mode=prediction_batch,
            )
            if prepare
            else super().find_implementation(
                None, backend=Backend.numpy, batch_mode=prediction_batch,
            )
        )

        def invoke(item):
            return {
                "prediction": _evaluation_host_array(
                    item["prediction"], prediction.backend,
                ),
                "target": _evaluation_host_array(item["target"], target.backend),
            }

        return replace(
            implementation,
            _input_specs=(input_node,),
            _output_spec=output_node,
            _invoker=invoke,
        )


def _symbolic_evaluation_source(
        test_ds: object,
        model: object,
        *,
        x: Path,
        y: Path,
        prediction_labels: object | None = None,
        target_labels: object | None = None) -> Definition:
    """Author an evaluation Map Definition without constructing graph Objects."""

    prediction = Pipe.defn(Select.defn(x), model)
    if prediction_labels is not None:
        prediction = Pipe.defn(prediction, prediction_labels)
    target = Select.defn(y) if target_labels is None else Pipe.defn(
        Select.defn(y), target_labels,
    )
    evaluation = Map.defn(
        test_ds, Project.defn(prediction=prediction, target=target),
    )
    return Map.defn(evaluation, _AlignEvaluationTarget.defn())


def _symbolic_metric_definition(name: str, arguments: Mapping[str, object]) -> Definition:
    """Author one complete class-rooted Fold Definition for a supplied helper."""

    test_ds, model = arguments["test_ds"], arguments["model"]
    x, y = arguments["x"], arguments["y"]
    if name in {"regressor_mae", "regressor_mse"}:
        error = Abs if name == "regressor_mae" else Squared
        evaluation = _symbolic_evaluation_source(test_ds, model, x=x, y=y)
        source = Map.defn(
            evaluation,
            Pipe.defn(Diff.defn(x="prediction", y="target"), error.defn()),
        )
        mode = arguments["mode"]
        return Fold.defn(
            DefLink.finalized(EdgeKind.REF, source),
            initial_state=MeanInitial.defn(mode=mode),
            accumulator=MeanUpdate.defn(mode=mode),
            finalize=MeanFinalize.defn(),
        )
    source = _symbolic_evaluation_source(
        test_ds,
        model,
        x=x,
        y=y,
        prediction_labels=arguments["prediction_labels"],
        target_labels=arguments["target_labels"],
    )
    domain = arguments["classes"]
    if name == "classifier_confusion_matrix":
        finalize = None
    elif name == "classifier_accuracy":
        finalize = AccuracyFromConfusion.defn()
    else:
        finalize = F1FromConfusion.defn(
            average=arguments["average"],
            positive_index=arguments["positive_index"],
        )
    return Fold.defn(
        DefLink.finalized(EdgeKind.REF, source),
        initial_state=ConfusionInitial.defn(domain),
        accumulator=ConfusionCounts.defn(domain),
        finalize=finalize,
    )


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


def _consume_count_capacity(matrix: object, remaining: object, backend: str):
    """Return native int64 capacity after exactly subtracting nonnegative count cells.

    Raises:
        OverflowError: If the matrix population exceeds the supplied capacity.

    This walks the fixed matrix coordinates instead of reducing them, because an
    int64 aggregate can wrap before a native reduction can be compared safely.
    """

    for row in range(int(matrix.shape[0])):
        for column in range(int(matrix.shape[1])):
            count = matrix[row, column]
            if _native_bool(count > remaining):
                raise OverflowError("confusion count update would overflow int64.")
            remaining = remaining - count
    return remaining


def _confusion_state_spec(observation_spec, classes, prediction, target):
    """Infer fixed count storage without constructing or invoking a Method."""
    prediction, target = _path(observation_spec, prediction), _path(observation_spec, target)
    if not isinstance(prediction, TensorSpec) or not isinstance(target, TensorSpec):
        raise TypeError("ConfusionInitial requires TensorSpec prediction and target labels.")
    if prediction.backend != target.backend or prediction.shape != target.shape:
        raise ValueError("prediction and target label specs must match.")
    return TensorSpec("int64", shape=(len(classes), len(classes)), backend=prediction.backend)


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

    @staticmethod
    def __dryml_normalize_definition_arguments__(args, kwargs):
        """Validate a fully bound class domain during Definition concretization."""

        classes = args[0] if args else kwargs.get("classes")
        _validate_known_classes(classes)
        return args, kwargs

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

        return _confusion_state_spec(observation_spec, self.classes, self.prediction, self.target)


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

    @staticmethod
    def __dryml_normalize_definition_arguments__(args, kwargs):
        """Validate a fully bound class domain during Definition concretization."""

        classes = args[0] if args else kwargs.get("classes")
        _validate_known_classes(classes)
        return args, kwargs

    def _runtime_selection_facts(self, args, kwargs):
        """Keep direct NumPy string-label calls outside TensorSpec's numeric adapter."""

        return None, None

    def __call__(self, observation: Any, state: object):
        """Validate labels before calculating one native, overflow-safe successor."""

        backend, predicted, expected = _validate_labels(observation, self.classes, self.prediction, self.target)
        if _validate_count_matrix(state, require_int64=True) != backend or int(state.shape[0]) != len(self.classes):
            raise ValueError("confusion carry must be an int64 matrix for the configured class domain.")
        remaining = _consume_count_capacity(state, _scalar(_MAX_INT64, backend), backend)
        increments = _bincount(expected * len(self.classes) + predicted, len(self.classes) ** 2, backend)
        increments = _reshape(increments, tuple(state.shape), backend)
        if _native_bool(_any(state > _MAX_INT64 - increments, backend)):
            raise OverflowError("confusion count update would overflow int64.")
        _consume_count_capacity(increments, remaining, backend)
        return state + increments

    def infer_output_spec(self, observation_spec: SpecTree, state_spec: SpecTree) -> SpecTree:
        """Require the declared fixed carry shape and return it unchanged."""

        expected = _confusion_state_spec(observation_spec, self.classes, self.prediction, self.target)
        if state_spec != expected:
            raise ValueError("ConfusionCounts state spec must match its fixed class-domain matrix.")
        return state_spec


class AccuracyFromConfusion(Method):
    """Compute zero-safe scalar accuracy from one validated count matrix.

    Args:
        matrix: A non-empty square native signed-integer count matrix.

    Returns:
        A native float64 scalar; an all-zero matrix returns zero. Aggregate
        arithmetic promotes counts before summing, so large int64 totals retain
        float64 rounding semantics rather than wrapping.

    Raises:
        TypeError: If the matrix is not a supported native signed-integer tensor.
        ValueError: If the matrix is empty, non-square, or contains negative counts.

    Side Effects:
        None. The matrix remains native and is not mutated.
    """

    def __call__(self, matrix: object):
        """Return native diagonal-over-total accuracy without changing the matrix."""

        backend = _validate_count_matrix(matrix)
        matrix_float = _cast_float64(matrix, backend)
        total = _sum(matrix_float, backend)
        if _native_bool(total == 0):
            return _zeros((), backend, dtype="float64")
        return _sum(_diagonal(matrix_float, backend), backend) / total

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
        Aggregate arithmetic promotes counts before summing, so large int64
        totals retain float64 rounding semantics rather than wrapping.

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

    @staticmethod
    def __dryml_normalize_definition_arguments__(args, kwargs):
        """Validate fully bound F1 controls during Definition concretization."""

        _validate_known_f1_controls(
            kwargs.get("average"), kwargs.get("positive_index")
        )
        return args, kwargs

    def __call__(self, matrix: object):
        """Return native F1 values using zero for unsupported per-class terms."""

        backend = _validate_count_matrix(matrix)
        classes = int(matrix.shape[0])
        if self.average == "binary" and (classes != 2 or self.positive_index >= classes):
            raise ValueError("binary F1 requires exactly two classes and a valid positive_index.")
        matrix_float = _cast_float64(matrix, backend)
        total = _sum(matrix_float, backend)
        if _native_bool(total == 0):
            return _zeros((classes,), backend, dtype="float64") if self.average == "none" else _zeros((), backend, dtype="float64")
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
            return 2.0 * _sum(true_positive, backend) / (2.0 * total)
        if self.average == "macro":
            return _sum(per_class, backend) / float(classes)
        return _sum(per_class * support, backend) / total

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

    return _symbolic_evaluation_source(
        test_ds,
        model,
        x=x,
        y=y,
        prediction_labels=prediction_labels,
        target_labels=target_labels,
    ).concretize()


def _error_source(test_ds: object, model: object, *, x: Path, y: Path, error: type[Method]) -> ConcreteDefinition:
    """Declare one projected error stream and concretize it without materializing inputs."""

    evaluation = _concrete_evaluation_source(test_ds, model, x=x, y=y)
    return Map.defn(evaluation, Pipe.defn(Diff.defn(x="prediction", y="target"), error.defn())).concretize()


@_metric_factory
def regressor_mae(test_ds: Ref[AutoRef], model: Ref[AutoRef], *, x: Path = "x", y: Path = "y", mode: ReductionMode) -> Fold | Definition:
    """Declare an inert streaming MAE Fold over one model evaluation graph.

    Args:
        test_ds: Non-materializing evaluation Dataset reference, Definition, or
            Expr placeholder.
        model: Non-materializing model Method reference, Definition, or Expr
            placeholder.
        x: Source path selecting model input.
        y: Source path selecting regression target.
        mode: Global or coordinate-wise mean population definition.

    Returns:
        An uncomputed Fold using U6's native mean carry, or an inert Definition
        when any supplied argument contains a Definition or symbolic Expr,
        including compound expressions. Exact reference inputs alone retain the
        Fold return form; expression-dependent validation is deferred to binding.

    Raises:
        TypeError: If references or paths cannot form the declared Method graph.
        ValueError: If ``mode`` is not a supported mean population definition.

    Side Effects:
        None. It neither saves nor materializes either input, evaluates the
        model, or begins Dataset iteration.
    """

    return mean(_error_source(test_ds, model, x=x, y=y, error=Abs), mode=mode)


@_metric_factory
def regressor_mse(test_ds: Ref[AutoRef], model: Ref[AutoRef], *, x: Path = "x", y: Path = "y", mode: ReductionMode) -> Fold | Definition:
    """Declare an inert streaming MSE Fold over one model evaluation graph.

    Args:
        test_ds: Non-materializing evaluation Dataset reference, Definition, or
            Expr placeholder.
        model: Non-materializing model Method reference, Definition, or Expr
            placeholder.
        x: Source path selecting model input.
        y: Source path selecting regression target.
        mode: Global or coordinate-wise mean population definition.

    Returns:
        An uncomputed Fold using U6's native mean carry, or an inert Definition
        when any supplied argument contains a Definition or symbolic Expr,
        including compound expressions. Exact reference inputs alone retain the
        Fold return form; expression-dependent validation is deferred to binding.

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


@_metric_factory
def classifier_confusion_matrix(test_ds: Ref[AutoRef], model: Ref[AutoRef], *, classes: tuple[Label, ...], prediction_labels: Method, target_labels: Method, x: Path = "x", y: Path = "y") -> Fold | Definition:
    """Declare an inert fixed-domain confusion-matrix Fold without label guessing.

    Args:
        test_ds: Non-materializing evaluation Dataset reference, Definition, or
            Expr placeholder.
        model: Non-materializing model Method reference, Definition, or Expr
            placeholder.
        classes: Ordered finite domain of decoded labels.
        prediction_labels: Declared Method converting model output to labels.
        target_labels: Declared Method converting targets to labels.
        x: Source path selecting model input.
        y: Source path selecting target values.

    Returns:
        An uncomputed Fold whose result is a truth-row, prediction-column matrix,
        or an inert Definition when any supplied argument contains a Definition
        or symbolic Expr, including compound expressions. Exact reference inputs
        alone retain the Fold form; expression-dependent validation is deferred
        to binding.

    Raises:
        TypeError: If references, labels, or Methods are unsupported.
        ValueError: If ``classes`` is empty, mixed, or duplicated.

    Side Effects:
        None. Construction does not save or materialize inputs or decode labels.
    """

    return _classifier_fold(test_ds, model, classes=classes, prediction_labels=prediction_labels, target_labels=target_labels, x=x, y=y, finalize=None)


@_metric_factory
def classifier_accuracy(test_ds: Ref[AutoRef], model: Ref[AutoRef], *, classes: tuple[Label, ...], prediction_labels: Method, target_labels: Method, x: Path = "x", y: Path = "y") -> Fold | Definition:
    """Declare an inert confusion-based accuracy Fold without a second evaluation loop.

    Args:
        test_ds: Non-materializing evaluation Dataset reference, Definition, or
            Expr placeholder.
        model: Non-materializing model Method reference, Definition, or Expr
            placeholder.
        classes: Ordered finite domain of decoded labels.
        prediction_labels: Declared Method converting model output to labels.
        target_labels: Declared Method converting targets to labels.
        x: Source path selecting model input.
        y: Source path selecting target values.

    Returns:
        An uncomputed Fold whose result is native scalar accuracy, or an inert
        Definition when any supplied argument contains a Definition or symbolic
        Expr, including compound expressions. Exact reference inputs alone retain
        the Fold form; expression-dependent validation is deferred to binding.

    Raises:
        TypeError: If references, labels, or Methods are unsupported.
        ValueError: If ``classes`` is empty, mixed, or duplicated.

    Side Effects:
        None. Construction does not save or materialize inputs or decode labels.
    """

    return _classifier_fold(test_ds, model, classes=classes, prediction_labels=prediction_labels, target_labels=target_labels, x=x, y=y, finalize=AccuracyFromConfusion())


@_metric_factory
def classifier_f1(test_ds: Ref[AutoRef], model: Ref[AutoRef], *, classes: tuple[Label, ...], prediction_labels: Method, target_labels: Method, average: F1Average, positive_index: int | None = None, x: Path = "x", y: Path = "y") -> Fold | Definition:
    """Declare an inert confusion-based F1 Fold using explicit label conversion.

    Args:
        test_ds: Non-materializing evaluation Dataset reference, Definition, or
            Expr placeholder.
        model: Non-materializing model Method reference, Definition, or Expr
            placeholder.
        classes: Ordered finite domain of decoded labels.
        prediction_labels: Declared Method converting model output to labels.
        target_labels: Declared Method converting targets to labels.
        average: Binary, micro, macro, weighted, or per-class F1 mode.
        positive_index: Required positive class coordinate for binary F1 only.
        x: Source path selecting model input.
        y: Source path selecting target values.

    Returns:
        An uncomputed Fold with scalar or per-class native F1 result semantics,
        or an inert Definition when any supplied argument contains a Definition
        or symbolic Expr, including compound expressions. Exact reference inputs
        alone retain the Fold form; expression-dependent validation is deferred
        to binding.

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
