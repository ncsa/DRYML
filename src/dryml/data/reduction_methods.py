"""Backend-native numerical Method programs used by streaming reductions.

The public classes in this module keep numerical observations and carry values in
their originating NumPy, Torch CPU, or TensorFlow CPU backend.  They deliberately
do not collect a Dataset or move an intermediate result to the host.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any, Literal, TypeAlias

import numpy as np

from dryml.core.tensor_spec import SpecTree, TensorSpec
from dryml.methods import Accumulator, Method, traits


ReductionMode: TypeAlias = Literal["global", "coordinate"]
Path: TypeAlias = str | int | tuple[str | int, ...]
_MASK32 = 0xFFFFFFFF
_MAX_INT64 = (1 << 63) - 1
_THREEFRY_PARITY = 0x1BD11BDA
_ROTATIONS = (13, 15, 26, 6, 17, 29, 16, 24)


def _backend(value: object) -> str:
    """Return the supported dense CPU backend for one runtime tensor."""

    if isinstance(value, (np.ndarray, np.generic)):
        return "numpy"
    module = type(value).__module__
    if module.startswith("torch"):
        if value.device.type != "cpu":
            raise ValueError("Numerical reductions support Torch CPU tensors only.")
        return "torch"
    if module.startswith("tensorflow"):
        return "tf"
    raise TypeError("Numerical reductions require dense NumPy, Torch, or TensorFlow tensors.")


def _dtype_name(value: object) -> str:
    """Return the canonical dtype spelling without converting tensor contents."""

    dtype = getattr(value, "dtype", None)
    name = getattr(dtype, "name", None)
    if isinstance(name, str):
        return name
    text = str(dtype)
    return text.removeprefix("torch.")


def _validate_numeric(value: object, *, allow_bool: bool, name: str) -> str:
    """Validate one dense finite supported tensor and return its backend."""

    backend = _backend(value)
    dtype = _dtype_name(value)
    allowed = {"int8", "int16", "int32", "int64", "float32", "float64"}
    if allow_bool:
        allowed.add("bool")
    if dtype not in allowed:
        raise TypeError(f"{name} does not support dtype {dtype!r}.")
    if any(int(dimension) == 0 for dimension in getattr(value, "shape", ())):
        raise ValueError(f"{name} does not accept zero-sized tensors.")
    if dtype.startswith("float") and not _native_all_finite(value, backend):
        raise ValueError(f"{name} does not accept non-finite values.")
    return backend


def _native_all_finite(value: object, backend: str) -> bool:
    """Evaluate a bounded validation predicate without converting tensor payloads."""

    if backend == "numpy":
        return bool(np.all(np.isfinite(value)))
    if backend == "torch":
        import torch

        return bool(torch.all(torch.isfinite(value)))
    import tensorflow as tf

    return bool(tf.reduce_all(tf.math.is_finite(value)))


def _cast_float64(value: object, backend: str):
    """Promote supported values to native float64 before arithmetic."""

    if backend == "numpy":
        return value.astype(np.float64, copy=False)
    if backend == "torch":
        import torch

        return value.to(dtype=torch.float64)
    import tensorflow as tf

    return tf.cast(value, tf.float64)


def _cast_int64(value: object, backend: str):
    """Cast metadata to native signed int64."""

    if backend == "numpy":
        return np.array(value, dtype=np.int64)
    if backend == "torch":
        import torch

        return torch.as_tensor(value, dtype=torch.int64)
    import tensorflow as tf

    return tf.cast(value, tf.int64)


def _scalar(value: int | float, backend: str, *, dtype: str = "int64"):
    """Create one native scalar without deriving it from numerical input data."""

    if backend == "numpy":
        return np.array(value, dtype=np.int64 if dtype == "int64" else np.float64)
    if backend == "torch":
        import torch

        return torch.tensor(value, dtype=torch.int64 if dtype == "int64" else torch.float64)
    import tensorflow as tf

    return tf.constant(value, dtype=tf.int64 if dtype == "int64" else tf.float64)


def _zeros(shape: tuple[int, ...], backend: str, *, dtype: str):
    """Allocate one native carry buffer from resolved shape metadata."""

    if backend == "numpy":
        return np.zeros(shape, dtype=np.float64 if dtype == "float64" else np.int64)
    if backend == "torch":
        import torch

        return torch.zeros(shape, dtype=torch.float64 if dtype == "float64" else torch.int64)
    import tensorflow as tf

    return tf.zeros(shape, dtype=tf.float64 if dtype == "float64" else tf.int64)


def _sum(value: object, backend: str, axis=None):
    """Reduce a native tensor without host conversion."""

    if backend == "numpy":
        return np.sum(value, axis=axis, dtype=np.float64)
    if backend == "torch":
        import torch

        return torch.sum(value, dim=axis, dtype=torch.float64)
    import tensorflow as tf

    return tf.reduce_sum(value, axis=axis)


def _require_finite(value: object, backend: str, name: str) -> None:
    """Reject a native non-finite intermediate before it enters later state."""

    if not _native_all_finite(value, backend):
        raise ValueError(f"{name} produced a non-finite result.")


def _shape_after_axis(shape: tuple[object, ...] | None, axis: int | tuple[int, ...] | None) -> tuple[object, ...] | None:
    """Return the retained static shape after validating array-reduction axes."""

    if shape is None:
        return None
    axes = _normalize_axes(axis, len(shape))
    return tuple(size for index, size in enumerate(shape) if index not in axes)


def _normalize_axes(axis: int | tuple[int, ...] | None, rank: int) -> tuple[int, ...]:
    """Normalize explicit axes and reject duplicates or out-of-range values."""

    if axis is None:
        return tuple(range(rank))
    raw = (axis,) if type(axis) is int else axis
    if not isinstance(raw, tuple) or not raw:
        raise TypeError("axis must be an int, non-empty tuple of ints, or None.")
    result = []
    for item in raw:
        if type(item) is not int:
            raise TypeError("axis entries must be exact ints.")
        normalized = item + rank if item < 0 else item
        if normalized < 0 or normalized >= rank or normalized in result:
            raise ValueError("axis contains an invalid or duplicate dimension.")
        result.append(normalized)
    return tuple(result)


def _path(value: Any, path: Path) -> Any:
    """Select a declared nested path without constructing a helper Method."""

    result = value
    for key in path if isinstance(path, tuple) else (path,):
        result = result[key]
    return result


def _same_shape(left: object, right: object, name: str) -> str:
    """Require matching supported backend, dtype family, and exact runtime shape."""

    backend = _validate_numeric(left, allow_bool=True, name=name)
    other = _validate_numeric(right, allow_bool=True, name=name)
    if backend != other or tuple(left.shape) != tuple(right.shape):
        raise ValueError(f"{name} requires equal backend and shape without broadcasting.")
    return backend


class Diff(Method):
    """Subtract equal-shaped selected values after float64 promotion.

    Args:
        x: Path selecting the minuend from one structured item.
        y: Path selecting the subtrahend from one structured item.

    Returns:
        A native float64 tensor with the selected shape.

    Raises:
        TypeError: If either value has an unsupported dtype or boolean arithmetic is requested.
        ValueError: If shapes/backends differ or arithmetic becomes non-finite.
    """

    def __init__(self, *, x: Path = "x", y: Path = "y") -> None:
        self.x, self.y = x, y

    def __call__(self, item: Any):
        """Return the selected float64 difference without implicit broadcasting."""

        left, right = _path(item, self.x), _path(item, self.y)
        backend = _same_shape(left, right, "Diff")
        if _dtype_name(left) == "bool" or _dtype_name(right) == "bool":
            raise TypeError("Diff does not support boolean arithmetic.")
        result = _cast_float64(left, backend) - _cast_float64(right, backend)
        _require_finite(result, backend, "Diff")
        return result

    def infer_output_spec(self, input_spec: SpecTree) -> SpecTree:
        """Infer an unchanged selected shape with required float64 dtype."""

        left, right = _path(input_spec, self.x), _path(input_spec, self.y)
        if not isinstance(left, TensorSpec) or not isinstance(right, TensorSpec) or left.shape != right.shape:
            raise ValueError("Diff requires equal TensorSpec leaves.")
        return left.with_dtype("float64")


class Abs(Method):
    """Return the float64 absolute value of one supported numerical tensor."""

    def __call__(self, value: object):
        """Promote then compute native absolute values."""

        backend = _validate_numeric(value, allow_bool=False, name="Abs")
        result = _abs(_cast_float64(value, backend), backend)
        _require_finite(result, backend, "Abs")
        return result

    def infer_output_spec(self, input_spec: SpecTree) -> SpecTree:
        """Infer float64 output preserving the input tensor shape."""

        if not isinstance(input_spec, TensorSpec):
            raise TypeError("Abs requires one TensorSpec.")
        return input_spec.with_dtype("float64")


class Squared(Method):
    """Return the float64 square of one supported numerical tensor."""

    def __call__(self, value: object):
        """Promote then square natively, rejecting floating overflow."""

        backend = _validate_numeric(value, allow_bool=False, name="Squared")
        value = _cast_float64(value, backend)
        result = value * value
        _require_finite(result, backend, "Squared")
        return result

    def infer_output_spec(self, input_spec: SpecTree) -> SpecTree:
        """Infer float64 output preserving the input tensor shape."""

        if not isinstance(input_spec, TensorSpec):
            raise TypeError("Squared requires one TensorSpec.")
        return input_spec.with_dtype("float64")


class Equal(Method):
    """Compare equal-shaped selected values without backend conversion or broadcasting."""

    def __init__(self, *, x: Path = "x", y: Path = "y") -> None:
        self.x, self.y = x, y

    def __call__(self, item: Any):
        """Return native elementwise equality for the two selected values."""

        left, right = _path(item, self.x), _path(item, self.y)
        backend = _same_shape(left, right, "Equal")
        if backend == "numpy":
            return np.equal(left, right)
        if backend == "torch":
            return left == right
        import tensorflow as tf

        return tf.equal(left, right)

    def infer_output_spec(self, input_spec: SpecTree) -> SpecTree:
        """Infer boolean output preserving the selected input shape."""

        left, right = _path(input_spec, self.x), _path(input_spec, self.y)
        if not isinstance(left, TensorSpec) or not isinstance(right, TensorSpec) or left.shape != right.shape:
            raise ValueError("Equal requires equal TensorSpec leaves.")
        return left.with_dtype("bool")


class ArrayMean(Method):
    """Compute an exact native float64 mean across validated array axes.

    Args:
        axis: ``None`` for all axes or one exact axis/unique tuple of axes.

    Returns:
        A native float64 scalar or tensor with reduced axes removed.

    Raises:
        TypeError: If the dtype or axis declaration is unsupported.
        ValueError: If selected axes have an empty population or values are non-finite.
    """

    def __init__(self, *, axis: int | tuple[int, ...] | None = None) -> None:
        self.axis = axis

    def __call__(self, values: object):
        """Compute one native exact array mean after promotion."""

        backend = _validate_numeric(values, allow_bool=True, name="ArrayMean")
        axes = _normalize_axes(self.axis, len(values.shape))
        promoted = _cast_float64(values, backend)
        total = _sum(promoted, backend, axis=None if not axes else (axes if len(axes) > 1 else axes[0]))
        _require_finite(total, backend, "ArrayMean")
        population = 1
        for axis in axes:
            population *= int(values.shape[axis])
        result = total / _scalar(population, backend, dtype="float64")
        _require_finite(result, backend, "ArrayMean")
        return result

    def infer_output_spec(self, input_spec: SpecTree) -> SpecTree:
        """Infer reduced float64 shape without executing an array operation."""

        if not isinstance(input_spec, TensorSpec):
            raise TypeError("ArrayMean requires one TensorSpec.")
        return input_spec.with_dtype("float64").with_shape(_shape_after_axis(input_spec.shape, self.axis))


class ArrayQuantile(Method):
    """Compute exact bounded-array linear quantiles in the input backend.

    Args:
        q: One finite quantile or a non-empty tuple of finite quantiles in ``[0, 1]``.
        axis: ``None`` for all axes or one exact axis/unique tuple of axes.

    Returns:
        Native float64 quantiles. A tuple request adds a leading requested-quantile axis.

    Raises:
        TypeError: If dtype, axes, or request types are unsupported.
        ValueError: If a request is out of range, selected data are empty, or values are non-finite.
    """

    def __init__(self, q: float | tuple[float, ...], *, axis: int | tuple[int, ...] | None = None) -> None:
        self.q = _normalize_quantiles(q)
        self._plural = isinstance(q, tuple)
        self.axis = axis

    def __call__(self, values: object):
        """Sort and linearly interpolate one bounded native input array."""

        backend = _validate_numeric(values, allow_bool=False, name="ArrayQuantile")
        axes = _normalize_axes(self.axis, len(values.shape))
        return _native_quantile(_cast_float64(values, backend), backend, self.q, axes, plural=self._plural)

    def infer_output_spec(self, input_spec: SpecTree) -> SpecTree:
        """Infer float64 result shape and plural leading quantile dimension."""

        if not isinstance(input_spec, TensorSpec):
            raise TypeError("ArrayQuantile requires one TensorSpec.")
        shape = _shape_after_axis(input_spec.shape, self.axis)
        if self._plural and shape is not None:
            shape = (len(self.q), *shape)
        return input_spec.with_dtype("float64").with_shape(shape)


def _abs(value: object, backend: str):
    """Apply native absolute value."""

    if backend == "numpy":
        return np.abs(value)
    if backend == "torch":
        import torch

        return torch.abs(value)
    import tensorflow as tf

    return tf.abs(value)


def _normalize_quantiles(q: float | tuple[float, ...]) -> tuple[float, ...]:
    """Validate and preserve scalar/tuple quantile request order and duplicates."""

    values = (q,) if type(q) is float else q
    if not isinstance(values, tuple) or not values:
        raise TypeError("q must be a finite float or non-empty tuple of floats.")
    for value in values:
        if type(value) is not float or not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("q values must be finite exact floats in [0, 1].")
    return values


def _native_quantile(values: object, backend: str, q: tuple[float, ...], axes: tuple[int, ...], *, plural: bool):
    """Perform native linear interpolation, keeping all intermediate values native."""

    rank = len(values.shape)
    retained_axes = tuple(index for index in range(rank) if index not in axes)
    permutation = (*retained_axes, *axes)
    reduced = 1
    for axis in axes:
        reduced *= int(values.shape[axis])
    retained_shape = tuple(int(values.shape[axis]) for axis in retained_axes)
    ordered = _sort_last(_reshape(_transpose(values, permutation, backend), (*retained_shape, reduced), backend), backend)
    requests = _native_requests(q, backend)
    positions = requests * _scalar(reduced - 1, backend, dtype="float64")
    lower, upper = _floor_to_int64(positions, backend), _ceil_to_int64(positions, backend)
    lower_values = _gather_last(ordered, lower, backend)
    upper_values = _gather_last(ordered, upper, backend)
    fraction = _reshape(positions - _cast_float64(lower, backend), (1,) * len(retained_shape) + (len(q),), backend)
    span = upper_values - lower_values
    _require_finite(span, backend, "ArrayQuantile")
    result = lower_values + fraction * span
    _require_finite(result, backend, "ArrayQuantile")
    # Gather yields retained dimensions followed by requested q.  Public plural
    # results place requested q first, consistently across all native backends.
    result = _transpose(result, (len(retained_shape), *range(len(retained_shape))), backend)
    return result if plural else result[0]


def _transpose(value: object, axes: tuple[int, ...], backend: str):
    """Permute a native tensor using shape metadata rather than host data."""

    if backend == "numpy":
        return np.transpose(value, axes)
    if backend == "torch":
        return value.permute(axes)
    import tensorflow as tf

    return tf.transpose(value, axes)


def _sort_last(value: object, backend: str):
    """Sort independent flattened reduction populations along their final axis."""

    if backend == "numpy":
        return np.sort(value, axis=-1)
    if backend == "torch":
        import torch

        return torch.sort(value, dim=-1).values
    import tensorflow as tf

    return tf.sort(value, axis=-1)


def _gather_last(value: object, indexes: object, backend: str):
    """Gather native interpolation endpoints from a final sorted axis."""

    if backend == "numpy":
        return np.take(value, indexes, axis=-1)
    if backend == "torch":
        return value.index_select(-1, indexes)
    import tensorflow as tf

    return tf.gather(value, indexes, axis=-1)


class MeanInitial(Method):
    """Allocate independent float64 sum/int64 count state for a streaming mean.

    Args:
        mode: ``"global"`` to reduce every scalar, or ``"coordinate"`` to retain
            each non-batch coordinate independently.
    """

    def __init__(self, *, mode: ReductionMode) -> None:
        self.mode = _normalize_mode(mode)

    @traits(batch_mode="element")
    def element(self, observation: object):
        """Allocate state for one unbatched observation."""

        return _mean_initial(observation, self.mode, batched=False)

    @traits(batch_mode="batched")
    def batched(self, observation: object):
        """Allocate state for one explicitly batched observation."""

        return _mean_initial(observation, self.mode, batched=True)

    def infer_output_spec(self, observation_spec: SpecTree) -> SpecTree:
        """Infer fixed sum/count state from declared observation and batch facts."""

        if not isinstance(observation_spec, TensorSpec):
            raise TypeError("MeanInitial requires one TensorSpec.")
        _validate_numeric_spec(observation_spec, allow_bool=True, name="Mean")
        shape = () if self.mode == "global" else observation_spec.shape
        return (TensorSpec("float64", shape=shape, backend=observation_spec.backend), TensorSpec("int64", shape=(), backend=observation_spec.backend))


class MeanUpdate(Accumulator):
    """Advance native sum/count state once for each observation or declared batch."""

    def __init__(self, *, mode: ReductionMode) -> None:
        self.mode = _normalize_mode(mode)

    @traits(batch_mode="element")
    def element(self, observation: object, state: tuple[object, object]):
        """Add every unbatched scalar or coordinate exactly once."""

        return _mean_update(observation, state, self.mode, batched=False)

    @traits(batch_mode="batched")
    def batched(self, observation: object, state: tuple[object, object]):
        """Weight every batch by its true scalar or row population."""

        return _mean_update(observation, state, self.mode, batched=True)

    def infer_output_spec(self, observation_spec: SpecTree, state_spec: SpecTree) -> SpecTree:
        """Require and retain the initializer's fixed two-tensor carry specification."""

        if not isinstance(state_spec, tuple) or len(state_spec) != 2:
            raise TypeError("MeanUpdate requires a (sum, count) state specification.")
        return state_spec


class MeanFinalize(Method):
    """Divide native float64 sum by native int64 count after a complete Fold."""

    def __call__(self, state: tuple[object, object]):
        """Return native float64 mean after Fold has established a non-empty stream."""

        total, count = state
        backend = _backend(total)
        _validate_mean_carry(total, count, backend, "Mean finalization")
        if bool(count <= _scalar(0, backend)):
            raise ValueError("Mean finalization requires a positive count.")
        result = total / _cast_float64(count, backend)
        _require_finite(result, backend, "Mean finalization")
        return result

    def infer_output_spec(self, state_spec: SpecTree) -> SpecTree:
        """Infer the float64 sum shape from valid mean state specs."""

        if not isinstance(state_spec, tuple) or len(state_spec) != 2 or not isinstance(state_spec[0], TensorSpec):
            raise TypeError("MeanFinalize requires a (sum, count) state specification.")
        return state_spec[0]


def _normalize_mode(mode: ReductionMode) -> ReductionMode:
    """Validate the explicit Dataset reduction population mode."""

    if mode not in ("global", "coordinate"):
        raise ValueError("mode must be 'global' or 'coordinate'.")
    return mode


def _validate_numeric_spec(spec: TensorSpec, *, allow_bool: bool, name: str) -> None:
    """Reject unsupported numerical contracts before Method selection."""

    allowed = {"int8", "int16", "int32", "int64", "float32", "float64"}
    if allow_bool:
        allowed.add("bool")
    if spec.layout.value != "dense" or spec.dtype.name not in allowed:
        raise TypeError(f"{name} does not support the declared dtype/layout.")


def _mean_initial(observation: object, mode: ReductionMode, *, batched: bool):
    """Allocate one invocation-local mean carry from actual shape facts."""

    backend = _validate_numeric(observation, allow_bool=True, name="Mean")
    shape = () if mode == "global" else tuple(int(item) for item in observation.shape[1 if batched else 0:])
    return _zeros(shape, backend, dtype="float64"), _scalar(0, backend)


def _validate_mean_carry(total: object, count: object, backend: str, name: str) -> None:
    """Require finite float64 totals and scalar int64 counts from one backend."""

    if _backend(total) != backend or _backend(count) != backend:
        raise ValueError(f"{name} requires matching native carry tensors.")
    if _dtype_name(total) != "float64" or _dtype_name(count) != "int64" or tuple(count.shape) != ():
        raise TypeError(f"{name} requires float64 sum and scalar int64 count carry.")
    _require_finite(total, backend, name)
    if bool(count < _scalar(0, backend)):
        raise ValueError(f"{name} requires a nonnegative count.")


def _mean_update(observation: object, state: tuple[object, object], mode: ReductionMode, *, batched: bool):
    """Advance fixed native mean carry without averaging batch means."""

    total, count = state
    backend = _validate_numeric(observation, allow_bool=True, name="Mean")
    _validate_mean_carry(total, count, backend, "Mean")
    values = _cast_float64(observation, backend)
    if mode == "global":
        increment = math.prod(int(item) for item in observation.shape)
        addition = _sum(values, backend)
    elif batched:
        increment = int(observation.shape[0])
        addition = _sum(values, backend, axis=0)
    else:
        increment = 1
        addition = values
    if tuple(total.shape) != tuple(addition.shape):
        raise ValueError("Mean observation coordinate shape changed during traversal.")
    _require_finite(addition, backend, "Mean")
    if increment > _MAX_INT64 or bool(count > _scalar(_MAX_INT64 - increment, backend)):
        raise OverflowError("Mean count would wrap int64.")
    next_total = total + addition
    _require_finite(next_total, backend, "Mean")
    return next_total, count + _scalar(increment, backend)


def _u32(value: object, backend: str):
    """Mask signed int64 limbs to their unsigned 32-bit representation."""

    mask = _scalar(_MASK32, backend)
    if backend == "numpy":
        return np.bitwise_and(value, mask)
    if backend == "torch":
        import torch

        return torch.bitwise_and(value, mask)
    import tensorflow as tf

    return tf.bitwise.bitwise_and(value, mask)


def _xor(left: object, right: object, backend: str):
    """Apply native signed-int64 xor to bounded 32-bit limbs."""

    if backend == "numpy":
        return np.bitwise_xor(left, right)
    if backend == "torch":
        import torch

        return torch.bitwise_xor(left, right)
    import tensorflow as tf

    return tf.bitwise.bitwise_xor(left, right)


def _rotl32(value: object, amount: int, backend: str):
    """Rotate a masked signed-int64 limb left by one fixed 32-bit amount."""

    if backend == "numpy":
        return _u32(np.left_shift(value, amount) | np.right_shift(value, 32 - amount), backend)
    if backend == "torch":
        return _u32((value << amount) | (value >> (32 - amount)), backend)
    import tensorflow as tf

    return _u32(tf.bitwise.bitwise_or(tf.bitwise.left_shift(value, amount), tf.bitwise.right_shift(value, 32 - amount)), backend)


def _pair(left: object, right: object, backend: str):
    """Stack two native scalar limbs in high-word-first order."""

    if backend == "numpy":
        return np.stack((left, right), axis=-1)
    if backend == "torch":
        import torch

        return torch.stack((left, right), dim=-1)
    import tensorflow as tf

    return tf.stack((left, right), axis=-1)


def _threefry2x32_20(counter: object, key: object):
    """Return Random123 v1.14 Threefry2x32-20 output using signed-int64 limbs.

    Args:
        counter: Native two-limb high-word-first counter tensor.
        key: Native two-limb high-word-first key tensor.

    Returns:
        Native two-limb unsigned-32 values represented in signed int64 tensors.

    Raises:
        TypeError: If the pair tensors are not from one supported backend.
        ValueError: If their final dimension is not exactly two.
    """

    backend = _backend(counter)
    if _backend(key) != backend or tuple(counter.shape)[-1:] != (2,) or tuple(key.shape)[-1:] != (2,):
        raise ValueError("Threefry requires matching native (..., 2) counter and key tensors.")
    c0, c1 = _u32(counter[..., 0], backend), _u32(counter[..., 1], backend)
    k0, k1 = _u32(key[..., 0], backend), _u32(key[..., 1], backend)
    k2 = _xor(_xor(k0, k1, backend), _scalar(_THREEFRY_PARITY, backend), backend)
    keys = (k0, k1, k2)
    x0, x1 = _u32(c0 + k0, backend), _u32(c1 + k1, backend)
    for round_index in range(20):
        x0 = _u32(x0 + x1, backend)
        x1 = _xor(_rotl32(x1, _ROTATIONS[round_index % 8], backend), x0, backend)
        if round_index % 4 == 3:
            schedule = round_index // 4 + 1
            x0 = _u32(x0 + keys[schedule % 3], backend)
            x1 = _u32(x1 + keys[(schedule + 1) % 3] + _scalar(schedule, backend), backend)
    return _pair(x0, x1, backend)


class ReservoirInitial(Method):
    """Allocate fixed-capacity Algorithm R state with native key/counter metadata."""

    def __init__(self, *, mode: ReductionMode, capacity: int, seed: int = 0) -> None:
        self.mode = _normalize_mode(mode)
        self.capacity = _validate_capacity(capacity)
        self.seed = _validate_seed(seed)

    @traits(batch_mode="element")
    def element(self, observation: object):
        """Allocate state for an unbatched observation."""

        return _reservoir_initial(observation, self.mode, self.capacity, self.seed, batched=False)

    @traits(batch_mode="batched")
    def batched(self, observation: object):
        """Allocate state for an explicitly batched observation."""

        return _reservoir_initial(observation, self.mode, self.capacity, self.seed, batched=True)

    def infer_output_spec(self, observation_spec: SpecTree) -> SpecTree:
        """Infer bounded reservoir shape and scalar metadata without allocation."""

        if not isinstance(observation_spec, TensorSpec):
            raise TypeError("ReservoirInitial requires one TensorSpec.")
        _validate_numeric_spec(observation_spec, allow_bool=False, name="Quantile")
        shape = (self.capacity,) if self.mode == "global" else (self.capacity, *(observation_spec.shape or ()))
        backend = observation_spec.backend
        return (TensorSpec("float64", shape=shape, backend=backend), TensorSpec("int64", shape=(), backend=backend), TensorSpec("int64", shape=(), backend=backend), TensorSpec("int64", shape=(2,), backend=backend))


class ReservoirUpdate(Accumulator):
    """Advance fixed Algorithm R reservoir state in logical stream order."""

    def __init__(self, *, mode: ReductionMode, capacity: int) -> None:
        self.mode = _normalize_mode(mode)
        self.capacity = _validate_capacity(capacity)

    @traits(batch_mode="element")
    def element(self, observation: object, state: tuple[object, object, object, object]):
        """Advance every scalar in an unbatched observation."""

        return _reservoir_update(observation, state, self.mode, self.capacity, batched=False)

    @traits(batch_mode="batched")
    def batched(self, observation: object, state: tuple[object, object, object, object]):
        """Advance every row/scalar in an explicitly batched observation."""

        return _reservoir_update(observation, state, self.mode, self.capacity, batched=True)

    def infer_output_spec(self, observation_spec: SpecTree, state_spec: SpecTree) -> SpecTree:
        """Retain the fixed bounded reservoir carry specification."""

        if not isinstance(state_spec, tuple) or len(state_spec) != 4:
            raise TypeError("ReservoirUpdate requires reservoir/count/counter/key state.")
        return state_spec


class ReservoirQuantile(Method):
    """Extract requested linear quantiles from one completed bounded reservoir."""

    def __init__(self, q: float | tuple[float, ...]) -> None:
        self.q = _normalize_quantiles(q)
        self._plural = isinstance(q, tuple)

    def __call__(self, state: tuple[object, object, object, object]):
        """Sort retained values natively and return linear sample quantiles."""

        reservoir, population, _, _ = state
        backend = _backend(reservoir)
        _validate_reservoir_final_state(state, backend)
        return _reservoir_quantile(reservoir, population, backend, self.q, plural=self._plural)

    def infer_output_spec(self, state_spec: SpecTree) -> SpecTree:
        """Infer scalar/coordinate extraction shape from reservoir state."""

        if not isinstance(state_spec, tuple) or len(state_spec) != 4 or not isinstance(state_spec[0], TensorSpec):
            raise TypeError("ReservoirQuantile requires reservoir/count/counter/key state.")
        shape = state_spec[0].shape
        result_shape = None if shape is None else shape[1:]
        if self._plural and result_shape is not None:
            result_shape = (len(self.q), *result_shape)
        return TensorSpec("float64", shape=result_shape, backend=state_spec[0].backend)


def _validate_capacity(capacity: int) -> int:
    """Validate the positive exact bounded-reservoir capacity."""

    if type(capacity) is not int or capacity <= 0:
        raise ValueError("capacity must be a positive exact int.")
    return capacity


def _validate_seed(seed: int) -> int:
    """Validate the nonnegative exact 63-bit reservoir seed."""

    if type(seed) is not int or not 0 <= seed <= _MAX_INT64:
        raise ValueError("seed must be an exact int in [0, 2**63 - 1].")
    return seed


def _reservoir_initial(observation: object, mode: ReductionMode, capacity: int, seed: int, *, batched: bool):
    """Allocate bounded reservoir and native Algorithm R metadata from shape facts."""

    backend = _validate_numeric(observation, allow_bool=False, name="Quantile")
    coordinate = tuple(int(item) for item in observation.shape[1 if batched else 0:])
    shape = (capacity,) if mode == "global" else (capacity, *coordinate)
    key = _cast_int64((seed >> 32, seed & _MASK32), backend)
    return _zeros(shape, backend, dtype="float64"), _scalar(0, backend), _scalar(0, backend), key


def _validate_reservoir_carry(
    reservoir: object,
    population: object,
    counter: object,
    key: object,
    backend: str,
    capacity: int,
    coordinate: tuple[int, ...],
) -> None:
    """Validate the fixed native reservoir layout before a transition mutates it."""

    if any(_backend(value) != backend for value in (reservoir, population, counter, key)):
        raise ValueError("Quantile observation and carry backend changed.")
    if (
        _dtype_name(reservoir) != "float64"
        or _dtype_name(population) != "int64"
        or _dtype_name(counter) != "int64"
        or _dtype_name(key) != "int64"
        or tuple(reservoir.shape) != (capacity, *coordinate)
        or tuple(population.shape) != ()
        or tuple(counter.shape) != ()
        or tuple(key.shape) != (2,)
    ):
        raise ValueError("Quantile carry shape or dtype changed during traversal.")
    _require_finite(reservoir, backend, "Quantile")
    if (
        bool(population < _scalar(0, backend))
        or bool(population > _scalar(_MAX_INT64, backend))
        or bool(counter < _scalar(0, backend))
        or bool(counter > _scalar(_MAX_INT64, backend))
    ):
        raise ValueError("Quantile carry metadata is outside signed int64 bounds.")


def _validate_reservoir_final_state(state: tuple[object, object, object, object], backend: str) -> None:
    """Validate completed reservoir metadata before native final extraction."""

    reservoir, population, counter, key = state
    capacity = int(reservoir.shape[0])
    _validate_reservoir_carry(
        reservoir,
        population,
        counter,
        key,
        backend,
        capacity,
        tuple(int(item) for item in reservoir.shape[1:]),
    )
    if bool(population <= _scalar(0, backend)):
        raise ValueError("Quantile finalization requires a positive population.")


def _reservoir_update(observation: object, state: tuple[object, object, object, object], mode: ReductionMode, capacity: int, *, batched: bool):
    """Run Algorithm R without extracting observations or carry values to host."""

    reservoir, population, counter, key = state
    backend = _validate_numeric(observation, allow_bool=False, name="Quantile")
    coordinate = () if mode == "global" else tuple(int(item) for item in observation.shape[1 if batched else 0:])
    _validate_reservoir_carry(reservoir, population, counter, key, backend, capacity, coordinate)
    if bool(population >= _scalar(_MAX_INT64, backend)):
        raise OverflowError("Quantile population count would wrap int64.")
    if bool(population >= _scalar(capacity, backend)) and bool(counter >= _scalar(_MAX_INT64, backend)):
        raise OverflowError("Quantile draw counter would wrap int64.")
    values = _cast_float64(observation, backend)
    if backend == "tf":
        item_count = (
            math.prod(int(item) for item in values.shape)
            if mode == "global"
            else (int(values.shape[0]) if batched else 1)
        )
        if item_count <= capacity and bool(population <= _scalar(capacity - item_count, backend)):
            return _tf_reservoir_fill(values, state, mode, batched=batched)
        return _tf_reservoir_update(values, state, mode, capacity, batched=batched)
    if mode == "global":
        flat = _reshape(values, (-1,), backend)
        for index in range(int(flat.shape[0])):
            reservoir, population, counter = _reservoir_one(flat[index], reservoir, population, counter, key, capacity, backend)
        return reservoir, population, counter, key
    rows = values if batched else _reshape(values, (1, *tuple(int(item) for item in values.shape)), backend)
    for index in range(int(rows.shape[0])):
        reservoir, population, counter = _reservoir_one(rows[index], reservoir, population, counter, key, capacity, backend)
    return reservoir, population, counter, key


def _tf_reservoir_update(values: object, state: tuple[object, object, object, object], mode: ReductionMode, capacity: int, *, batched: bool):
    """Advance one TensorFlow batch with native control flow and no eager scalar loop.

    TensorFlow's eager per-item dispatch is prohibitively expensive for the fixed
    qualification population.  ``tf.while_loop`` is ordinary backend control flow,
    not JIT/XLA compilation, and retains the same item/draw order as the NumPy and
    Torch transition paths.
    """

    reservoir, population, counter, key = state
    import tensorflow as tf

    rows = tf.reshape(values, (-1,)) if mode == "global" else (
        values if batched else tf.expand_dims(values, 0)
    )
    return _tf_reservoir_loop(mode, capacity, len(reservoir.shape))(rows, reservoir, population, counter, key)


def _tf_reservoir_fill(values: object, state: tuple[object, object, object, object], mode: ReductionMode, *, batched: bool):
    """Fill a wholly fitting TensorFlow batch without tracing the sampler branch."""

    reservoir, population, counter, key = state
    import tensorflow as tf

    rows = tf.reshape(values, (-1,)) if mode == "global" else (
        values if batched else tf.expand_dims(values, 0)
    )
    row_count = int(rows.shape[0])
    indexes = tf.range(population, population + tf.constant(row_count, dtype=tf.int64), dtype=tf.int64)
    updated = tf.tensor_scatter_nd_update(reservoir, tf.expand_dims(indexes, 1), rows)
    return updated, population + tf.constant(row_count, dtype=tf.int64), counter, key


@lru_cache(maxsize=16)
def _tf_reservoir_loop(mode: ReductionMode, capacity: int, reservoir_rank: int):
    """Build one non-XLA TensorFlow control-flow kernel for a fixed state layout.

    The graph is deliberately limited to Algorithm R's already-declared Method
    transition.  It does not compile or fuse Dataset/Fold programs, and
    ``jit_compile=False`` explicitly excludes XLA/JIT execution.
    """

    import tensorflow as tf

    @tf.function(jit_compile=False, reduce_retracing=True)
    def run(rows, reservoir, population, counter, key):
        capacity_value = tf.constant(capacity, dtype=tf.int64)
        max_value = tf.constant(_MAX_INT64, dtype=tf.int64)

        def body(index, current_reservoir, current_population, current_counter):
            value = rows[index]
            population_check = tf.debugging.assert_less(
                current_population, max_value,
                message="Quantile population count would wrap int64.",
            )
            with tf.control_dependencies((population_check,)):
                next_population = tf.identity(current_population) + tf.constant(1, dtype=tf.int64)
            fill = current_population < capacity_value

            def draw():
                return _tf_bounded_index(key, current_counter, next_population)

            selected, next_counter = tf.cond(
                fill,
                lambda: (current_population, current_counter),
                draw,
            )
            safe = tf.where(selected < capacity_value, selected, tf.constant(0, dtype=tf.int64))
            updated = tf.tensor_scatter_nd_update(
                current_reservoir,
                tf.reshape(safe, (1, 1)),
                tf.expand_dims(value, 0),
            )
            replace = tf.logical_or(fill, selected < capacity_value)
            return index + 1, tf.where(replace, updated, current_reservoir), next_population, next_counter

        _, next_reservoir, next_population, next_counter = tf.while_loop(
            lambda index, *_: index < tf.shape(rows)[0],
            body,
            (tf.constant(0, dtype=tf.int32), reservoir, population, counter),
            parallel_iterations=1,
        )
        return next_reservoir, next_population, next_counter, key

    return run


def _tf_bounded_index(key: object, counter: object, population: object):
    """Run TensorFlow's rejection sampler with one native counter increment per draw."""

    import tensorflow as tf

    max_value = tf.constant(_MAX_INT64, dtype=tf.int64)
    population_checks = (
        tf.debugging.assert_greater_equal(population, tf.constant(1, dtype=tf.int64), message="Quantile population must be in [1, 2**63 - 1]."),
        tf.debugging.assert_less_equal(population, max_value, message="Quantile population must be in [1, 2**63 - 1]."),
    )
    with tf.control_dependencies(population_checks):
        checked_population = tf.identity(population)
        limit = (max_value // checked_population) * checked_population

    def condition(current_counter, random):
        return random >= limit

    def body(current_counter, _):
        counter_check = tf.debugging.assert_less(
            current_counter, max_value,
            message="Quantile draw counter would wrap int64.",
        )
        with tf.control_dependencies((counter_check,)):
            checked_counter = tf.identity(current_counter)
        output = _threefry2x32_20(_counter_pair(checked_counter, "tf"), key)
        random = (
            tf.bitwise.bitwise_and(output[0], tf.constant(0x7FFFFFFF, dtype=tf.int64))
            * tf.constant(1 << 32, dtype=tf.int64)
            + output[1]
        )
        return checked_counter + 1, random

    first_counter, first_random = body(counter, tf.constant(0, dtype=tf.int64))
    final_counter, random = tf.while_loop(
        condition,
        body,
        (first_counter, first_random),
        parallel_iterations=1,
    )
    return tf.math.floormod(random, checked_population), final_counter


def _reservoir_one(value: object, reservoir: object, population: object, counter: object, key: object, capacity: int, backend: str):
    """Advance one scalar/global item or one coordinate row using tensor decisions."""

    if bool(population >= _scalar(_MAX_INT64, backend)):
        raise OverflowError("Quantile population count would wrap int64.")
    next_population = population + _scalar(1, backend)
    fill = population < _scalar(capacity, backend)
    if bool(fill):
        return _scatter_first_axis(reservoir, population, value, backend), next_population, counter
    index, next_counter = _bounded_index(key, counter, next_population, backend)
    if bool(index < _scalar(capacity, backend)):
        return _scatter_first_axis(reservoir, index, value, backend), next_population, next_counter
    return reservoir, next_population, next_counter


def _bounded_index(key: object, counter: object, population: object, backend: str):
    """Sample an unbiased Algorithm R index and advance every attempted draw."""

    if bool(population < _scalar(1, backend)) or bool(population > _scalar(_MAX_INT64, backend)):
        raise ValueError("Quantile population must be in [1, 2**63 - 1].")
    if bool(counter < _scalar(0, backend)):
        raise ValueError("Quantile draw counter must be nonnegative.")
    limit = (_scalar(_MAX_INT64, backend) // population) * population
    current = counter
    while True:
        if bool(current >= _scalar(_MAX_INT64, backend)):
            raise OverflowError("Quantile draw counter would wrap int64.")
        output = _threefry2x32_20(_counter_pair(current, backend), key)
        current = current + _scalar(1, backend)
        random = (_u32(output[0], backend) & _scalar(0x7FFFFFFF, backend)) * _scalar(1 << 32, backend) + _u32(output[1], backend)
        if bool(random < limit):
            return random % population, current


def _counter_pair(counter: object, backend: str):
    """Encode a native 63-bit draw counter in high-word-first limb order."""

    return _pair(counter // _scalar(1 << 32, backend), counter % _scalar(1 << 32, backend), backend)


def _reshape(value: object, shape: tuple[int, ...], backend: str):
    """Reshape a native tensor using only already-known shape facts."""

    if backend == "numpy":
        return value.reshape(shape)
    if backend == "torch":
        return value.reshape(shape)
    import tensorflow as tf

    return tf.reshape(value, shape)


def _where(condition: object, left: object, right: object, backend: str):
    """Select native values without a Python numerical branch."""

    if backend == "numpy":
        return np.where(condition, left, right)
    if backend == "torch":
        import torch

        return torch.where(condition, left, right)
    import tensorflow as tf

    return tf.where(condition, left, right)


def _scatter_first_axis(reservoir: object, index: object, value: object, backend: str):
    """Return a copied native reservoir with one first-axis row replaced."""

    if backend == "numpy":
        result = reservoir.copy()
        result[index] = value
        return result
    if backend == "torch":
        import torch

        selector = index.reshape(1)
        expanded = selector.reshape((1,) + (1,) * (reservoir.ndim - 1)).expand((1, *reservoir.shape[1:]))
        return reservoir.scatter(0, expanded, value.reshape((1, *reservoir.shape[1:])))
    import tensorflow as tf

    return tf.tensor_scatter_nd_update(reservoir, tf.reshape(index, (1, 1)), tf.expand_dims(value, 0))


def _reservoir_quantile(reservoir: object, population: object, backend: str, q: tuple[float, ...], *, plural: bool):
    """Extract quantiles from valid rows without host slicing native state.

    Unused rows are replaced with infinity before sorting, so they sort after every
    retained finite value.  The native count then controls only native gather
    positions; no carry scalar crosses the host boundary.
    """

    capacity = int(reservoir.shape[0])
    retained = _minimum(population, _scalar(capacity, backend), backend)
    rows = _arange(capacity, backend)
    mask = rows < retained
    if len(reservoir.shape) > 1:
        mask = _reshape(mask, (capacity,) + (1,) * (len(reservoir.shape) - 1), backend)
    infinity = _scalar(float("inf"), backend, dtype="float64")
    values = _where(mask, reservoir, infinity, backend)
    ordered = _sort(values, backend)
    requests = _native_requests(q, backend)
    positions = requests * _cast_float64(retained - _scalar(1, backend), backend)
    lower = _floor_to_int64(positions, backend)
    upper = _ceil_to_int64(positions, backend)
    lower_values = _gather_rows(ordered, lower, backend)
    upper_values = _gather_rows(ordered, upper, backend)
    fraction = positions - _cast_float64(lower, backend)
    if len(reservoir.shape) > 1:
        fraction = _reshape(fraction, (len(q),) + (1,) * (len(reservoir.shape) - 1), backend)
    span = upper_values - lower_values
    _require_finite(span, backend, "ReservoirQuantile")
    result = lower_values + fraction * span
    _require_finite(result, backend, "ReservoirQuantile")
    return result if plural else result[0]


def _minimum(left: object, right: object, backend: str):
    """Return a native scalar minimum used for bounded reservoir occupancy."""

    if backend == "numpy":
        return np.minimum(left, right)
    if backend == "torch":
        import torch

        return torch.minimum(left, right)
    import tensorflow as tf

    return tf.minimum(left, right)


def _arange(stop: int, backend: str):
    """Create native signed-int64 row indices from fixed capacity metadata."""

    if backend == "numpy":
        return np.arange(stop, dtype=np.int64)
    if backend == "torch":
        import torch

        return torch.arange(stop, dtype=torch.int64)
    import tensorflow as tf

    return tf.range(stop, dtype=tf.int64)


def _sort(value: object, backend: str):
    """Sort a reservoir along its retained-row axis in the native backend."""

    if backend == "numpy":
        return np.sort(value, axis=0)
    if backend == "torch":
        import torch

        return torch.sort(value, dim=0).values
    import tensorflow as tf

    return tf.sort(value, axis=0)


def _native_requests(q: tuple[float, ...], backend: str):
    """Create native float64 request metadata from validated factory configuration."""

    if backend == "numpy":
        return np.array(q, dtype=np.float64)
    if backend == "torch":
        import torch

        return torch.tensor(q, dtype=torch.float64)
    import tensorflow as tf

    return tf.constant(q, dtype=tf.float64)


def _floor_to_int64(value: object, backend: str):
    """Floor native float64 positions into native int64 gather indexes."""

    if backend == "numpy":
        return np.floor(value).astype(np.int64)
    if backend == "torch":
        import torch

        return torch.floor(value).to(torch.int64)
    import tensorflow as tf

    return tf.cast(tf.floor(value), tf.int64)


def _ceil_to_int64(value: object, backend: str):
    """Ceil native float64 positions into native int64 gather indexes."""

    if backend == "numpy":
        return np.ceil(value).astype(np.int64)
    if backend == "torch":
        import torch

        return torch.ceil(value).to(torch.int64)
    import tensorflow as tf

    return tf.cast(tf.math.ceil(value), tf.int64)


def _gather_rows(value: object, indexes: object, backend: str):
    """Gather one or more native first-axis rows without host index extraction."""

    if backend == "numpy":
        return value[indexes]
    if backend == "torch":
        return value[indexes]
    import tensorflow as tf

    return tf.gather(value, indexes)


__all__ = [
    "Abs", "ArrayMean", "ArrayQuantile", "Diff", "Equal", "MeanFinalize",
    "MeanInitial", "MeanUpdate", "Path", "ReductionMode", "ReservoirInitial",
    "ReservoirQuantile", "ReservoirUpdate", "Squared",
]
