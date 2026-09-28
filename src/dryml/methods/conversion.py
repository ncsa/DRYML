"""Local dense host-value handoff planning and execution.

This module records one direct NumPy, TensorFlow, or Torch adapter edge for a
prepared Method boundary.  It is deliberately dependency-light while planning:
optional frameworks are imported only when a selected adapter converts values.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module

import numpy as np

from dryml.core.backend import Backend
from dryml.core.tensor_spec import Layout, SpecTree, TensorSpec, map_spec_tree
from dryml.core.utils.recurse import map_leaf_groups, map_leaves


_DENSE_BACKENDS = frozenset((Backend.numpy, Backend.tf, Backend.torch))
_EXACT_DTYPE_NAMES = frozenset({
    "bool", "int8", "int16", "int32", "int64", "uint8",
    "float16", "float32", "float64",
})


@dataclass(frozen=True, slots=True)
class ConversionEdge:
    """One inspectable direct dense host-copy handoff in a prepared graph.

    Args:
        producer_spec: Declared source value contract before conversion.
        consumer_spec: Declared target value contract after conversion.
        adapter: Stable direct adapter name, such as ``"numpy_to_torch"``.
        dtype_shape_batch_policy: Exact preservation policy for supported leaves.
        device_policy: Required source/target host-device policy.

    The edge is a planning fact, not evidence that a producer trait describes a
    runtime value.  Invocation validates actual leaves before target execution.
    """

    producer_spec: SpecTree
    consumer_spec: SpecTree
    adapter: str
    dtype_shape_batch_policy: str = "preserve_exact_dense_dtype_shape_batch"
    device_policy: str = "cpu_host_copy_only"


def backend_spec(spec: SpecTree, backend: Backend) -> SpecTree:
    """Return a same-shaped spec tree whose leaves declare ``backend``.

    Args:
        spec: Dense logical source specification tree.
        backend: Consumer backend for every tensor leaf.

    Returns:
        A new immutable spec tree retaining dtype, shape, layout, and batch facts.

    Raises:
        TypeError: If a leaf is not a supported dense tensor specification.
    """

    def convert(leaf: TensorSpec) -> TensorSpec:
        if leaf.layout is not Layout.DENSE:
            raise TypeError("Only dense tensor specs support local conversion.")
        if leaf.dtype.name not in _EXACT_DTYPE_NAMES:
            raise TypeError(f"Unsupported dense conversion dtype {leaf.dtype.name!r}.")
        return TensorSpec(
            leaf.dtype,
            shape=leaf.shape,
            batch=leaf.batch,
            backend=backend,
            layout=leaf.layout,
            axis_names=leaf.axis_names,
            batch_axis_name=leaf.batch_axis_name,
        )

    return map_spec_tree(spec, convert)


def direct_adapter(source: Backend | None, target: Backend | None) -> str | None:
    """Return the supported direct adapter name for distinct dense CPU backends.

    No route search, framework preference, or multi-edge conversion is performed.
    """

    if source not in _DENSE_BACKENDS or target not in _DENSE_BACKENDS or source == target:
        return None
    return f"{source.value}_to_{target.value}"


def make_edge(producer_spec: SpecTree, target_backend: Backend | str) -> ConversionEdge:
    """Create one validated direct conversion fact for a complete source spec."""

    target_backend = Backend(target_backend)
    source_backends = {
        leaf.backend
        for leaf in _iter_spec_leaves(producer_spec)
    }
    if len(source_backends) != 1 or None in source_backends:
        raise TypeError("Dense conversion requires one declared source backend.")
    source_backend = next(iter(source_backends))
    adapter = direct_adapter(source_backend, target_backend)
    if adapter is None:
        raise TypeError("No supported direct dense conversion adapter exists.")
    return ConversionEdge(producer_spec, backend_spec(producer_spec, target_backend), adapter)


def convert(edge: ConversionEdge, value: object) -> object:
    """Convert a supported nested value through one planned direct adapter.

    Args:
        edge: Immutable edge retained during preparation.
        value: Runtime tree matching the producer specification.

    Returns:
        An independent target-backend dense tree with unchanged structure, values,
        dtype, shape, and batch interpretation.

    Raises:
        TypeError: If values are mixed, sparse/ragged/object/inexact, non-CPU, or
            otherwise outside the dense host-copy handoff contract.
        RuntimeError: If a Torch value requiring gradients would cross a boundary.
    """

    source = _tree_backend(value)
    expected = _tree_backend_from_spec(edge.producer_spec)
    target = _tree_backend_from_spec(edge.consumer_spec)
    if source != expected:
        raise TypeError("Runtime value backend does not match the planned conversion producer.")
    if source is None or target is None:
        raise TypeError("Conversion edges require complete backend specifications.")
    _validate_runtime_tree(value, edge.producer_spec)
    host = map_leaves(value, lambda leaf: _host_copy(leaf, source))
    return map_leaves(host, lambda leaf: _target_copy(leaf, target))


def _iter_spec_leaves(spec: SpecTree):
    if isinstance(spec, TensorSpec):
        yield spec
    elif isinstance(spec, dict):
        for value in spec.values():
            yield from _iter_spec_leaves(value)
    elif isinstance(spec, (tuple, list)):
        for value in spec:
            yield from _iter_spec_leaves(value)
    else:
        raise TypeError(f"Expected TensorSpec tree, got {type(spec).__name__}.")


def _tree_backend_from_spec(spec: SpecTree) -> Backend | None:
    backends = {leaf.backend for leaf in _iter_spec_leaves(spec)}
    return next(iter(backends)) if len(backends) == 1 else None


def _tree_backend(value: object) -> Backend | None:
    leaves = list(_iter_value_leaves(value))
    backends = {_leaf_backend(leaf) for leaf in leaves}
    if None in backends or len(backends) != 1:
        return None
    return next(iter(backends))


def _iter_value_leaves(value: object):
    if isinstance(value, dict):
        for child in value.values():
            yield from _iter_value_leaves(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from _iter_value_leaves(child)
    else:
        yield value


def _leaf_backend(value: object) -> Backend | None:
    if isinstance(value, np.ndarray):
        return Backend.numpy
    module = type(value).__module__
    if module.startswith("tensorflow"):
        return Backend.tf
    if module.startswith("torch"):
        return Backend.torch
    return None


def _validate_runtime_tree(value: object, spec: SpecTree) -> None:
    try:
        map_leaf_groups((value, spec), lambda pair: _validate_leaf(*pair))
    except (TypeError, ValueError) as error:
        raise TypeError("Runtime conversion value does not match its dense spec tree.") from error


def _validate_leaf(value: object, spec: TensorSpec) -> object:
    if not isinstance(spec, TensorSpec):
        raise TypeError("Conversion requires TensorSpec leaves.")
    array = _host_copy(value, spec.backend)
    if array.dtype == np.dtype(object) or not spec.compatible_with_shape(array.shape):
        raise TypeError("Runtime conversion value has an unsupported dtype or shape.")
    if array.dtype != np.dtype(spec.dtype.np()):
        raise TypeError("Runtime conversion value changes the declared dtype.")
    return value


def _host_copy(value: object, backend: Backend | None) -> np.ndarray:
    if backend is Backend.numpy:
        if not isinstance(value, np.ndarray):
            raise TypeError("Expected a NumPy dense array for conversion.")
        array = value
    elif backend is Backend.tf:
        tf = import_module("tensorflow")
        if not tf.is_tensor(value) or isinstance(value, (tf.RaggedTensor, tf.SparseTensor)):
            raise TypeError("Only dense TensorFlow tensors support conversion.")
        if "CPU" not in (getattr(value, "device", "") or "CPU"):
            raise TypeError("Only CPU TensorFlow tensors support conversion.")
        array = value.numpy()
    elif backend is Backend.torch:
        torch = import_module("torch")
        if not isinstance(value, torch.Tensor) or value.layout is not torch.strided:
            raise TypeError("Only dense strided Torch tensors support conversion.")
        if value.device.type != "cpu":
            raise TypeError("Only CPU Torch tensors support conversion.")
        if value.requires_grad:
            raise RuntimeError("Torch tensors requiring gradients cannot cross a framework boundary.")
        array = value.numpy()
    else:
        raise TypeError("Unsupported conversion backend.")
    if array.dtype == np.dtype(object) or array.dtype.kind not in "?biufc":
        raise TypeError(f"Unsupported dense conversion dtype {array.dtype!s}.")
    # np.array provides independent writable C storage for read-only, transposed,
    # and negative-stride sources without reinterpreting their values.
    return np.array(array, copy=True, order="C")


def _target_copy(array: np.ndarray, backend: Backend) -> object:
    if backend is Backend.numpy:
        return np.array(array, copy=True, order="C")
    if backend is Backend.tf:
        tf = import_module("tensorflow")
        with tf.device("/CPU:0"):
            return tf.convert_to_tensor(array)
    if backend is Backend.torch:
        return import_module("torch").tensor(array, device="cpu")
    raise TypeError("Unsupported conversion backend.")


__all__ = ["ConversionEdge", "backend_spec", "convert", "direct_adapter", "make_edge"]
