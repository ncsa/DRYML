from __future__ import annotations

from dataclasses import replace

from dryml.core.dtype import normalize_dtype
from dryml.core.tensor_spec import Dynamic, SpecTree, map_spec_tree
from dryml.methods import Method, traits


def _validate_tree_key(key):
    if not isinstance(key, (str, int)):
        raise TypeError(f"Project dict keys must be str or int, got {type(key).__name__}.")


def _map_method_tree(tree, fn):
    if isinstance(tree, dict):
        for key in tree:
            _validate_tree_key(key)
        return {k: _map_method_tree(v, fn) for k, v in tree.items()}
    if isinstance(tree, tuple):
        return tuple(_map_method_tree(v, fn) for v in tree)
    if isinstance(tree, list):
        return [_map_method_tree(v, fn) for v in tree]
    return fn(tree)


def _iter_method_leaves(tree):
    if isinstance(tree, dict):
        for key, v in tree.items():
            _validate_tree_key(key)
            yield from _iter_method_leaves(v)
        return
    if isinstance(tree, (tuple, list)):
        for v in tree:
            yield from _iter_method_leaves(v)
        return
    yield tree


def _normalize_path(idxs):
    if len(idxs) == 1 and isinstance(idxs[0], (tuple, list)):
        return tuple(idxs[0])
    return tuple(idxs)


def _pack_project_tree(args, kwargs):
    if args and kwargs:
        raise ValueError("Project accepts positional branches or keyword branches, not both.")
    if kwargs:
        return dict(kwargs)
    if not args:
        raise ValueError("Project requires at least one branch.")
    if len(args) == 1:
        return args[0]
    return tuple(args)


def _validate_method_leaf(method):
    if not isinstance(method, Method):
        raise TypeError(f"Project expects Method leaves, got {type(method).__name__}.")
    return method


# Elementwise methods

class Select(Method):
    def __init__(self, *idxs):
        self.idxs = _normalize_path(idxs)

    @classmethod
    def from_path(cls, path):
        return cls(path)

    def __call__(self, x):
        result = x
        for idx in self.idxs:
            result = result[idx]
        return result

    def infer_output_spec(self, input_spec: SpecTree) -> SpecTree:
        result = input_spec
        for idx in self.idxs:
            result = result[idx]
        return result


class Project(Method):
    """Build a new element structure from one input element using branch Methods."""

    def __init__(self, *branches, **named_branches):
        self.branches = _pack_project_tree(branches, named_branches)
        if not tuple(_iter_method_leaves(self.branches)):
            raise ValueError("Project requires at least one Method leaf.")
        _map_method_tree(self.branches, _validate_method_leaf)

    def __call__(self, x):
        return _map_method_tree(self.branches, lambda method: method(x))

    def infer_output_spec(self, input_spec: SpecTree) -> SpecTree:
        return _map_method_tree(
            self.branches,
            lambda method: method.infer_output_spec(input_spec),
        )

    def find_implementation(self, input_spec=None, *, backend=None, batch_mode=None):
        """Select each branch once using one shared completed input specification.

        Args:
            input_spec: Normalized source constraint for every branch.
            backend: Optional completed backend constraint.
            batch_mode: Optional completed batch-mode constraint.

        Returns:
            A selected Project callable that validates its source input then invokes
            each locally selected branch without further selection.

        Raises:
            ImplementationSelectionError: If the Project or any branch cannot be
            uniquely selected from the supplied constraints.
        """

        implementation = super().find_implementation(
            input_spec,
            backend=backend,
            batch_mode=batch_mode,
        )
        return self._specialize_implementation(
            implementation,
            input_spec,
            backend=backend,
            batch_mode=batch_mode,
            derive_spec_batch=True,
        )

    def _prepare_implementation(self, input_spec, *, backend, batch_mode):
        """Build one learning-time Project plan without guessing batch intent."""

        implementation = super()._prepare_implementation(
            input_spec,
            backend=backend,
            batch_mode=batch_mode,
        )
        return self._specialize_implementation(
            implementation,
            input_spec,
            backend=backend,
            batch_mode=batch_mode,
            derive_spec_batch=False,
        )

    def _specialize_implementation(
        self,
        implementation,
        input_spec,
        *,
        backend,
        batch_mode,
        derive_spec_batch,
    ):
        """Attach branch-local selected callables to one Project implementation."""

        if input_spec is None:
            return implementation
        if derive_spec_batch:
            def select(method):
                return method.find_implementation(
                    input_spec,
                    backend=backend,
                    batch_mode=batch_mode,
                )
        else:
            def select(method):
                return method._prepare_implementation(
                    input_spec,
                    backend=backend,
                    batch_mode=batch_mode,
                )
        selected_branches = _map_method_tree(
            self.branches,
            select,
        )

        def invoke_project(x):
            return _map_method_tree(selected_branches, lambda method: method(x))

        return replace(implementation, _invoker=invoke_project)


class Pipe(Method):
    def __init__(self, *methods):
        if not methods:
            raise ValueError("Pipe requires at least one Method.")
        self.methods = methods

    def __call__(self, x):
        result = x
        for method in self.methods:
            result = method(result)
        return result

    def infer_output_spec(self, input_spec: SpecTree) -> SpecTree:
        spec = input_spec
        for method in self.methods:
            spec = method.infer_output_spec(spec)
        return spec

    def find_implementation(self, input_spec=None, *, backend=None, batch_mode=None):
        """Select each child once while threading pure intermediate specifications.

        Args:
            input_spec: Normalized source constraint for the first child.
            backend: Optional completed backend constraint.
            batch_mode: Optional completed batch-mode constraint.

        Returns:
            A selected Pipe callable that validates its source input then invokes
            the locally selected children in declaration order.

        Raises:
            ImplementationSelectionError: If the Pipe or any child cannot be
            uniquely selected from the threaded constraints.
        """

        implementation = super().find_implementation(
            input_spec,
            backend=backend,
            batch_mode=batch_mode,
        )
        return self._specialize_implementation(
            implementation,
            input_spec,
            backend=backend,
            batch_mode=batch_mode,
            derive_spec_batch=True,
        )

    def _prepare_implementation(self, input_spec, *, backend, batch_mode):
        """Build one learning-time Pipe plan without guessing batch intent."""

        implementation = super()._prepare_implementation(
            input_spec,
            backend=backend,
            batch_mode=batch_mode,
        )
        return self._specialize_implementation(
            implementation,
            input_spec,
            backend=backend,
            batch_mode=batch_mode,
            derive_spec_batch=False,
        )

    def _specialize_implementation(
        self,
        implementation,
        input_spec,
        *,
        backend,
        batch_mode,
        derive_spec_batch,
    ):
        """Attach spec-threaded child callables to one Pipe implementation."""

        if input_spec is None:
            return implementation
        selected_methods = []
        spec = input_spec
        for index, method in enumerate(self.methods):
            child_backend = backend if index == 0 else None
            selected_methods.append(
                method.find_implementation(
                    spec,
                    backend=child_backend,
                    batch_mode=batch_mode,
                )
                if derive_spec_batch
                else method._prepare_implementation(
                    spec,
                    backend=child_backend,
                    batch_mode=batch_mode,
                )
            )
            spec = method.infer_output_spec(spec)

        def invoke_pipe(x):
            result = x
            for method in selected_methods:
                result = method(result)
            return result

        return replace(implementation, _invoker=invoke_pipe)


class Cast(Method):
    def __init__(self, dtype):
        self.dtype = normalize_dtype(dtype)

    def infer_output_spec(self, input_spec: SpecTree) -> SpecTree:
        return map_spec_tree(input_spec, lambda spec: spec.with_dtype(self.dtype))

    @traits(backend="numpy")
    def numpy_call(self, x):
        return x.astype(self.dtype.np())

    @traits(backend="tf")
    def tf_call(self, x):
        import tensorflow as tf
        return tf.cast(x, self.dtype.tf())

    @traits(backend="torch")
    def torch_call(self, x):
        return x.to(self.dtype.torch())


def _flat_shape(shape):
    if shape is None:
        return None
    size = 1
    for dim in shape:
        if dim is Dynamic:
            return (Dynamic,)
        size *= int(dim)
    return (size,)


def _argmax_shape(shape, axis):
    if shape is None:
        return None
    rank = len(shape)
    if rank == 0:
        raise ValueError("ArgMax requires a non-scalar input shape.")
    if axis < 0:
        axis += rank
    if axis < 0 or axis >= rank:
        raise ValueError(f"ArgMax axis {axis} is out of bounds for rank {rank}.")
    return (*shape[:axis], *shape[axis + 1:])


class ArgMax(Method):
    def __init__(self, axis: int = -1):
        if not isinstance(axis, int):
            raise TypeError("ArgMax axis must be an int.")
        self.axis = axis

    def _runtime_axis(self, *, batched: bool):
        return self.axis + 1 if batched and self.axis >= 0 else self.axis

    def infer_output_spec(self, input_spec: SpecTree) -> SpecTree:
        return map_spec_tree(
            input_spec,
            lambda spec: spec.with_dtype("int64").with_shape(_argmax_shape(spec.shape, self.axis)),
        )

    @traits(backend="numpy", batch_mode="element")
    def numpy_call(self, x):
        import numpy as np

        return np.argmax(x, axis=self.axis)

    @traits(backend="numpy", batch_mode="batched")
    def numpy_batched(self, x):
        import numpy as np

        return np.argmax(x, axis=self._runtime_axis(batched=True))

    @traits(backend="tf", batch_mode="element")
    def tf_call(self, x):
        import tensorflow as tf

        return tf.argmax(x, axis=self.axis, output_type=tf.int64)

    @traits(backend="tf", batch_mode="batched")
    def tf_batched(self, x):
        import tensorflow as tf

        return tf.argmax(x, axis=self._runtime_axis(batched=True), output_type=tf.int64)

    @traits(backend="torch", batch_mode="element")
    def torch_call(self, x):
        import torch

        return torch.argmax(x, dim=self.axis)

    @traits(backend="torch", batch_mode="batched")
    def torch_batched(self, x):
        import torch

        return torch.argmax(x, dim=self._runtime_axis(batched=True))


class Flatten(Method):
    def infer_output_spec(self, input_spec: SpecTree) -> SpecTree:
        return map_spec_tree(input_spec, lambda spec: spec.with_shape(_flat_shape(spec.shape)))

    @traits(backend="numpy", batch_mode="element")
    def numpy_call(self, x):
        return x.reshape(-1)

    @traits(backend="numpy", batch_mode="batched")
    def numpy_batched(self, x):
        return x.reshape((x.shape[0], -1))

    @traits(backend="tf", batch_mode="element")
    def tf_call(self, x):
        import tensorflow as tf

        return tf.reshape(x, [-1])

    @traits(backend="tf", batch_mode="batched")
    def tf_batched(self, x):
        import tensorflow as tf

        return tf.reshape(x, [tf.shape(x)[0], -1])

    @traits(backend="torch", batch_mode="element")
    def torch_call(self, x):
        return x.reshape(-1)

    @traits(backend="torch", batch_mode="batched")
    def torch_batched(self, x):
        return x.reshape((x.shape[0], -1))


class Scale(Method):
    def __init__(self, mean=0.0, std=1.0, *, dtype="float32"):
        if std == 0:
            raise ValueError("std must be non-zero.")
        self.mean = mean
        self.std = std
        self.dtype = normalize_dtype(dtype)

    @classmethod
    def from_range(cls, min=0.0, max=1.0, *, dtype="float32"):
        if max == min:
            raise ValueError("max and min must differ.")
        return cls(mean=min, std=max - min, dtype=dtype)

    def infer_output_spec(self, input_spec: SpecTree) -> SpecTree:
        return map_spec_tree(input_spec, lambda spec: spec.with_dtype(self.dtype))

    @traits(backend="numpy")
    def numpy_call(self, x):
        return (x.astype(self.dtype.np()) - self.mean) / self.std

    @traits(backend="tf")
    def tf_call(self, x):
        import tensorflow as tf

        x = tf.cast(x, self.dtype.tf())
        return (x - self.mean) / self.std

    @traits(backend="torch")
    def torch_call(self, x):
        x = x.to(self.dtype.torch())
        return (x - self.mean) / self.std
