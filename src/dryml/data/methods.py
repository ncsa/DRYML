from __future__ import annotations

from dryml.core.methods import Method, traits
from dryml.core.dtype import normalize_dtype
from dryml.core.tensor_spec import Dynamic, SpecTree, map_spec_tree


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

    def bind_first(self, first_value, *, input_spec=None):
        def bind_tree(tree):
            if isinstance(tree, dict):
                bound = {}
                first = {}
                for key, value in tree.items():
                    _validate_tree_key(key)
                    bound[key], first[key] = bind_tree(value)
                return bound, first
            if isinstance(tree, tuple):
                pairs = tuple(bind_tree(value) for value in tree)
                return tuple(pair[0] for pair in pairs), tuple(pair[1] for pair in pairs)
            if isinstance(tree, list):
                pairs = [bind_tree(value) for value in tree]
                return [pair[0] for pair in pairs], [pair[1] for pair in pairs]
            return tree.bind_first(first_value, input_spec=input_spec)

        bound_branches, first_out = bind_tree(self.branches)

        def bound_project(x):
            return _map_method_tree(bound_branches, lambda method: method(x))

        return bound_project, first_out


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

    def bind_first(self, first_value, *, input_spec=None):
        bound_methods = []
        value = first_value
        spec = input_spec

        for method in self.methods:
            if hasattr(method, "bind_first"):
                bound, value = method.bind_first(value, input_spec=spec)
            else:
                bound = method
                value = bound(value)
            bound_methods.append(bound)

            if spec is not None and hasattr(method, "infer_output_spec"):
                spec = method.infer_output_spec(spec)
            else:
                spec = None

        def bound_pipe(x):
            result = x
            for method in bound_methods:
                result = method(result)
            return result

        return bound_pipe, value


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
        from dryml.runtime import import_configured_framework
        tf = import_configured_framework("tensorflow")
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

    @traits(backend="numpy")
    def numpy_call(self, x):
        import numpy as np

        return np.argmax(x, axis=self.axis)

    @traits(backend="numpy", batch_mode="batched")
    def numpy_batched(self, x):
        import numpy as np

        return np.argmax(x, axis=self._runtime_axis(batched=True))

    @traits(backend="tf")
    def tf_call(self, x):
        from dryml.runtime import import_configured_framework
        tf = import_configured_framework("tensorflow")

        return tf.argmax(x, axis=self.axis, output_type=tf.int64)

    @traits(backend="tf", batch_mode="batched")
    def tf_batched(self, x):
        from dryml.runtime import import_configured_framework
        tf = import_configured_framework("tensorflow")

        return tf.argmax(x, axis=self._runtime_axis(batched=True), output_type=tf.int64)

    @traits(backend="torch")
    def torch_call(self, x):
        from dryml.runtime import import_configured_framework
        torch = import_configured_framework("torch")

        return torch.argmax(x, dim=self.axis)

    @traits(backend="torch", batch_mode="batched")
    def torch_batched(self, x):
        from dryml.runtime import import_configured_framework
        torch = import_configured_framework("torch")

        return torch.argmax(x, dim=self._runtime_axis(batched=True))


class Flatten(Method):
    def infer_output_spec(self, input_spec: SpecTree) -> SpecTree:
        return map_spec_tree(input_spec, lambda spec: spec.with_shape(_flat_shape(spec.shape)))

    @traits(backend="numpy")
    def numpy_call(self, x):
        return x.reshape(-1)

    @traits(backend="numpy", batch_mode="batched")
    def numpy_batched(self, x):
        return x.reshape((x.shape[0], -1))

    @traits(backend="tf")
    def tf_call(self, x):
        from dryml.runtime import import_configured_framework
        tf = import_configured_framework("tensorflow")

        return tf.reshape(x, [-1])

    @traits(backend="tf", batch_mode="batched")
    def tf_batched(self, x):
        from dryml.runtime import import_configured_framework
        tf = import_configured_framework("tensorflow")

        return tf.reshape(x, [tf.shape(x)[0], -1])

    @traits(backend="torch")
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
        from dryml.runtime import import_configured_framework
        tf = import_configured_framework("tensorflow")

        x = tf.cast(x, self.dtype.tf())
        return (x - self.mean) / self.std

    @traits(backend="torch")
    def torch_call(self, x):
        x = x.to(self.dtype.torch())
        return (x - self.mean) / self.std
