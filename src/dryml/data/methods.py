"""Composable data Methods with pure spec inference and selected native execution."""

from __future__ import annotations

from dataclasses import replace
from math import prod

from dryml.core.dtype import normalize_dtype
from dryml.core.tensor_spec import Dynamic, Layout, SpecTree, map_spec_tree
from dryml.methods import ImplementationSelectionError, Method, traits


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
    @property
    def iteration_independent(self) -> bool:
        """Return whether discarded Select calls can be omitted safely."""

        return True

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
    """Build an element structure from selected branch Method occurrences.

    Branch selection is metadata-only. Executable preparation compiles each
    occurrence once and retains its invoker without changing child Method state.

    Args:
        *branches: Positional Method leaves or a supported dict/tuple/list tree.
        **named_branches: Named Method branches, exclusive of positional branches.

    Raises:
        ValueError: If branches are empty or positional/named forms are combined.
        TypeError: If a leaf is not a Method or a mapping key is not a str/int.
    """

    def __init__(self, *branches, **named_branches):
        self.branches = _pack_project_tree(branches, named_branches)
        if not tuple(_iter_method_leaves(self.branches)):
            raise ValueError("Project requires at least one Method leaf.")
        _map_method_tree(self.branches, _validate_method_leaf)

    @property
    def iteration_independent(self) -> bool:
        """Return true only when every branch leaf permits omitted calls."""

        return all(method.iteration_independent for method in _iter_method_leaves(self.branches))

    def __call__(self, x):
        return _map_method_tree(self.branches, lambda method: method(x))

    def infer_output_spec(self, input_spec: SpecTree, *additional_input_specs: SpecTree) -> SpecTree:
        """Infer branch output specs from exactly one logical input specification."""

        if additional_input_specs:
            raise TypeError("Project accepts exactly one input specification.")
        return _map_method_tree(
            self.branches,
            lambda method: method.infer_output_spec(input_spec),
        )

    def find_implementation(self, input_spec=None, *additional_input_specs, backend=None, batch_mode=None,
                            output_spec=None):
        """Select each branch once using one shared completed input specification.

        Args:
            input_spec: Normalized source constraint for every branch.
            *additional_input_specs: Unsupported because Project is unary.
            backend: Optional completed backend constraint.
            batch_mode: Optional completed batch-mode constraint.
            output_spec: Optional retained result constraint for the selected
                composite call.

        Returns:
            A selected Project callable that validates its source input then invokes
            each locally selected branch without further selection.

        Raises:
            ImplementationSelectionError: If the Project or any branch cannot be
            uniquely selected from the supplied constraints.

        Side Effects:
            Retains detached branch facts without compiling or invoking targets.
        """

        if additional_input_specs:
            raise ImplementationSelectionError("conflict")
        implementation = super().find_implementation(
            input_spec,
            backend=backend,
            batch_mode=batch_mode,
            output_spec=output_spec,
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
        from dryml.methods.signature import spec_from_node

        input_spec = spec_from_node(implementation.consumer_input_specs[0])
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
        children = tuple(_iter_method_leaves(selected_branches))
        indices = iter(range(len(children)))
        structure = _map_method_tree(selected_branches, lambda _: next(indices))

        def prepare_project(prepared):
            def invoke_project(x):
                return _map_method_tree(structure, lambda index: prepared[index](x))
            return invoke_project

        def infer_project(children):
            return _map_method_tree(structure, lambda index: children[index].infer_output_spec())

        return replace(implementation, _children=children, _invoker_factory=prepare_project,
                       _output_inference=infer_project)


class Pipe(Method):
    """Compose unary Methods in order through pure intermediate specifications.

    Args:
        *methods: Nonempty sequence of child Methods in execution order.

    Selection retains distinct child occurrences without compilation or mutation
    of child preparation state. Executable preparation retains each invoker once.

    Raises:
        ValueError: If no child Method is supplied.
    """

    def __init__(self, *methods):
        if not methods:
            raise ValueError("Pipe requires at least one Method.")
        self.methods = methods

    @property
    def iteration_independent(self) -> bool:
        """Return true only when every sequential child permits omitted calls."""

        return all(method.iteration_independent for method in self.methods)

    def __call__(self, x):
        result = x
        for method in self.methods:
            result = method(result)
        return result

    def infer_output_spec(self, input_spec: SpecTree, *additional_input_specs: SpecTree) -> SpecTree:
        """Infer sequential output specs from exactly one logical input specification."""

        if additional_input_specs:
            raise TypeError("Pipe accepts exactly one input specification.")
        spec = input_spec
        for method in self.methods:
            spec = method.infer_output_spec(spec)
        return spec

    def find_implementation(self, input_spec=None, *additional_input_specs, backend=None, batch_mode=None,
                            output_spec=None):
        """Select each child once while threading pure intermediate specifications.

        Args:
            input_spec: Normalized source constraint for the first child.
            *additional_input_specs: Unsupported because Pipe is unary.
            backend: Optional completed backend constraint.
            batch_mode: Optional completed batch-mode constraint.
            output_spec: Optional retained result constraint for the selected
                composite call.

        Returns:
            A selected Pipe callable that validates its source input then invokes
            the locally selected children in declaration order.

        Raises:
            ImplementationSelectionError: If the Pipe or any child cannot be
            uniquely selected from the threaded constraints.

        Side Effects:
            Infers and retains child occurrence facts without compiling signatures
            or invoking bodies; child direct-call state remains independent.
        """

        if additional_input_specs:
            raise ImplementationSelectionError("conflict")
        implementation = super().find_implementation(
            input_spec,
            backend=backend,
            batch_mode=batch_mode,
            output_spec=output_spec,
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
        from dryml.methods.signature import spec_from_node

        input_spec = spec_from_node(implementation.consumer_input_specs[0])
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
            spec = selected_methods[-1].infer_output_spec()
        return replace(implementation, _children=tuple(selected_methods),
                       _invoker_factory=Pipe._prepare_invoker,
                       _output_inference=lambda children: children[-1].infer_output_spec())

    @staticmethod
    def _prepare_invoker(prepared_methods):
        """Capture finalized invokers, never their owner-bearing selected carriers."""

        def invoke_pipe(x):
            result = x
            for method in prepared_methods:
                result = method(result)
            return result

        return invoke_pipe


class Cast(Method):
    """Convert one native dense or ragged tensor to a declared semantic dtype.

    Args:
        dtype: Canonical DRYML or supported native dtype spelling.

    Returns:
        A native value preserving shape, layout, batch and ragged partitions.
        NumPy ragged outputs own their buffers; native backend cast semantics
        otherwise apply. Pure inference maps the same dtype change over a tree.

    Raises:
        TypeError, ValueError: For unsupported dtypes, values or native casts.
    """
    @property
    def iteration_independent(self) -> bool:
        """Return whether discarded Cast calls can be omitted safely."""

        return True

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
    if 0 in shape:
        return (0,)
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
    """Select maximum indices along one sample axis, preserving the batch axis.

    Args:
        axis: Positive or negative sample-axis index, excluding the batch axis.

    Returns:
        A native int64 result with the selected sample axis removed. Pure output
        inference preserves remaining Dynamic dimensions and their axis names.

    Raises:
        TypeError: If axis is not an integer.
        ValueError: For a known scalar spec, an out-of-range axis, or a runtime
            reduction over an empty axis.
    """
    @property
    def iteration_independent(self) -> bool:
        """Return whether discarded ArgMax calls can be omitted safely."""

        return True

    def __init__(self, axis: int = -1):
        if not isinstance(axis, int):
            raise TypeError("ArgMax axis must be an int.")
        self.axis = axis

    def _runtime_axis(self, *, batched: bool):
        return self.axis + 1 if batched and self.axis >= 0 else self.axis

    def infer_output_spec(self, input_spec: SpecTree) -> SpecTree:
        """Remove the selected sample axis and its name without executing values.

        Args:
            input_spec: TensorSpec tree with known or unknown sample rank.

        Returns:
            The same tree with int64 leaves, the reduced shapes/names, and
            unchanged batch/backend metadata. Unknown ranks stay unknown.

        Raises:
            ValueError: If a known rank cannot admit the selected axis.
            TypeError: For ragged input, whose reduction contract is not defined.
        """
        def infer(spec):
            if spec.layout is Layout.RAGGED:
                raise TypeError("ArgMax does not yet define a ragged-axis reduction contract.")
            names = None if spec.axis_names is None else _argmax_shape(spec.axis_names, self.axis)
            return replace(spec, dtype="int64", shape=_argmax_shape(spec.shape, self.axis), axis_names=names)

        return map_spec_tree(input_spec, infer)

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
    """Flatten sample axes into one axis while retaining an explicit batch axis.

    Returns:
        A native vector per sample. A known zero sample dimension proves a
        zero-width output; otherwise any Dynamic sample dimension makes the
        flattened width Dynamic. A collapsed multi-axis name becomes unnamed.

    Runtime reshaping allocates or views storage according to the native backend.
    Empty sample dimensions and empty batches are supported. Incompatible values
    fail through normal selected Method validation or native reshape errors.
    """
    @property
    def iteration_independent(self) -> bool:
        """Return whether discarded Flatten calls can be omitted safely."""

        return True

    def infer_output_spec(self, input_spec: SpecTree) -> SpecTree:
        """Infer flattened widths and axis names from a TensorSpec tree.

        Args:
            input_spec: Input spec tree. Unknown sample rank stays unknown.

        Returns:
            A new tree preserving dtype, batch and backend. Single-axis names
            survive; collapsed or newly introduced axes have no name.

        Side Effects:
            None. No values, sources or native frameworks are evaluated.

        Raises:
            TypeError: For ragged input, whose sample-flattening contract is not defined.
        """
        def infer(spec):
            if spec.layout is Layout.RAGGED:
                raise TypeError("Flatten does not yet define a ragged sample-flattening contract.")
            names = spec.axis_names
            if names is not None and spec.rank != 1:
                names = (None,)
            return replace(spec, shape=_flat_shape(spec.shape), axis_names=names)

        return map_spec_tree(input_spec, infer)

    @traits(backend="numpy", batch_mode="element")
    def numpy_call(self, x):
        return x.reshape(-1)

    @traits(backend="numpy", batch_mode="batched")
    def numpy_batched(self, x):
        """Reshape a NumPy batch to (batch, sample product), including zero sizes.

        Args:
            x: NumPy array with a leading batch axis.

        Returns:
            A NumPy array preserving batch length and dtype, normally as a view.
            Invalid native shapes raise ValueError without changing x.
        """
        # A concrete sample product avoids ambiguous reshape(0, -1) for empty batches.
        return x.reshape((x.shape[0], prod(x.shape[1:])))

    @traits(backend="tf", batch_mode="element")
    def tf_call(self, x):
        import tensorflow as tf

        return tf.reshape(x, [-1])

    @traits(backend="tf", batch_mode="batched")
    def tf_batched(self, x):
        """Flatten TensorFlow sample axes using concrete or graph shape arithmetic.

        Args:
            x: TensorFlow dense tensor with a leading batch axis.

        Returns:
            A same-dtype tensor of shape (batch, sample product), including zero
            sizes. TensorFlow reports invalid reshapes through native errors.
        """
        import tensorflow as tf

        shape = tf.shape(x)
        return tf.reshape(x, [shape[0], tf.reduce_prod(shape[1:])])

    @traits(backend="torch", batch_mode="element")
    def torch_call(self, x):
        return x.reshape(-1)

    @traits(backend="torch", batch_mode="batched")
    def torch_batched(self, x):
        """Flatten Torch sample axes without inferring an ambiguous zero product.

        Args:
            x: Dense Torch tensor with a leading batch axis.

        Returns:
            A same-dtype tensor shaped (batch, sample product), allocating or
            viewing according to Torch reshape semantics. Native errors propagate.
        """
        return x.reshape((x.shape[0], prod(x.shape[1:])))


class Scale(Method):
    """Apply scalar centering/scaling to a dense or ragged native tensor.

    Args:
        mean: Scalar center subtracted after dtype conversion.
        std: Nonzero scalar divisor.
        dtype: Semantic output dtype, default float32.

    Returns:
        A native value with unchanged shape/layout and preserved row partitions.
        NumPy ragged output owns new buffers. Inference performs only dtype changes.

    Raises:
        ValueError: If std is zero or native arithmetic/broadcasting is invalid.
        TypeError: For unsupported dtypes or values. No input buffers are mutated.
    """
    @property
    def iteration_independent(self) -> bool:
        """Return whether discarded Scale calls can be omitted safely."""

        return True

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
        """Return dtype-converted, centered and scaled NumPy values.

        Args:
            x: Dense ndarray or packed RaggedTensor.

        Returns:
            A new value of the same layout; ragged row partitions are preserved.
            Unsupported casts or arithmetic propagate NumPy TypeError/ValueError.
        """
        from dryml.numpy.ragged import RaggedTensor
        if isinstance(x, RaggedTensor):
            values = (x.flat_values.astype(self.dtype.np()) - self.mean) / self.std
            return x.with_flat_values(values)
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
