"""Immutable normalized Method call signatures and specification validation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import sys
from typing import Literal

import numpy as np

from dryml.core.backend import Backend, _native_backend, backend_testers
from dryml.core.tensor_spec import (
    BatchMode, Dynamic, SpecTree, TensorSpec, _python_object_batch_size,
    _python_object_spec, is_spec_tree, iter_specs,
)

from .errors import MethodError

MethodCallMode = Literal["eager", "learning", "cached"]
MethodCallNodeKind = Literal["tensor", "tuple", "list", "mapping"]


@dataclass(frozen=True, slots=True, eq=False)
class MethodCallNode:
    """One recursively immutable, normalized Method argument value.

    Args:
        kind: The preserved tensor, tuple, list, or mapping structure kind.
        value: An immutable tensor specification, child-node tuple, or ordered
            ``(str | int, node)`` mapping entries.

    Nodes copy mutable source containers during normalization. Tensor leaves use
    :meth:`TensorSpec.equal_exact` so exact signature comparisons include the
    backend without changing ordinary TensorSpec equality.
    """

    kind: MethodCallNodeKind
    value: TensorSpec | tuple["MethodCallNode", ...] | tuple[tuple[str | int, "MethodCallNode"], ...]

    def __eq__(self, other: object) -> bool:
        """Return whether another node has the same exact recursive signature."""

        return isinstance(other, MethodCallNode) and _nodes_equal(self, other)

    def __hash__(self) -> int:
        """Return a stable hash consistent with exact recursive equality."""

        return hash((self.kind, self.value))


@dataclass(frozen=True, slots=True, eq=False)
class MethodCallSignature:
    """The exact immutable normalized form of one logical Method call.

    Args:
        args: Positional argument nodes in original order.
        kwargs: Keyword argument nodes sorted by name while retaining their
            positional-versus-keyword layout.
        backend: The one observed backend, or ``None`` when unavailable.
        batch_mode: The observed or learning-time effective batch mode, or
            ``None`` when unobservable.

    Signatures are diagnostics and cache keys only. Constructing one does not
    select implementations, invoke user code, or mutate Method state.
    """

    args: tuple[MethodCallNode, ...]
    kwargs: tuple[tuple[str, MethodCallNode], ...]
    backend: Backend | None
    batch_mode: BatchMode | None

    def __eq__(self, other: object) -> bool:
        """Return whether another signature is an exact normalized call match."""

        return (
            isinstance(other, MethodCallSignature)
            and self.args == other.args
            and self.kwargs == other.kwargs
            and self.backend is other.backend
            and self.batch_mode == other.batch_mode
        )

    def __hash__(self) -> int:
        """Return a hash consistent with exact normalized comparison."""

        return hash((self.args, self.kwargs, self.backend, self.batch_mode))


def spec_node(spec: SpecTree) -> MethodCallNode:
    """Copy one validated specification tree into an immutable node.

    Raises:
        TypeError: If ``spec`` is not a supported normalized specification tree.
    """

    if not is_spec_tree(spec):
        raise TypeError("Method input_spec must be a normalized SpecTree.")
    return _node_from_spec(spec)


def spec_from_node(node: MethodCallNode) -> SpecTree:
    """Return a fresh public specification tree from immutable node facts.

    Args:
        node: A normalized immutable Method specification node.

    Returns:
        A newly structured ``SpecTree`` whose TensorSpec leaves remain immutable.

    This reverses :func:`spec_node` for diagnostic and graph-view APIs without
    selecting an implementation, invoking a target, or importing a backend.
    """

    if node.kind == "tensor":
        return node.value  # type: ignore[return-value]
    if node.kind == "tuple":
        return tuple(spec_from_node(child) for child in node.value)  # type: ignore[union-attr]
    if node.kind == "list":
        return [spec_from_node(child) for child in node.value]  # type: ignore[union-attr]
    return {
        key: spec_from_node(child)
        for key, child in node.value  # type: ignore[union-attr]
    }


def spec_nodes(
    input_spec: SpecTree | None,
    additional_input_specs: tuple[SpecTree, ...],
) -> tuple[MethodCallNode, ...]:
    """Normalize retained positional selected-call specifications immutably.

    Args:
        input_spec: Optional first logical input specification.
        additional_input_specs: Later positional specifications, which require a
            first specification.

    Returns:
        Immutable normalized nodes in positional order.

    Raises:
        TypeError: If a specification is malformed or later specifications are
        supplied without a first specification.
    """

    if input_spec is None:
        if additional_input_specs:
            raise TypeError("Additional Method specs require a first input spec.")
        return ()
    return (spec_node(input_spec), *map(spec_node, additional_input_specs))


def runtime_node(value: object) -> MethodCallNode:
    """Normalize one runtime value without retaining opaque content or guessing batching.

    Raises:
        TypeError: For unsupported native tensor adapters/dtypes or container
            keys. Unrecognized Python leaves use the same object/Python-layout
            vocabulary as retained inputs and outputs. This function never
            imports an optional framework to recognize a value.
    """

    if isinstance(value, TensorSpec):
        return MethodCallNode("tensor", value)
    if isinstance(value, tuple):
        return MethodCallNode("tuple", tuple(runtime_node(item) for item in value))
    if isinstance(value, list):
        return MethodCallNode("list", tuple(runtime_node(item) for item in value))
    if isinstance(value, Mapping):
        entries: list[tuple[str | int, MethodCallNode]] = []
        for key, item in value.items():
            if not isinstance(key, (str, int)):
                raise TypeError("Method mapping keys must be str or int.")
            entries.append((key, runtime_node(item)))
        return MethodCallNode("mapping", tuple(entries))
    if isinstance(value, (np.ndarray, np.generic)):
        array = np.asarray(value)
        return MethodCallNode(
            "tensor",
            TensorSpec(dtype=array.dtype, shape=array.shape, backend=Backend.numpy),
        )
    if np.isscalar(value):
        array = np.asarray(value)
        return MethodCallNode(
            "tensor",
            TensorSpec(dtype=array.dtype, shape=array.shape),
        )
    # Optional adapters are usable only after their framework package has
    # explicitly registered itself. Looking in sys.modules avoids importing a
    # framework merely to inspect an ordinary Method call.
    for backend, tester in tuple(backend_testers.items()):
        if backend is Backend.numpy:
            continue
        try:
            recognized = tester(value)
        except Exception as error:
            raise MethodError(f"The {backend.value!r} backend detector failed.") from error
        if not recognized:
            continue
        module = sys.modules.get(f"dryml.{backend.value}")
        converter = None if module is None else vars(module).get("as_tensor_spec")
        if callable(converter):
            return spec_node(converter(value))
        raise TypeError("Method call signatures require a registered native tensor spec adapter.")
    if _native_backend(value) is not None:
        raise TypeError("Method call signatures require a registered native tensor spec adapter.")
    return MethodCallNode("tensor", _python_object_spec())


def runtime_node_for_constraint(value: object, constraint: MethodCallNode) -> MethodCallNode:
    """Normalize a selected-call runtime value using its retained physical batch contract.

    A retained batched TensorSpec makes the first physical runtime axis observable
    without guessing from a dense value during ordinary eager selection.
    Python object leaves use only the declared outer batch boundary, retaining no
    content, class, shape attributes or array-conversion evidence. Trees are
    traversed by their spec so object-valued containers remain opaque leaves.
    Native values inherit logical sample-axis labels from the retained declaration;
    explicit TensorSpec values retain their own labels and must satisfy them.

    Args:
        value: Runtime native tensor, opaque value, or declared structural tree.
        constraint: Immutable MethodCallNode carrying the retained logical spec.

    Returns:
        An immutable node with observed tensor metadata and declared logical axis
        interpretation, without retaining opaque values or their classes.

    Raises:
        TypeError: If ``value`` cannot be normalized by the supported adapters.

    Side Effects:
        Reads native metadata through registered adapters; executes no Method.
    """

    return _runtime_node_with_schema(value, constraint, retain_batch=True)


def _runtime_node_with_schema(value, constraint, *, retain_batch):
    """Use opaque leaf boundaries without changing exact physical tensor signatures."""
    if constraint.kind == "tensor":
        expected = constraint.value
        if expected.is_python_object:
            size = _python_object_batch_size(value) if expected.batched else None
            observed = replace(expected, shape=(), batch=size)
            return MethodCallNode("tensor", observed)
        observed = runtime_node(value)
        if not retain_batch:
            return observed
        observed = _apply_retained_batch(observed, constraint)
        if (not isinstance(value, TensorSpec) and observed.kind == "tensor"
                and expected.axis_names is not None and observed.value.rank == expected.rank):
            observed = MethodCallNode("tensor", replace(observed.value, axis_names=expected.axis_names))
        return observed
    if constraint.kind == "mapping" and isinstance(value, Mapping):
        expected = dict(constraint.value)
        if not all(isinstance(key, (str, int)) for key in value):
            raise TypeError("Method mapping keys must be str or int.")
        return MethodCallNode("mapping", tuple(
            (key, _runtime_node_with_schema(item, expected[key], retain_batch=retain_batch)
             if key in expected else runtime_node(item))
            for key, item in value.items()))
    if constraint.kind in {"tuple", "list"} and isinstance(value, (tuple, list)):
        children = constraint.value
        return MethodCallNode("tuple" if isinstance(value, tuple) else "list", tuple(
            _runtime_node_with_schema(item, children[index], retain_batch=retain_batch)
            if index < len(children) else runtime_node(item)
            for index, item in enumerate(value)))
    return runtime_node(value)


def _python_object_inputs(nodes) -> bool:
    """Report a nonempty first-input tree whose backend is inapplicable, not unknown."""
    leaves = tuple(_tensor_leaves(nodes[0])) if nodes else ()
    return bool(leaves) and all(spec.is_python_object for spec in leaves)


def _runtime_backends_for_spec(value, spec):
    """Discover tensor backends without traversing explicitly opaque Python leaves."""
    from dryml.core.backend import discover_backends
    if not any(leaf.is_python_object for leaf in iter_specs(spec)):
        return discover_backends(value, activate_native=True)

    def tensor_values(value, spec):
        if isinstance(spec, TensorSpec):
            if not spec.is_python_object:
                yield value
        elif isinstance(spec, dict) and isinstance(value, Mapping):
            for key, child in spec.items():
                if key in value:
                    yield from tensor_values(value[key], child)
        elif isinstance(spec, (tuple, list)) and isinstance(value, (tuple, list)):
            for item, child in zip(value, spec):
                yield from tensor_values(item, child)
        else:
            raise TypeError("Runtime input structure differs from its declared spec.")

    tensors = tuple(tensor_values(value, spec))
    return discover_backends(*tensors, activate_native=True) if tensors else set()


def complete_backend_constraint(
    constraint: MethodCallNode | None,
    backend: Backend | None,
) -> MethodCallNode | None:
    """Complete unknown backend leaves in a retained selection constraint.

    Args:
        constraint: The immutable first-argument constraint retained by a selected
            implementation, if the caller supplied one.
        backend: The explicitly selected backend, if known.

    Returns:
        A recursively copied constraint whose backend-unknown tensor leaves use
        ``backend``. Known leaf backends remain unchanged.

    Selection completes producer-input facts before conversion planning, and
    selected-call validation retains the completed constraints. This operation
    neither discovers a backend nor imports optional framework adapters.
    """

    if constraint is None or backend is None:
        return constraint
    if constraint.kind == "tensor":
        spec = constraint.value  # type: ignore[assignment]
        if spec.backend is None and not spec.is_python_object:
            spec = replace(spec, backend=backend)
        return MethodCallNode("tensor", spec)
    if constraint.kind == "mapping":
        return MethodCallNode(
            "mapping",
            tuple(
                (key, complete_backend_constraint(value, backend))
                for key, value in constraint.value  # type: ignore[union-attr]
            ),
        )
    return MethodCallNode(
        constraint.kind,
        tuple(
            complete_backend_constraint(value, backend)
            for value in constraint.value  # type: ignore[union-attr]
        ),
    )


def spec_from_runtime_node(
    node: MethodCallNode,
    batch_mode: BatchMode | None,
) -> SpecTree:
    """Convert one runtime node to a logical first-input specification.

    Args:
        node: Immutable runtime argument structure.
        batch_mode: Effective batch intent selected for the learning call.

    Returns:
        A fresh specification tree. Batched intent moves each physical leading
        tensor axis into its logical batch field; unknown intent preserves the
        physical shape without asserting element selection.

    Raises:
        TypeError: If batched intent is incompatible with a scalar or unknown
        physical tensor shape.
    """

    if (batch_mode is BatchMode.batched and node.kind in {"list", "tuple"}
            and node.value and all(
                child.kind == "tensor" and child.value.is_python_object for child in node.value)):
        return _python_object_spec(batch=len(node.value))
    if node.kind == "tensor":
        spec = node.value  # type: ignore[assignment]
        if batch_mode is BatchMode.batched and spec.batch is None:
            if spec.shape is None or not spec.shape:
                raise TypeError("Batched Method calls require a physical leading axis.")
            spec = replace(
                spec,
                shape=spec.shape[1:],
                batch=spec.shape[0],
                batch_axis_name="batch",
            )
        return spec
    if node.kind == "tuple":
        return tuple(spec_from_runtime_node(child, batch_mode) for child in node.value)  # type: ignore[union-attr]
    if node.kind == "list":
        return [spec_from_runtime_node(child, batch_mode) for child in node.value]  # type: ignore[union-attr]
    return {
        key: spec_from_runtime_node(child, batch_mode)
        for key, child in node.value  # type: ignore[union-attr]
    }


def call_signature(
    args: tuple[object, ...], kwargs: Mapping[str, object], *,
    expected: MethodCallSignature | None = None,
) -> MethodCallSignature:
    """Normalize a complete logical call into an exact immutable signature.

    Args:
        args: Positional values in call order.
        kwargs: Named values, normalized in sorted keyword order.
        expected: Optional retained signature defining opaque leaf/batch
            boundaries. Numerical fields still use exact physical signatures.

    Returns:
        Immutable call metadata without retaining opaque values or their classes.

    Raises:
        TypeError: If a positional/keyword value or keyword name cannot be
            represented by the supported tensor/spec adapter vocabulary.
        ValueError: For conflicting observed backend or batch facts.

    An existing signature supplies opaque leaf/batch boundaries, so their content
    is never reclassified as a tensor or structural container during cached calls.
    Physical numerical signatures and positional/keyword layout remain exact.
    Registered lightweight detectors/adapters may run, but no target is invoked
    and no optional framework is imported merely to classify an opaque value.
    """

    if not all(isinstance(name, str) for name in kwargs):
        raise TypeError("Method keyword names must be strings.")
    constraints = () if expected is None else expected.args
    named = {} if expected is None else dict(expected.kwargs)
    arg_nodes = tuple(_runtime_node_with_schema(value, constraints[index], retain_batch=False)
                      if index < len(constraints) else runtime_node(value) for index, value in enumerate(args))
    kw_nodes = tuple((name, _runtime_node_with_schema(kwargs[name], named[name], retain_batch=False)
                     if name in named else runtime_node(kwargs[name])) for name in sorted(kwargs))
    backend, _ = node_facts((*arg_nodes, *(node for _, node in kw_nodes)))
    if expected is None:
        _, batch_mode = runtime_facts(args, kwargs)
    else:
        facts = [fact for index, value in enumerate(args)
                 for fact in _runtime_fact_leaves(value, constraints[index] if index < len(constraints) else None)]
        facts.extend(fact for name, value in kwargs.items() for fact in _runtime_fact_leaves(value, named.get(name)))
        _, batch_mode = _facts_from_leaves(facts)
    return MethodCallSignature(arg_nodes, kw_nodes, backend, batch_mode)


def runtime_facts(args: tuple[object, ...], kwargs: Mapping[str, object]) -> tuple[Backend | None, BatchMode | None]:
    """Return only unambiguous registered runtime backend and batch facts.

    Opaque leaves contribute no tensor backend or unobserved batch facts so
    ordinary eager calls can reach genuinely generic implementations. Conflicting observed
    backend or batch facts raise ``ValueError`` before target invocation.
    """

    leaves: list[tuple[TensorSpec, BatchMode | None]] = []
    for value in (*args, *kwargs.values()):
        try:
            leaves.extend(_runtime_fact_leaves(value))
        except TypeError:
            continue
    return _facts_from_leaves(leaves)


def _facts_from_leaves(leaves):
    """Reduce already-admitted leaf facts without observing source values again."""
    backends = {spec.backend for spec, _ in leaves if spec.backend is not None}
    if len(backends) > 1:
        raise ValueError("Method call contains conflicting backend facts.")
    batch_modes = {batch_mode for _, batch_mode in leaves if batch_mode is not None}
    if len(batch_modes) > 1:
        raise ValueError("Method call contains conflicting batch facts.")
    return next(iter(backends), None), next(iter(batch_modes), None)


def node_facts(nodes: object) -> tuple[Backend | None, BatchMode | None]:
    """Return the unambiguous backend and declared batching facts in nodes."""

    leaves = tuple(_tensor_leaves(nodes))
    backends = {leaf.backend for leaf in leaves if leaf.backend is not None}
    if len(backends) > 1:
        raise ValueError("Method call contains conflicting backend facts.")
    batch_modes = {
        BatchMode.batched if leaf.batched else BatchMode.element
        for leaf in leaves
        if isinstance(leaf, TensorSpec) and (leaf.batch is not None or isinstance(nodes, MethodCallNode))
    }
    # Runtime ndarray leaves do not expose batching intent. A caller-provided
    # TensorSpec does, but its element form is also explicit.
    if len(batch_modes) > 1:
        raise ValueError("Method call contains conflicting batch facts.")
    return next(iter(backends), None), next(iter(batch_modes), None)


def satisfies(constraint: MethodCallNode, observed: MethodCallNode) -> bool:
    """Return whether observed runtime facts satisfy one directional input constraint."""

    if constraint.kind != observed.kind:
        return False
    if constraint.kind == "tensor":
        return _spec_satisfies(constraint.value, observed.value)  # type: ignore[arg-type]
    if constraint.kind == "mapping":
        expected_entries = constraint.value  # type: ignore[assignment]
        actual_entries = observed.value  # type: ignore[assignment]
        return len(expected_entries) == len(actual_entries) and all(
            expected_key == actual_key and satisfies(expected, actual)
            for (expected_key, expected), (actual_key, actual) in zip(expected_entries, actual_entries)
        )
    expected_children = constraint.value  # type: ignore[assignment]
    actual_children = observed.value  # type: ignore[assignment]
    return len(expected_children) == len(actual_children) and all(
        satisfies(expected, actual) for expected, actual in zip(expected_children, actual_children)
    )


def _node_from_spec(spec: SpecTree) -> MethodCallNode:
    if isinstance(spec, TensorSpec):
        return MethodCallNode("tensor", spec)
    if isinstance(spec, tuple):
        return MethodCallNode("tuple", tuple(_node_from_spec(item) for item in spec))
    if isinstance(spec, list):
        return MethodCallNode("list", tuple(_node_from_spec(item) for item in spec))
    return MethodCallNode("mapping", tuple((key, _node_from_spec(item)) for key, item in spec.items()))


def _nodes_equal(left: MethodCallNode, right: MethodCallNode) -> bool:
    if left.kind != right.kind:
        return False
    if left.kind == "tensor":
        return left.value.equal_exact(right.value)  # type: ignore[union-attr]
    return left.value == right.value


def _tensor_leaves(nodes: object):
    if isinstance(nodes, MethodCallNode):
        if nodes.kind == "tensor":
            yield nodes.value
        elif nodes.kind == "mapping":
            for _, child in nodes.value:  # type: ignore[union-attr]
                yield from _tensor_leaves(child)
        else:
            for child in nodes.value:  # type: ignore[union-attr]
                yield from _tensor_leaves(child)
    else:
        for node in nodes:  # type: ignore[union-attr]
            yield from _tensor_leaves(node)


def _runtime_fact_leaves(value: object, constraint=None) -> tuple[tuple[TensorSpec, BatchMode | None], ...]:
    """Return runtime tensor leaves and only explicitly observable batch facts."""

    if constraint is not None and constraint.kind == "tensor" and constraint.value.is_python_object:
        return ((_runtime_node_with_schema(value, constraint, retain_batch=False).value, None),)
    if isinstance(value, TensorSpec):
        batch_mode = BatchMode.batched if value.batched else BatchMode.element
        return ((value, batch_mode),)
    if isinstance(value, (tuple, list)):
        children = constraint.value if constraint is not None and constraint.kind in {"tuple", "list"} else ()
        return tuple(leaf for index, item in enumerate(value)
                     for leaf in _runtime_fact_leaves(item, children[index] if index < len(children) else None))
    if isinstance(value, Mapping):
        named = dict(constraint.value) if constraint is not None and constraint.kind == "mapping" else {}
        return tuple(leaf for key, item in value.items() for leaf in _runtime_fact_leaves(item, named.get(key)))
    node = runtime_node(value)
    return tuple((spec, None) for spec in _tensor_leaves(node))


def _spec_satisfies(expected: TensorSpec, actual: TensorSpec) -> bool:
    if expected.dtype != actual.dtype or expected.layout != actual.layout:
        return False
    if expected.backend is not None and expected.backend is not actual.backend:
        return False
    if expected.shape is not None:
        if actual.shape is None or len(expected.shape) != len(actual.shape):
            return False
        if any(wanted is not Dynamic and wanted != got for wanted, got in zip(expected.shape, actual.shape)):
            return False
    if expected.batch is None:
        if actual.batch is not None:
            return False
    elif actual.batch is None or (expected.batch is not Dynamic and expected.batch != actual.batch):
        return False
    for field in ("axis_names", "batch_axis_name", "ragged_rank", "row_splits_dtype", "sparse_format"):
        wanted = getattr(expected, field)
        if wanted is not None and wanted != getattr(actual, field):
            return False
    return True


def _apply_retained_batch(observed: MethodCallNode, constraint: MethodCallNode) -> MethodCallNode:
    if observed.kind != constraint.kind:
        return observed
    if observed.kind == "tensor":
        actual = observed.value  # type: ignore[assignment]
        expected = constraint.value  # type: ignore[assignment]
        if expected.batch is not None and actual.batch is None and actual.shape is not None and actual.shape:
            actual = replace(
                actual,
                shape=actual.shape[1:],
                batch=actual.shape[0],
                batch_axis_name=expected.batch_axis_name,
            )
        return MethodCallNode("tensor", actual)
    if observed.kind == "mapping":
        expected_entries = constraint.value  # type: ignore[assignment]
        expected_by_key = dict(expected_entries)
        return MethodCallNode(
            "mapping",
            tuple((key, _apply_retained_batch(value, expected_by_key[key]) if key in expected_by_key else value)
                  for key, value in observed.value),  # type: ignore[union-attr]
        )
    expected_children = constraint.value  # type: ignore[assignment]
    return MethodCallNode(
        observed.kind,
        tuple(
            _apply_retained_batch(value, expected_children[index]) if index < len(expected_children) else value
            for index, value in enumerate(observed.value)  # type: ignore[union-attr]
        ),
    )


__all__ = ["MethodCallMode", "MethodCallNode", "MethodCallNodeKind", "MethodCallSignature"]
