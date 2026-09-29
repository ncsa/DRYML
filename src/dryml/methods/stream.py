"""Explicit iterator-port declarations and synchronous local stream execution.

This module keeps stream consumption separate from element-call arity.  Stream
nodes borrow their inputs; a :class:`StreamGraphCursor` owns every resource it
opens for one graph traversal.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, replace
from typing import Any

from dryml.core.cardinality import Cardinality
from dryml.core.tensor_spec import SpecTree

from .errors import ImplementationSelectionError


@dataclass(frozen=True, slots=True)
class IteratorPort:
    """One ordered iterator port carrying a specification for each yielded value.

    Args:
        spec: Immutable element specification, or ``None`` when it is not known.
    """

    spec: SpecTree | None


@dataclass(frozen=True, slots=True)
class StreamNode:
    """A bounded deterministic authoring declaration for one stream operation.

    Args:
        name: Stable descriptive name used in diagnostics and inspection.
        inputs: Ordered borrowed input iterator ports.
        output: The iterator port produced by this node.
        element_spec_transform: Pure function mapping input element specs to the
            output element spec.
        cardinality_transform: Pure function mapping input cardinalities to the
            declared output cardinality.
        pull_policy: Ordered, descriptive synchronous pull policy declarations.
        max_buffered_items: Exact nonnegative upper bound for node-owned buffered
            input/output values.
        implementation: Callable receiving the declared input iterators and
            returning an output iterator.

    The declaration does not execute ``implementation`` during planning.  Custom
    code is trusted to honor its declared policy and bound; this class validates
    the declarative boundary but is not an asynchronous scheduler or generator
    serializer.
    """

    name: str
    inputs: tuple[IteratorPort, ...]
    output: IteratorPort
    element_spec_transform: Callable[[tuple[SpecTree | None, ...]], SpecTree | None]
    cardinality_transform: Callable[[tuple[Cardinality, ...]], Cardinality]
    pull_policy: tuple[str, ...]
    max_buffered_items: int
    implementation: Callable[..., Iterator[Any]]

    def __post_init__(self) -> None:
        """Validate bounded synchronous declaration facts before execution."""

        if not self.name:
            raise ValueError("StreamNode name must be nonempty.")
        if not all(isinstance(port, IteratorPort) for port in self.inputs):
            raise TypeError("StreamNode inputs must be IteratorPort declarations.")
        if not isinstance(self.output, IteratorPort):
            raise TypeError("StreamNode output must be an IteratorPort declaration.")
        if not self.inputs:
            raise ValueError("StreamNode requires at least one input port.")
        if len(self.pull_policy) != len(self.inputs) or not all(isinstance(item, str) and item for item in self.pull_policy):
            raise ValueError("StreamNode pull_policy must declare one nonempty policy per input.")
        if type(self.max_buffered_items) is not int:
            raise TypeError("StreamNode max_buffered_items must be a nonnegative exact int.")
        if self.max_buffered_items < 0:
            raise ValueError("StreamNode max_buffered_items must be non-negative.")
        if not callable(self.element_spec_transform) or not callable(self.cardinality_transform) or not callable(self.implementation):
            raise TypeError("StreamNode transforms and implementation must be callable.")

    def output_spec(self, input_specs: tuple[SpecTree | None, ...]) -> SpecTree | None:
        """Return the pure output element specification for ordered input specs."""

        if len(input_specs) != len(self.inputs):
            raise ValueError("StreamNode received the wrong number of input specs.")
        return self.element_spec_transform(input_specs)

    def output_cardinality(self, input_cardinalities: tuple[Cardinality, ...]) -> Cardinality:
        """Return the pure declared output cardinality for ordered input cardinalities."""

        if len(input_cardinalities) != len(self.inputs):
            raise ValueError("StreamNode received the wrong number of input cardinalities.")
        normalized = tuple(
            value if isinstance(value, Cardinality) else Cardinality.finite(int(value))
            for value in input_cardinalities
        )
        result = self.cardinality_transform(normalized)
        return result if isinstance(result, Cardinality) else Cardinality.finite(int(result))

    def open(self, *inputs: Iterator[Any]) -> Iterator[Any]:
        """Open this selected trusted implementation over borrowed input iterators.

        Raises:
            TypeError: If the implementation returns a non-iterator value.

        The graph cursor, rather than this declaration, owns and closes the
        returned iterator and all input resources.
        """

        if len(inputs) != len(self.inputs):
            raise ValueError("StreamNode received the wrong number of input iterators.")
        output = self.implementation(*inputs)
        if not hasattr(output, "__next__"):
            raise TypeError("StreamNode implementation must return an iterator.")
        return output


def _declared_only(*inputs: Iterator[Any]) -> Iterator[Any]:
    """Reject direct execution of inspection-only built-in stream declarations."""

    del inputs
    raise RuntimeError("Built-in stream declarations are opened by the graph cursor.")


def _declared_output_spec(output: SpecTree | None):
    """Return a pure transform exposing one operator's already-declared output."""

    return lambda input_specs: output


def _declared_cardinality(output: Cardinality):
    """Return a pure transform exposing one operator's declared cardinality."""

    return lambda input_cardinalities: output


@dataclass(frozen=True, slots=True)
class _StreamOp:
    """One immutable prepared occurrence in a Dataset-owned stream graph."""

    kind: str
    dataset: object | None = None
    inputs: tuple["_StreamOp", ...] = ()
    selected: object | None = None
    node: StreamNode | None = None
    tree: object | None = None


class StreamPlan:
    """Prepare and open the qualified synchronous Dataset stream subset.

    Args:
        dataset: Root Dataset-like object whose qualified operators expose
            ``_stream_operator`` and normal Dataset source capabilities.

    Planning records immutable selected Map invokers but never opens a source.
    Each :meth:`iterator` creates isolated cursor/discovery state and therefore
    reopens every source occurrence independently.
    """

    def __init__(self, dataset: object) -> None:
        self.dataset = dataset
        self._root: _StreamOp | None = None
        self._nodes: tuple[object, ...] = ()
        self._conversion_edges: tuple[object, ...] = ()

    @property
    def nodes(self) -> tuple[object, ...]:
        """Return immutable inspection nodes generated by the most recent plan."""

        return self._nodes

    @property
    def conversion_edges(self) -> tuple[object, ...]:
        """Return immutable edges selected by Map occurrences in this stream plan."""

        return self._conversion_edges

    def learn(self, input_spec=None, additional_input_specs=(), *, strategy: str = "local", output_spec=None) -> None:
        """Prepare qualified operators without opening sources or invoking bodies.

        Args:
            input_spec: Optional first root-source spec assertion.
            additional_input_specs: Optional later root-source spec assertions.
            strategy: Only ``"local"`` is supported.
            output_spec: Optional root output spec assertion.

        Raises:
            ValueError: If assertions or strategy are incompatible.
            NotImplementedError: If the graph includes an unqualified operator.
            ImplementationSelectionError: If a known Map spec cannot select one
                local implementation.
        """

        if strategy != "local":
            raise ValueError(f"Unsupported Method preparation strategy {strategy!r}; only 'local' is available.")
        records: list[object] = []
        root = self._build(self.dataset, records)
        from .signature import spec_from_node

        root_specs = tuple(
            spec_from_node(node.outputs[0].spec)
            for node in records
            if node.kind == "source" and node.outputs and node.outputs[0].spec is not None
        )
        supplied = (() if input_spec is None else (input_spec, *additional_input_specs))
        if supplied and supplied != root_specs:
            raise ValueError("Dataset graph input specs do not match its declared source specs.")
        if output_spec is not None and output_spec != getattr(self.dataset, "spec"):
            raise ValueError("Dataset graph output spec does not match its declared output spec.")
        self._root = root
        self._nodes = tuple(records)
        edges = []
        for node in self._nodes:
            selected = getattr(node, "selected", None)
            edge = None if selected is None else selected.conversion_edge
            if edge is not None and edge not in edges:
                edges.append(edge)
        self._conversion_edges = tuple(edges)

    def eager(self) -> None:
        """Discard graph selections and reset every nested Method occurrence.

        Map graph inspection prepares the actual Method instances. Resetting only
        the plan leaves stale shared selection state for later backend graphs.
        """

        self._eager_dataset(self.dataset, set())
        self._root = None
        self._nodes = ()
        self._conversion_edges = ()

    @classmethod
    def _eager_dataset(cls, dataset: object, seen: set[int]) -> None:
        """Reset each Method reachable through one stream Dataset graph once."""

        identifier = id(dataset)
        if identifier in seen:
            return
        seen.add(identifier)
        method = getattr(dataset, "method", None)
        if method is not None:
            cls._eager_method(method, seen)
        source = getattr(dataset, "src", None)
        if source is not None:
            cls._eager_dataset(source, seen)
        for child in _iter_dataset_leaves(getattr(dataset, "sources", ())):
            cls._eager_dataset(child, seen)

    @classmethod
    def _eager_method(cls, method: object, seen: set[int]) -> None:
        """Reset one Method and nested Method fields retained by composition."""

        identifier = id(method)
        if identifier in seen:
            return
        seen.add(identifier)
        eager = getattr(method, "eager", None)
        if callable(eager):
            eager()
        for value in vars(method).values():
            cls._eager_value(value, seen)

    @classmethod
    def _eager_value(cls, value: object, seen: set[int]) -> None:
        """Traverse declared Method/container fields without probing opaque values."""

        if hasattr(value, "eager") and hasattr(value, "method_graph"):
            cls._eager_method(value, seen)
        elif isinstance(value, Mapping):
            for item in value.values():
                cls._eager_value(item, seen)
        elif isinstance(value, (tuple, list)):
            for item in value:
                cls._eager_value(item, seen)

    def iterator(self) -> "StreamGraphCursor":
        """Return one isolated pull cursor over this prepared graph.

        Raises:
            RuntimeError: If :meth:`learn` has not prepared this graph.
        """

        if self._root is None:
            raise RuntimeError("Dataset graph iteration requires graph.learn() first.")
        return StreamGraphCursor(self._root)

    def _build(self, dataset: object, records: list[object]) -> _StreamOp:
        from .ir import MethodGraphNode, MethodPort
        from .signature import spec_node

        # Only qualified concrete operators opt in.  Other Dataset subclasses
        # inherit the source marker but must not silently gain stream semantics.
        kind = type(dataset).__dict__.get("_stream_operator")
        if kind is None:
            kind = "unsupported" if hasattr(dataset, "src") else "source"
        if kind not in {"source", "map", "batch", "unbatch", "zip", "chain", "custom"}:
            raise NotImplementedError(f"Dataset graph planning does not support {type(dataset).__name__}.")
        spec = getattr(dataset, "spec")
        if kind == "source":
            records.append(MethodGraphNode(len(records), "source", None, outputs=(MethodPort("iterator", spec_node(spec)),)))
            return _StreamOp("source", dataset=dataset)
        if kind == "map":
            child = self._build(dataset.src, records)
            selected = None
            try:
                # A Map is an iterator boundary around an element Method graph.  Keep
                # the composed Method occurrences visible through the canonical
                # MethodGraph rather than exposing composition-private fields.
                element_graph = dataset.method.method_graph()
                element_graph.learn(dataset.src.spec, strategy="local")
                element_nodes = element_graph.method_nodes
                records.extend(replace(node, occurrence=len(records)) for node in element_nodes)
                selected = element_nodes[0].selected if element_nodes else None
            except ImplementationSelectionError as error:
                if error.reason != "unknown_traits" or error.unknown_traits != ("backend",):
                    raise
            records.append(MethodGraphNode(
                len(records), "method", type(dataset.method),
                inputs=(MethodPort("iterator", spec_node(dataset.src.spec)),),
                outputs=(MethodPort("iterator", spec_node(spec)),),
                selected=selected,
                traits=None if selected is None else selected.traits,
                stream_node=self._builtin_node("Map", (dataset.src.spec,), spec, dataset, ("one_to_one",), 1),
            ))
            return _StreamOp("map", dataset=dataset, inputs=(child,), selected=selected)
        if kind in {"batch", "unbatch"}:
            child = self._build(dataset.src, records)
            records.append(MethodGraphNode(
                len(records), "method", type(dataset),
                inputs=(MethodPort("iterator", spec_node(dataset.src.spec)),),
                outputs=(MethodPort("iterator", spec_node(spec)),),
                stream_node=self._builtin_node(
                    type(dataset).__name__,
                    (dataset.src.spec,),
                    spec,
                    dataset,
                    ("bounded_group" if kind == "batch" else "one_to_many",),
                    dataset.batch_size if kind == "batch" else 1,
                ),
            ))
            return _StreamOp(kind, dataset=dataset, inputs=(child,))
        if kind == "zip":
            leaves = tuple(_iter_dataset_leaves(dataset.sources))
            children = tuple(self._build(source, records) for source in leaves)
            records.append(MethodGraphNode(
                len(records), "method", type(dataset),
                inputs=tuple(MethodPort("iterator", spec_node(source.spec)) for source in leaves),
                outputs=(MethodPort("iterator", spec_node(spec)),),
                stream_node=self._builtin_node("Zip", tuple(source.spec for source in leaves), spec, dataset, ("position_order",) * len(leaves), 0),
            ))
            return _StreamOp("zip", dataset=dataset, inputs=children, tree=dataset.sources)
        if kind == "chain":
            children = tuple(self._build(source, records) for source in dataset.sources)
            records.append(MethodGraphNode(
                len(records), "method", type(dataset),
                inputs=tuple(MethodPort("iterator", spec_node(source.spec)) for source in dataset.sources),
                outputs=(MethodPort("iterator", spec_node(spec)),),
                stream_node=self._builtin_node("Chain", tuple(source.spec for source in dataset.sources), spec, dataset, ("declaration_order",) * len(dataset.sources), 0),
            ))
            return _StreamOp("chain", dataset=dataset, inputs=children)
        children = tuple(self._build(source, records) for source in dataset.sources)
        node = dataset.node
        records.append(MethodGraphNode(
            len(records), "method", type(node),
            inputs=tuple(MethodPort("iterator", spec_node(source.spec)) for source in dataset.sources),
            outputs=(MethodPort("iterator", spec_node(spec)),),
            stream_node=node,
        ))
        return _StreamOp("custom", dataset=dataset, inputs=children, node=node)

    @staticmethod
    def _builtin_node(
        name: str,
        input_specs: tuple[SpecTree, ...],
        output_spec: SpecTree,
        dataset: object,
        pull_policy: tuple[str, ...],
        max_buffered_items: int,
    ) -> StreamNode:
        """Expose built-in stream transforms and bounds without opening a source."""

        try:
            cardinality = dataset.__len__()
        except NotImplementedError:
            cardinality = Cardinality.UNKNOWN
        if not isinstance(cardinality, Cardinality):
            cardinality = Cardinality.finite(int(cardinality))
        return StreamNode(
            name=name,
            inputs=tuple(IteratorPort(spec) for spec in input_specs),
            output=IteratorPort(output_spec),
            element_spec_transform=_declared_output_spec(output_spec),
            cardinality_transform=_declared_cardinality(cardinality),
            pull_policy=pull_policy,
            max_buffered_items=max_buffered_items,
            implementation=_declared_only,
        )


class StreamGraphCursor(Iterator[Any]):
    """Own one local graph traversal and all resources acquired for it.

    The cursor acquires lazily, closes resources once in reverse acquisition order,
    and preserves a body/acquisition failure if cleanup also fails.  It does not
    serialize or retain a generator frame for replay.
    """

    def __init__(self, root: _StreamOp) -> None:
        self._root_op = root
        self._root: Iterator[Any] | None = None
        self._resources: list[object] = []
        self._resource_ids: set[int] = set()
        self._position = 0
        self._closed = False

    def __iter__(self) -> "StreamGraphCursor":
        """Return this cursor as its own iterator."""

        return self

    @property
    def position(self) -> int:
        """Return the number of graph output elements already yielded."""

        return self._position

    def __next__(self) -> Any:
        """Pull one graph output and close all acquired resources at termination."""

        if self._closed:
            raise StopIteration
        try:
            if self._root is None:
                self._root = self._open(self._root_op)
            value = next(self._root)
        except StopIteration:
            self.close()
            raise
        except BaseException as error:
            self._close_after_failure(error)
            raise
        self._position += 1
        return value

    def skip(self, n: int) -> None:
        """Consume exactly ``n`` graph outputs or report partial advancement.

        Args:
            n: Nonnegative exact integer count.

        Raises:
            TypeError: If ``n`` is not an exact integer.
            ValueError: If ``n`` is negative.
            DatasetExhaustedError: If the graph ends before ``n`` outputs.
        """

        from dryml.data.dataset import DatasetExhaustedError

        if type(n) is not int:
            raise TypeError("skip count must be a nonnegative exact int.")
        if n < 0:
            raise ValueError("skip count must be non-negative.")
        advanced = 0
        while advanced < n:
            try:
                next(self)
            except StopIteration as error:
                raise DatasetExhaustedError(n, advanced) from error
            advanced += 1

    def close(self) -> None:
        """Close every acquired resource once in reverse acquisition order.

        Raises:
            BaseException: The first reverse-acquisition cleanup failure. All
                resources are still given one close attempt; on runtimes that
                support exception notes, later failures are attached to the first.
        """

        if self._closed:
            return
        self._closed = True
        errors = self._close_resources()
        if errors:
            primary, *additional = errors
            self._annotate_cleanup(primary, additional)
            raise primary

    def _close_after_failure(self, primary: BaseException) -> None:
        """Close after a primary failure without allowing cleanup to mask it."""

        if self._closed:
            return
        self._closed = True
        errors = self._close_resources()
        self._annotate_cleanup(primary, errors)

    @staticmethod
    def _annotate_cleanup(primary: BaseException, errors: list[BaseException]) -> None:
        """Attach secondary cleanup diagnostics when the runtime supports notes."""

        add_note = getattr(primary, "add_note", None)
        if errors and add_note is not None:
            add_note(
                "Stream graph cleanup also failed: "
                + "; ".join(f"{type(error).__name__}: {error}" for error in errors)
            )

    def _close_resources(self) -> list[BaseException]:
        """Close acquired resources in reverse order and collect every failure."""

        errors: list[BaseException] = []
        for resource in reversed(self._resources):
            close = getattr(resource, "close", None)
            if close is None:
                continue
            try:
                close()
            except BaseException as error:
                errors.append(error)
        return errors

    def _own(self, resource: Iterator[Any]) -> Iterator[Any]:
        """Record one newly acquired owned iterator exactly once."""

        if id(resource) not in self._resource_ids:
            self._resource_ids.add(id(resource))
            self._resources.append(resource)
        return resource

    def _open(self, op: _StreamOp) -> Iterator[Any]:
        """Open one operation while retaining source/output ownership in this cursor."""

        if op.kind == "source":
            return self._own(op.dataset.iterator())
        if op.kind == "map":
            source = self._open(op.inputs[0])
            return self._own(_MapIterator(source, op.dataset, op.selected))
        if op.kind == "batch":
            source = self._open(op.inputs[0])
            return self._own(_BatchIterator(source, op.dataset))
        if op.kind == "unbatch":
            source = self._open(op.inputs[0])
            return self._own(_UnbatchIterator(source, op.dataset))
        if op.kind == "zip":
            return self._own(_ZipIterator(self, op.inputs, op.tree))
        if op.kind == "chain":
            return self._own(_ChainIterator(self, op.inputs))
        if op.kind == "custom":
            return self._own(_CustomIterator(self, op.inputs, op.node))
        raise AssertionError(f"unknown stream operation {op.kind!r}")


class _MapIterator(Iterator[Any]):
    """One Map output iterator with an occurrence-local discovery buffer."""

    def __init__(self, source: Iterator[Any], dataset: object, selected: object | None) -> None:
        self.source = source
        self.dataset = dataset
        self.selected = selected
        self.pending: object | None = None
        self.has_pending = False
        self.resolved = selected is not None

    def __iter__(self) -> "_MapIterator":
        return self

    def __next__(self) -> Any:
        if not self.resolved:
            self._resolve()
        if self.selected is None:
            raise StopIteration
        if self.has_pending:
            item = self.pending
            self.pending = None
            self.has_pending = False
        else:
            item = next(self.source)
        return self.selected(item)

    def _resolve(self) -> None:
        try:
            selected = self.dataset.method.find_implementation(input_spec=self.dataset.src.spec).prepared_invoker()
        except ImplementationSelectionError as error:
            if error.reason != "unknown_traits" or error.unknown_traits != ("backend",):
                raise
            item = next(self.source)
            from dryml.core.backend import discover_backends

            backends = discover_backends(item)
            if len(backends) != 1:
                raise ImplementationSelectionError("conflict" if backends else "unknown_traits", () if backends else ("backend",))
            selected = self.dataset.method.find_implementation(
                input_spec=self.dataset.src.spec,
                backend=next(iter(backends)),
            ).prepared_invoker()
            self.pending = item
            self.has_pending = True
        self.selected = selected
        self.resolved = True


class _BatchIterator(Iterator[Any]):
    """One bounded Batch output iterator retaining its collate selection once."""

    def __init__(self, source: Iterator[Any], dataset: object) -> None:
        self.source = source
        self.dataset = dataset
        self.collate = None

    def __iter__(self) -> "_BatchIterator":
        return self

    def __next__(self) -> Any:
        batch = []
        for _ in range(self.dataset.batch_size):
            try:
                batch.append(next(self.source))
            except StopIteration:
                break
        if not batch or (self.dataset.drop_remainder and len(batch) < self.dataset.batch_size):
            raise StopIteration
        if self.collate is None:
            self.collate = self.dataset._resolve_collate(batch)
        return self.collate(batch)


class _UnbatchIterator(Iterator[Any]):
    """One Unbatch output iterator retaining one selected split implementation."""

    def __init__(self, source: Iterator[Any], dataset: object) -> None:
        self.source = source
        self.split = dataset._resolve_split()
        self.pending: Iterator[Any] = iter(())

    def __iter__(self) -> "_UnbatchIterator":
        return self

    def __next__(self) -> Any:
        while True:
            try:
                return next(self.pending)
            except StopIteration:
                self.pending = iter(self.split(next(self.source)))


class _ZipIterator(Iterator[Any]):
    """Pull inputs in declared order and stop at the first exhausted position."""

    def __init__(self, cursor: StreamGraphCursor, inputs: tuple[_StreamOp, ...], tree: object) -> None:
        self.cursor = cursor
        self.inputs = inputs
        self.tree = tree
        self.iterators: tuple[Iterator[Any], ...] | None = None

    def __iter__(self) -> "_ZipIterator":
        return self

    def __next__(self) -> Any:
        if self.iterators is None:
            self.iterators = tuple(self.cursor._open(op) for op in self.inputs)
        values = [next(iterator) for iterator in self.iterators]
        value_iter = iter(values)
        return _build_zip_tree(self.tree, value_iter)


class _ChainIterator(Iterator[Any]):
    """Open sources only as their declared predecessors become exhausted."""

    def __init__(self, cursor: StreamGraphCursor, inputs: tuple[_StreamOp, ...]) -> None:
        self.cursor = cursor
        self.inputs = inputs
        self.index = 0
        self.current: Iterator[Any] | None = None

    def __iter__(self) -> "_ChainIterator":
        return self

    def __next__(self) -> Any:
        while self.index < len(self.inputs):
            if self.current is None:
                self.current = self.cursor._open(self.inputs[self.index])
            try:
                return next(self.current)
            except StopIteration:
                self.current = None
                self.index += 1
        raise StopIteration


class _CustomIterator(Iterator[Any]):
    """Open one custom output lazily after its ordered inputs are acquired."""

    def __init__(self, cursor: StreamGraphCursor, inputs: tuple[_StreamOp, ...], node: StreamNode) -> None:
        self.cursor = cursor
        self.inputs = inputs
        self.node = node
        self.output: Iterator[Any] | None = None

    def __iter__(self) -> "_CustomIterator":
        return self

    def __next__(self) -> Any:
        if self.output is None:
            inputs = tuple(self.cursor._open(op) for op in self.inputs)
            self.output = self.cursor._own(self.node.open(*inputs))
        return next(self.output)


def _iter_dataset_leaves(tree: object):
    """Yield Dataset leaves in the existing Zip positional traversal order."""

    if hasattr(tree, "_stream_operator"):
        yield tree
    elif isinstance(tree, dict):
        for value in tree.values():
            yield from _iter_dataset_leaves(value)
    elif isinstance(tree, (tuple, list)):
        for value in tree:
            yield from _iter_dataset_leaves(value)
    else:
        raise TypeError(f"Zip expects Dataset leaves, got {type(tree).__name__}.")


def _build_zip_tree(tree: object, values: Iterator[Any]) -> Any:
    """Rebuild one Zip output using the existing source tree's ordering."""

    if hasattr(tree, "_stream_operator"):
        return next(values)
    if isinstance(tree, dict):
        return {key: _build_zip_tree(value, values) for key, value in tree.items()}
    if isinstance(tree, tuple):
        return tuple(_build_zip_tree(value, values) for value in tree)
    if isinstance(tree, list):
        return [_build_zip_tree(value, values) for value in tree]
    raise TypeError(f"Zip expects Dataset leaves, got {type(tree).__name__}.")


__all__ = ["IteratorPort", "StreamGraphCursor", "StreamNode"]
