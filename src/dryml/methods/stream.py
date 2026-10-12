"""Explicit iterator-port declarations and synchronous local stream execution.

This module keeps stream consumption separate from element-call arity.  Stream
nodes borrow their inputs; a :class:`StreamGraphCursor` owns every resource it
opens for one graph traversal.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from typing import Any

from dryml.core.cardinality import Cardinality
from dryml.core.tensor_spec import SpecTree

from .errors import ImplementationSelectionError


class DatasetExhaustedError(ValueError):
    """Report a strict yield-count shortfall without changing exception identity.

    Args:
        requested: Nonnegative number of values required by the operation.
        yielded: Number actually consumed before exhaustion.

    Attributes:
        requested: Required yield count.
        yielded: Successfully consumed yield count.

    Data re-exports this lower-layer exception; it is not a Methods-root export.
    """

    def __init__(self, requested: int, yielded: int) -> None:
        self.requested = requested
        self.yielded = yielded
        super().__init__(
            f"Requested {requested} dataset elements, but the source yielded only {yielded}."
        )


class _CursorLifecycle:
    """Own resources and positive output-termination evidence for one traversal.

    Subclasses implement ``_read`` using designated ``_pull`` calls for iterator
    exhaustion and ``_call`` for acquisition, operators, and validation. Strict
    skip shares the failure boundary; consumers may specialize ``_skip`` without
    certifying EOF or closing resources. Persistence policy remains with consumers.
    """

    def __init__(self) -> None:
        self._resources: list[object] = []
        self._resource_ids: set[int] = set()
        self._deferred_cleanup: list[BaseException] = []
        self._closed = False
        self._cleanup_done = False
        self._cleanup_deferred = False
        self._observed_eof = False

    @property
    def observed_eof(self) -> bool:
        """Return positive root-output EOF evidence, independent of closure.

        True only after a designated output pull ends or a strict output bound is
        satisfied and subsequently observed. Cleanup failures do not erase this
        evidence. Explicit close, interruption, and failures never create it.
        """

        return self._observed_eof

    def __iter__(self):
        """Return this traversal's iterator without acquiring a resource."""

        return self

    def __next__(self):
        """Read one output, closing on EOF/failure while preserving primary errors."""

        if self._closed:
            raise StopIteration
        try:
            value = self._read()
        except StopIteration:
            self._observed_eof = True
            if self._cleanup_deferred:
                self._closed = True
            else:
                self.close()
            raise
        except BaseException as error:
            self._close_after_failure(error)
            raise
        return value

    @staticmethod
    def _validate_skip_count(n: int) -> None:
        """Reject counts that cannot express an exact nonnegative yield count."""

        if type(n) is not int:
            raise TypeError("skip count must be a nonnegative exact int.")
        if n < 0:
            raise ValueError("skip count must be non-negative.")

    def skip(self, n: int) -> None:
        """Consume exactly ``n`` outputs or report this call's partial advancement.

        Args:
            n: Nonnegative exact integer number of yields to discard.

        Returns:
            None after advancing exactly ``n`` outputs; zero is always a no-op.

        Raises:
            TypeError: If ``n`` is not an exact integer.
            ValueError: If ``n`` is negative.
            DatasetExhaustedError: If fewer than ``n`` outputs remain, including
                on a closed cursor. Counts describe requested and actual advances.
            BaseException: Acquisition, read, or operator failure.

        Invalid counts do not advance or close the cursor. Real failures close
        owned resources while keeping the primary error over cleanup failures.
        Shortfall never creates root EOF evidence or erases already observed EOF.
        """

        self._validate_skip_count(n)
        if n == 0:
            return
        if self._closed:
            raise DatasetExhaustedError(n, 0)
        try:
            self._call(self._skip, n)
        except BaseException as error:
            self._close_after_failure(error)
            raise

    def _skip(self, n: int) -> None:
        """Read a strict prefix before applying root termination/cleanup policy."""

        advanced = 0
        while advanced < n:
            try:
                self._read()
            except StopIteration as error:
                raise DatasetExhaustedError(n, advanced) from error
            advanced += 1

    def __enter__(self):
        """Enter an ownership boundary; acquisition remains lazy."""

        return self

    def __exit__(self, exc_type, error, traceback):
        """Close on exit, retaining consumer failures and interruption as primary.

        Cleanup alone is reported normally. Unexpected consumer StopIteration is
        translated to RuntimeError with the original exception as its cause.
        """

        if error is None:
            self.close()
        elif isinstance(error, StopIteration):
            failure = RuntimeError("Unexpected StopIteration in cursor consumer.")
            self._close_after_failure(failure)
            raise failure from error
        else:
            self._close_after_failure(error)
        return False

    @staticmethod
    def _call(function, *args, **kwargs):
        """Call a non-pull operation without mistaking StopIteration for EOF."""

        try:
            return function(*args, **kwargs)
        except StopIteration as error:
            raise RuntimeError("Unexpected StopIteration outside an iterator pull.") from error

    def _pull(self, iterator):
        """Pull an iterator, deferring known EOF cleanup errors to root policy.

        A child may have proved EOF before cleanup failed. The parent must first
        validate its own strict bound; only then can it certify its root EOF.
        """

        try:
            return next(iterator)
        except StopIteration:
            raise
        except BaseException as error:
            if isinstance(iterator, _CursorLifecycle) and iterator.observed_eof:
                self._deferred_cleanup.append(error)
                raise StopIteration from error
            raise

    def _own(self, resource):
        """Retain an acquired resource for exactly one reverse-order close attempt."""

        if id(resource) not in self._resource_ids:
            if isinstance(resource, _CursorLifecycle):
                # The parent owns cleanup ordering across sibling cursors. Child
                # termination records evidence but cannot close ahead of them.
                resource._cleanup_deferred = True
            self._resource_ids.add(id(resource))
            self._resources.append(resource)
        return resource

    def _release(self, resource) -> None:
        """Close and forget a finished child, bounding repeated-epoch ownership."""

        try:
            close = self._call(getattr, resource, "close", None)
            if close is not None:
                self._call(close)
        finally:
            self._resources = [item for item in self._resources if item is not resource]
            self._resource_ids.discard(id(resource))

    def close(self) -> None:
        """Close acquired resources once in reverse order, without proving EOF.

        Raises:
            BaseException: First cleanup failure after every resource gets an
                attempt. Unexpected cleanup StopIteration becomes RuntimeError
                with the original cause. Further failures are exception notes.
        """

        if self._cleanup_done:
            return
        self._closed = True
        self._cleanup_done = True
        errors = self._close_resources()
        if errors:
            primary, *additional = errors
            self._annotate_cleanup(primary, additional)
            raise primary

    def _close_after_failure(self, primary: BaseException) -> None:
        """Close after failure without replacing the primary diagnostic."""

        if self._cleanup_done:
            return
        self._closed = True
        if self._cleanup_deferred:
            return
        self._cleanup_done = True
        self._annotate_cleanup(primary, self._close_resources())

    @staticmethod
    def _annotate_cleanup(primary: BaseException, errors: list[BaseException]) -> None:
        add_note = getattr(primary, "add_note", None)
        if errors and add_note is not None:
            add_note(
                "Cursor cleanup also failed: "
                + "; ".join(f"{type(error).__name__}: {error}" for error in errors)
            )

    def _close_resources(self) -> list[BaseException]:
        errors = self._deferred_cleanup
        self._deferred_cleanup = []
        for resource in reversed(self._resources):
            try:
                close = self._call(getattr, resource, "close", None)
                if close is not None:
                    self._call(close)
            except BaseException as error:
                errors.append(error)
        return errors


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
    Child Methods retain their own direct-call state. Failed replacement leaves
    the previous complete plan and all independently captured cursors usable.
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

        Side Effects:
            Publishes a complete local plan only after selection, inference,
            signature compilation, and assertions succeed. Child Method state
            and previously captured cursors are never changed.
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
        edges = []
        for node in records:
            selected = getattr(node, "selected", None)
            edge = None if selected is None else selected.conversion_edge
            if edge is not None and edge not in edges:
                edges.append(edge)
        self._root, self._nodes, self._conversion_edges = root, tuple(records), tuple(edges)

    def eager(self) -> None:
        """Discard only this plan's selections; captured cursors and children survive."""

        self._root = None
        self._nodes = ()
        self._conversion_edges = ()

    def iterator(self, *, epoch: int = 0) -> "StreamGraphCursor":
        """Return one isolated pull cursor over this prepared graph.

        Args:
            epoch: Logical epoch forwarded only to qualified ``Take`` boundaries.
                Other prepared operators preserve their ordinary traversal
                semantics around that boundary.

        Raises:
            RuntimeError: If :meth:`learn` has not prepared this graph.
        """

        if self._root is None:
            raise RuntimeError("Dataset graph iteration requires graph.learn() first.")
        if type(epoch) is not int or epoch < 0:
            raise ValueError("epoch must be a nonnegative exact int.")
        return StreamGraphCursor(self._root, epoch=epoch)

    def _build(self, dataset: object, records: list[object]) -> _StreamOp:
        from .ir import MethodGraphNode, MethodPort
        from .signature import spec_node

        # Only qualified concrete operators opt in.  Other Dataset subclasses
        # inherit the source marker but must not silently gain stream semantics.
        kind = type(dataset).__dict__.get("_stream_operator")
        if kind is None:
            kind = "unsupported" if hasattr(dataset, "src") else "source"
        if kind not in {"source", "map", "batch", "unbatch", "take", "zip", "chain", "custom"}:
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
                # the composed occurrences visible through the same local build
                # used by MethodGraph, without publishing child Method state.
                selected, element_nodes = dataset.method._prepare_local(dataset.src.spec)
                element_nodes = tuple(node for node in element_nodes if node.kind == "method")
                records.extend(replace(node, occurrence=len(records)) for node in element_nodes)
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
        if kind == "take":
            child = self._build(dataset.src, records)
            records.append(MethodGraphNode(
                len(records), "method", type(dataset),
                inputs=(MethodPort("iterator", spec_node(dataset.src.spec)),),
                outputs=(MethodPort("iterator", spec_node(spec)),),
                stream_node=self._builtin_node(
                    "Take", (dataset.src.spec,), spec, dataset,
                    ("strict_prefix",), 0,
                ),
            ))
            return _StreamOp("take", dataset=dataset, inputs=(child,))
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

        cardinality = dataset.yield_cardinality()
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


class StreamGraphCursor(_CursorLifecycle, Iterator[Any]):
    """Own one local graph traversal and all resources acquired for it.

    The cursor acquires lazily, closes resources once in reverse acquisition order,
    and preserves a body/acquisition/consumer failure if cleanup also fails. Use
    it as a context manager around consumer work. ``observed_eof`` records only
    successful root termination, independently of closure; cleanup errors retain
    that evidence. Unexpected non-pull StopIteration is a causal RuntimeError.
    It does not serialize or retain a generator frame for replay.
    """

    def __init__(self, root: _StreamOp, *, epoch: int = 0) -> None:
        super().__init__()
        self._root_op = root
        self._epoch = epoch
        self._root: Iterator[Any] | None = None
        self._position = 0

    def __iter__(self) -> "StreamGraphCursor":
        """Return this cursor as its own iterator."""

        return self

    @property
    def position(self) -> int:
        """Return the number of graph output elements already yielded."""

        return self._position

    def _read(self) -> Any:
        if self._root is None:
            self._root = self._call(self._open, self._root_op)
        value = self._pull(self._root)
        self._position += 1
        return value

    def _open(self, op: _StreamOp, *, epoch: int | None = None) -> Iterator[Any]:
        """Open one operation at an inherited epoch while retaining ownership."""

        if epoch is None:
            epoch = self._epoch
        if op.kind == "source":
            opener = getattr(op.dataset, "iterator_for_epoch", None)
            return self._own(self._call(opener, epoch) if callable(opener) else self._call(op.dataset.iterator))
        if op.kind == "map":
            source = self._open(op.inputs[0], epoch=epoch)
            return self._own(_MapIterator(self, source, op.dataset, op.selected))
        if op.kind == "batch":
            source = self._open(op.inputs[0], epoch=epoch)
            return self._own(_BatchIterator(self, source, op.dataset))
        if op.kind == "unbatch":
            source = self._open(op.inputs[0], epoch=epoch)
            return self._own(_UnbatchIterator(self, source, op.dataset))
        if op.kind == "take":
            if op.dataset.n == 0:
                return self._own(iter(()))
            source_epoch = op.dataset.epoch if op.dataset.fixed_prefix else op.dataset.epoch + epoch
            source = self._open(op.inputs[0], epoch=source_epoch)
            return self._own(_TakeIterator(self, source, op.dataset.n))
        if op.kind == "zip":
            return self._own(_ZipIterator(self, op.inputs, op.tree, epoch))
        if op.kind == "chain":
            return self._own(_ChainIterator(self, op.inputs, epoch))
        if op.kind == "custom":
            return self._own(_CustomIterator(self, op.inputs, op.node, epoch))
        raise AssertionError(f"unknown stream operation {op.kind!r}")


class _MapIterator(Iterator[Any]):
    """One Map output iterator with an occurrence-local discovery buffer."""

    def __init__(self, cursor, source: Iterator[Any], dataset: object, selected: object | None) -> None:
        self.cursor = cursor
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
            self.cursor._call(self._resolve)
        if self.selected is None:
            raise StopIteration
        if self.has_pending:
            item = self.pending
            self.pending = None
            self.has_pending = False
        else:
            item = self.cursor._pull(self.source)
        result = self.cursor._call(self.selected, item)
        if getattr(self.dataset, "preserves_examples", False):
            source_examples = self.cursor._call(self.dataset.src.examples_in, item)
            if self.cursor._call(self.dataset.examples_in, result) != source_examples:
                raise ValueError("Map declared preserves_examples but changed an example count.")
        return result

    def _resolve(self) -> None:
        try:
            implementation = self.cursor._call(
                self.dataset.method.find_implementation, input_spec=self.dataset.src.spec,
            )
        except ImplementationSelectionError as error:
            if error.reason != "unknown_traits" or error.unknown_traits != ("backend",):
                raise
            try:
                item = self.cursor._pull(self.source)
            except StopIteration:
                self.resolved = True
                return
            from dryml.core.backend import discover_backends

            backends = self.cursor._call(discover_backends, item)
            if len(backends) != 1:
                raise ImplementationSelectionError("conflict" if backends else "unknown_traits", () if backends else ("backend",))
            implementation = self.cursor._call(self.dataset.method.find_implementation,
                input_spec=self.dataset.src.spec,
                backend=next(iter(backends)),
            )
            self.pending = item
            self.has_pending = True
        self.selected = self.cursor._call(implementation.prepared_invoker)
        self.resolved = True


class _BatchIterator(Iterator[Any]):
    """One bounded Batch output iterator retaining its collate selection once."""

    def __init__(self, cursor, source: Iterator[Any], dataset: object) -> None:
        self.cursor = cursor
        self.source = source
        self.dataset = dataset
        self.collate = None

    def __iter__(self) -> "_BatchIterator":
        return self

    def __next__(self) -> Any:
        batch = []
        for _ in range(self.dataset.batch_size):
            try:
                batch.append(self.cursor._pull(self.source))
            except StopIteration:
                break
        if not batch or (self.dataset.drop_remainder and len(batch) < self.dataset.batch_size):
            raise StopIteration
        if self.collate is None:
            self.collate = self.cursor._call(self.dataset._resolve_collate, batch)
        return self.cursor._call(self.collate, batch)


class _UnbatchIterator(Iterator[Any]):
    """One Unbatch output iterator retaining one selected split implementation."""

    def __init__(self, cursor, source: Iterator[Any], dataset: object) -> None:
        self.cursor = cursor
        self.source = source
        self.split = cursor._call(dataset._resolve_split)
        self.pending: Iterator[Any] = iter(())

    def __iter__(self) -> "_UnbatchIterator":
        return self

    def __next__(self) -> Any:
        while True:
            try:
                return self.cursor._pull(self.pending)
            except StopIteration:
                value = self.cursor._pull(self.source)
                self.cursor._release(self.pending)
                self.pending = self.cursor._own(self.cursor._call(iter, self.cursor._call(self.split, value)))


class _TakeIterator(Iterator[Any]):
    """Yield one strict prepared prefix without bypassing its child graph."""

    def __init__(self, cursor, source: Iterator[Any], count: int) -> None:
        self.cursor = cursor
        self.source = source
        self.count = count
        self.position = 0

    def __iter__(self) -> "_TakeIterator":
        return self

    def __next__(self) -> Any:
        if self.position >= self.count:
            raise StopIteration
        try:
            value = self.cursor._pull(self.source)
        except (StopIteration, DatasetExhaustedError) as error:
            raise DatasetExhaustedError(self.count, self.position) from error
        self.position += 1
        return value


class _ZipIterator(Iterator[Any]):
    """Pull inputs in declared order and stop at the first exhausted position."""

    def __init__(self, cursor: StreamGraphCursor, inputs: tuple[_StreamOp, ...], tree: object, epoch: int) -> None:
        self.cursor = cursor
        self.inputs = inputs
        self.tree = tree
        self.epoch = epoch
        self.iterators: tuple[Iterator[Any], ...] | None = None

    def __iter__(self) -> "_ZipIterator":
        return self

    def __next__(self) -> Any:
        if self.iterators is None:
            self.iterators = tuple(self.cursor._call(self.cursor._open, op, epoch=self.epoch) for op in self.inputs)
        values = [self.cursor._pull(iterator) for iterator in self.iterators]
        value_iter = iter(values)
        return self.cursor._call(_build_zip_tree, self.tree, value_iter)


class _ChainIterator(Iterator[Any]):
    """Open sources only as their declared predecessors become exhausted."""

    def __init__(self, cursor: StreamGraphCursor, inputs: tuple[_StreamOp, ...], epoch: int) -> None:
        self.cursor = cursor
        self.inputs = inputs
        self.epoch = epoch
        self.index = 0
        self.current: Iterator[Any] | None = None

    def __iter__(self) -> "_ChainIterator":
        return self

    def __next__(self) -> Any:
        while self.index < len(self.inputs):
            if self.current is None:
                self.current = self.cursor._call(self.cursor._open, self.inputs[self.index], epoch=self.epoch)
            try:
                return self.cursor._pull(self.current)
            except StopIteration:
                self.current = None
                self.index += 1
        raise StopIteration


class _CustomIterator(Iterator[Any]):
    """Open one custom output lazily after its ordered inputs are acquired."""

    def __init__(self, cursor: StreamGraphCursor, inputs: tuple[_StreamOp, ...], node: StreamNode, epoch: int) -> None:
        self.cursor = cursor
        self.inputs = inputs
        self.node = node
        self.epoch = epoch
        self.output: Iterator[Any] | None = None

    def __iter__(self) -> "_CustomIterator":
        return self

    def __next__(self) -> Any:
        if self.output is None:
            inputs = tuple(self.cursor._call(self.cursor._open, op, epoch=self.epoch) for op in self.inputs)
            self.output = self.cursor._own(self.cursor._call(self.node.open, *inputs))
        return self.cursor._pull(self.output)


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
