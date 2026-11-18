"""Explicit iterator-port declarations and synchronous local stream execution.

This module keeps stream consumption separate from element-call arity.  Stream
nodes borrow their inputs; a :class:`StreamGraphCursor` owns every resource it
opens for one graph traversal.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from typing import Any, TYPE_CHECKING

from dryml.core.cardinality import Cardinality
from dryml.core.tensor_spec import SpecTree

from .errors import ImplementationSelectionError
from .stream_state import PullKind, SourceBinding

if TYPE_CHECKING:
    from .signature import MethodCallNode


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
                with the original cause. Further failures are exception notes,
                retained in ``__notes__`` even on Python without ``add_note``.
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
        if not errors:
            return
        message = "Cursor cleanup also failed: " + "; ".join(
            f"{type(error).__name__}: {error}" for error in errors
        )
        add_note = getattr(primary, "add_note", None)
        if callable(add_note):
            add_note(message)
        else:
            primary.__notes__ = [*getattr(primary, "__notes__", ()), message]

    def _close_resources(self) -> list[BaseException]:
        errors = self._deferred_cleanup
        self._deferred_cleanup = []
        resources, self._resources = self._resources, []
        self._resource_ids.clear()
        for resource in reversed(resources):
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
    """One frozen execution occurrence; dataset is an inspection reference only."""

    kind: str
    dataset: object | None = None
    inputs: tuple["_StreamOp", ...] = ()
    selected: object | None = None
    node: StreamNode | None = None
    tree: object | None = None
    signature: MethodCallNode | None = None
    controls: tuple = ()
    source: SourceBinding | None = None
    occurrence: tuple[int, ...] = ()
    cardinality: Cardinality = Cardinality.UNKNOWN
    sectioned: bool = False
    sections: tuple | None = None
    section_bound: int | None = None

    @property
    def spec(self) -> SpecTree | None:
        from .signature import spec_from_node

        return None if self.signature is None else spec_from_node(self.signature)


@dataclass(frozen=True, slots=True)
class _PreparedStream:
    """Atomically published execution and inspection facts for one preparation."""

    root: _StreamOp
    nodes: tuple
    conversion_edges: tuple


@dataclass(frozen=True, slots=True)
class _ZipTree:
    """Immutable structural recipe preserving mapping order and sequence kinds."""

    kind: str
    children: tuple = ()

    def build(self, values):
        if self.kind == "leaf":
            return next(values)
        if self.kind == "dict":
            return {key: child.build(values) for key, child in self.children}
        items = (child.build(values) for child in self.children)
        return tuple(items) if self.kind == "tuple" else list(items)


def _freeze_zip_tree(tree, leaves):
    if hasattr(tree, "_stream_operator"):
        leaves.append(tree)
        return _ZipTree("leaf")
    if isinstance(tree, dict):
        return _ZipTree("dict", tuple((key, _freeze_zip_tree(value, leaves)) for key, value in tree.items()))
    if isinstance(tree, (tuple, list)):
        return _ZipTree("tuple" if isinstance(tree, tuple) else "list",
                        tuple(_freeze_zip_tree(value, leaves) for value in tree))
    raise TypeError(f"Zip expects Dataset leaves, got {type(tree).__name__}.")


def _section_total(runs, *, bound=None):
    if bound == 0:
        return Cardinality.finite(0)
    if runs is None:
        return Cardinality.UNKNOWN
    if any(count is None and size for size, count in runs):
        return Cardinality.INFINITE
    return Cardinality.finite(sum(size * count for size, count in runs if count is not None))


def _section_size(runs, index):
    if runs is None:
        return None
    for size, count in runs:
        if count is None or index < count:
            return size
        index -= count
    return None


def _prefix_sections(runs, n, *, skip=False):
    if runs is None:
        return None
    result = []
    for size, count in runs:
        if n == 0:
            if skip:
                result.append((size, count))
                continue
            break
        if size == 0:
            result.append((0, count))
            continue
        whole = n // size
        if count is not None:
            whole = min(count, whole)
        if whole:
            result.append((0 if skip else size, whole))
            n -= whole * size
        remaining = None if count is None else count - whole
        if remaining == 0:
            continue
        if n:
            result.append((size - n if skip else n, 1))
            n = 0
            remaining = None if remaining is None else remaining - 1
        if skip and remaining != 0:
            result.append((size, remaining))
        if not skip:
            break
    return tuple(result)


def _zip_sections(branches):
    if any(runs is None for runs in branches):
        return None
    positions = [0] * len(branches)
    remaining = [runs[0][1] if runs else 0 for runs in branches]
    result = []
    while all(index < len(runs) for index, runs in zip(positions, branches)):
        finite = [count for count in remaining if count is not None]
        count = min(finite) if finite else None
        size = min(runs[index][0] for index, runs in zip(positions, branches))
        result.append((size, count))
        if count is None:
            break
        for i, runs in enumerate(branches):
            if remaining[i] is not None:
                remaining[i] -= count
                if remaining[i] == 0:
                    positions[i] += 1
                    if positions[i] < len(runs):
                        remaining[i] = runs[positions[i]][1]
    return tuple(result)


def _dataset_sections(dataset):
    """Validate one visible axis and prove compact section-size runs, without IO.

    A run is (final-output size, repetition count); None repetitions are infinite.
    Unknown sizes remain unknown, separately from an optional per-section bound.
    This metadata seam never selects invokers or treats source lengths as pulls.
    """
    kind = type(dataset).__dict__.get("_stream_operator")
    if kind in {"zip", "chain", "custom"}:
        leaves = []
        _freeze_zip_tree(dataset.sources, leaves)
        children = [_dataset_sections(child) for child in leaves]
    elif kind is None and hasattr(dataset, "sources"):
        def declared_leaves(tree):
            if hasattr(tree, "_stream_operator"):
                yield tree
            elif isinstance(tree, (tuple, list, dict)):
                for child in (tree.values() if isinstance(tree, dict) else tree):
                    yield from declared_leaves(child)
        children = [_dataset_sections(child) for child in declared_leaves(dataset.sources)]
    elif hasattr(getattr(dataset, "src", None), "_stream_operator"):
        children = [_dataset_sections(dataset.src)]
    else:
        children = []
    modes = [child[0] for child in children]
    if kind == "epoch":
        if any(modes):
            raise ValueError("Nested Epoch axes are unsupported.")
        length = dataset.length
        cardinality = dataset.src.yield_cardinality()
        if cardinality.is_infinite:
            runs = ((length, None),)
        elif cardinality.is_finite:
            full, tail = divmod(cardinality.require_finite(), length)
            runs = (() if not full else ((length, full),)) + (() if not tail else ((tail, 1),))
        else:
            runs = None
        return True, runs, length
    if not any(modes):
        return False, None, None
    if kind in {"chain", "zip"}:
        if not all(modes):
            raise ValueError("Mixed explicit and unsectioned boundary modes are unsupported.")
        runs = (None if any(child[1] is None for child in children)
                else tuple(run for child in children for run in child[1]))
        bounds = [child[2] for child in children]
        if kind == "zip":
            runs = _zip_sections([child[1] for child in children])
            known = [bound for bound in bounds if bound is not None]
            bound = min(known) if known else None
        else:
            bound = max(bounds) if all(value is not None for value in bounds) else None
        return True, runs, bound
    if kind not in {"map", "batch", "unbatch", "take", "skip", "repeat", "shuffle"}:
        raise NotImplementedError("Unsupported custom operator over sectioned input.")
    _, runs, bound = children[0]
    if kind == "batch":
        size, drop = dataset.batch_size, dataset.drop_remainder
        def groups(n):
            return n // size if drop else (n + size - 1) // size
        runs = None if runs is None else tuple((groups(n), count) for n, count in runs)
        bound = None if bound is None else groups(bound)
    elif kind == "unbatch":
        from dryml.core.tensor_spec import TensorSpec, Dynamic
        spec = dataset.src.spec
        if isinstance(spec, TensorSpec) and spec.batched and spec.batch is not Dynamic:
            runs = None if runs is None else tuple((n * spec.batch, count) for n, count in runs)
            bound = None if bound is None else bound * spec.batch
        elif type(dataset.src).__dict__.get("_stream_operator") == "batch":
            batch = dataset.src
            _, runs, bound = _dataset_sections(batch.src)
            if batch.drop_remainder:
                runs = None if runs is None else tuple((n // batch.batch_size * batch.batch_size, count) for n, count in runs)
        else:
            runs, bound = None, None
    elif kind in {"take", "skip"}:
        runs = _prefix_sections(runs, dataset.n, skip=kind == "skip")
        if kind == "take":
            bound = dataset.n if bound is None else min(bound, dataset.n)
    elif kind == "repeat":
        count = dataset.count
        if count == 0:
            runs = ()
        elif runs is not None:
            if len(runs) == 1:
                n, repeats = runs[0]
                runs = ((n, None if repeats is None or count is None else repeats * count),)
            elif count is None:
                runs = None
            else:
                runs = runs * count
    return True, runs, bound


def _needs_boundary_kernel(op):
    return op.sectioned or op.kind in {"skip", "repeat", "shuffle", "epoch"} or any(
        _needs_boundary_kernel(child) for child in op.inputs)


class StreamPlan:
    """Prepare and open the qualified synchronous Dataset stream subset.

    Args:
        dataset: Root Dataset-like object whose qualified operators expose
            ``_stream_operator`` and normal Dataset source capabilities.

    Planning freezes controls, occurrence topology, specs, source adapters, and
    selected Map/collate/split invokers but never opens a source. Source payloads
    retain their ownership contracts; freezing does not clone their bytes.
    Each :meth:`iterator` creates isolated cursor/discovery state and therefore
    reopens every source occurrence independently.
    Child Methods retain their own direct-call state. Failed replacement leaves
    the previous complete plan and all independently captured cursors usable.
    """

    def __init__(self, dataset: object) -> None:
        self.dataset = dataset
        self._prepared: _PreparedStream | None = None

    @property
    def nodes(self) -> tuple[object, ...]:
        """Return immutable inspection nodes generated by the most recent plan."""

        prepared = self._prepared
        return () if prepared is None else prepared.nodes

    @property
    def conversion_edges(self) -> tuple[object, ...]:
        """Return immutable edges selected by Map occurrences in this stream plan."""

        prepared = self._prepared
        return () if prepared is None else prepared.conversion_edges

    def learn(
        self,
        input_spec: SpecTree | None = None,
        additional_input_specs: tuple[SpecTree, ...] = (),
        *,
        strategy: str = "local",
        output_spec: SpecTree | None = None,
    ) -> None:
        """Prepare qualified operators without opening sources or invoking bodies.

        Args:
            input_spec: Optional first root-source spec assertion.
            additional_input_specs: Optional later root-source spec assertions.
            strategy: Only ``"local"`` is supported.
            output_spec: Optional root output spec assertion.

        Raises:
            TypeError, ValueError: If specs, source controls, assertions or strategy
                are incompatible. Source binding/inference failures propagate.
            NotImplementedError: If the graph includes an unqualified operator.
            ImplementationSelectionError: If a known Map spec cannot select one
                local implementation. Native collate/split selection errors also
                propagate, including unavailable declared-backend dependencies.

        Side Effects:
            Publishes a complete local plan only after selection, inference,
            signature compilation, and assertions succeed. Child Method state
            and previously captured cursors are never changed.
            Declared collate/split selection may activate its configured backend;
            no factory, source iterator, or selected Method body is executed.
        """

        if strategy != "local":
            raise ValueError(f"Unsupported Method preparation strategy {strategy!r}; only 'local' is available.")
        records: list[object] = []
        _dataset_sections(self.dataset)
        root = self._build(self.dataset, records)
        from .signature import spec_from_node, spec_node

        root_specs = tuple(
            spec_from_node(node.outputs[0].spec)
            for node in records
            if node.kind == "source" and node.outputs and node.outputs[0].spec is not None
        )
        supplied = (() if input_spec is None else (input_spec, *additional_input_specs))
        if supplied and supplied != root_specs:
            raise ValueError("Dataset graph input specs do not match its declared source specs.")
        if output_spec is not None and spec_node(output_spec) != spec_node(root.spec):
            raise ValueError("Dataset graph output spec does not match its declared output spec.")
        edges = []
        for node in records:
            selected = getattr(node, "selected", None)
            edge = None if selected is None else selected.conversion_edge
            if edge is not None and edge not in edges:
                edges.append(edge)
        self._prepared = _PreparedStream(root, tuple(records), tuple(edges))

    def eager(self) -> None:
        """Discard only this plan's selections; captured cursors and children survive."""

        self._prepared = None

    def iterator(self, *, epoch: int = 0) -> "StreamGraphCursor":
        """Return one isolated pull cursor over this prepared graph.

        Args:
            epoch: Logical epoch forwarded only to qualified ``Take`` boundaries.
                Other prepared operators preserve their ordinary traversal
                semantics around that boundary.

        Raises:
            RuntimeError: If :meth:`learn` has not prepared this graph.
            ValueError: If ``epoch`` is not a nonnegative exact integer.

        Returns:
            A lazy closeable cursor retaining this complete preparation even if a
            later plan is published or reset. Creation acquires no source.
        """

        prepared = self._prepared
        if prepared is None:
            raise RuntimeError("Dataset graph iteration requires graph.learn() first.")
        if type(epoch) is not int or epoch < 0:
            raise ValueError("epoch must be a nonnegative exact int.")
        return StreamGraphCursor(prepared.root, epoch=epoch)

    def _build(self, dataset: object, records: list[object], occurrence: tuple[int, ...] = ()) -> _StreamOp:
        from .ir import MethodGraphNode, MethodPort
        from .signature import spec_node, spec_from_node
        from dryml.core.tensor_spec import batch_spec_tree, unbatch_spec_tree, merge_spec_trees, Dynamic

        # Only qualified concrete operators opt in.  Other Dataset subclasses
        # inherit the source marker but must not silently gain stream semantics.
        kind = type(dataset).__dict__.get("_stream_operator")
        if kind is None:
            kind = "unsupported" if hasattr(dataset, "src") else "source"
        if kind not in {"source", "map", "batch", "unbatch", "take", "skip", "repeat", "shuffle", "epoch", "zip", "chain", "custom"}:
            raise NotImplementedError(f"Dataset graph planning does not support {type(dataset).__name__}.")
        spec = spec_from_node(spec_node(getattr(dataset, "spec")))
        cardinality = dataset.yield_cardinality()

        def operation(kind, **kwargs):
            sectioned, sections, bound = _dataset_sections(dataset)
            return _StreamOp(kind, dataset=dataset, signature=spec_node(spec),
                             occurrence=occurrence, cardinality=cardinality,
                             sectioned=sectioned, sections=sections, section_bound=bound, **kwargs)

        if kind == "source":
            records.append(MethodGraphNode(len(records), "source", None, outputs=(MethodPort("iterator", spec_node(spec)),)))
            return operation("source", source=dataset._prepare_stream_source(spec))
        if kind == "map":
            child = self._build(dataset.src, records, occurrence + (0,))
            method = dataset.method
            selected = None
            try:
                # A Map is an iterator boundary around an element Method graph.  Keep
                # the composed occurrences visible through the same local build
                # used by MethodGraph, without publishing child Method state.
                selected, element_nodes = method._prepare_local(child.spec)
                element_nodes = tuple(node for node in element_nodes if node.kind == "method")
                spec = spec_from_node(element_nodes[0].outputs[0].spec)
                records.extend(replace(node, occurrence=len(records)) for node in element_nodes)
            except ImplementationSelectionError as error:
                if error.reason != "unknown_traits" or error.unknown_traits != ("backend",):
                    raise
            records.append(MethodGraphNode(
                len(records), "method", type(dataset.method),
                inputs=(MethodPort("iterator", spec_node(child.spec)),),
                outputs=(MethodPort("iterator", spec_node(spec)),),
                selected=selected,
                traits=None if selected is None else selected.traits,
                stream_node=self._builtin_node("Map", (child.spec,), spec, dataset, ("one_to_one",), 1),
            ))
            return operation("map", inputs=(child,), selected=selected,
                             controls=(getattr(dataset, "preserves_examples", False),
                                       method.find_implementation, dataset._examples_in_spec))
        if kind in {"batch", "unbatch"}:
            child = self._build(dataset.src, records, occurrence + (0,))
            controls = ((dataset.batch_size, dataset.drop_remainder, dataset.ragged)
                        if kind == "batch" else ())
            spec = (batch_spec_tree(child.spec, batch=dataset.batch_size if dataset.drop_remainder else Dynamic,
                                    ragged=dataset.ragged)
                    if kind == "batch" else unbatch_spec_tree(child.spec))
            records.append(MethodGraphNode(
                len(records), "method", type(dataset),
                inputs=(MethodPort("iterator", spec_node(child.spec)),),
                outputs=(MethodPort("iterator", spec_node(spec)),),
                stream_node=self._builtin_node(
                    type(dataset).__name__,
                    (child.spec,),
                    spec,
                    dataset,
                    ("bounded_group" if kind == "batch" else "one_to_many",),
                    dataset.batch_size if kind == "batch" else 1,
                ),
            ))
            selected = dataset._prepare_stream_selection(kind, child.spec)
            return operation(kind, inputs=(child,), controls=controls, selected=selected)
        if kind == "take":
            child = self._build(dataset.src, records, occurrence + (0,))
            spec = child.spec
            records.append(MethodGraphNode(
                len(records), "method", type(dataset),
                inputs=(MethodPort("iterator", spec_node(child.spec)),),
                outputs=(MethodPort("iterator", spec_node(spec)),),
                stream_node=self._builtin_node(
                    "Take", (child.spec,), spec, dataset,
                    ("strict_prefix",), 0,
                ),
            ))
            return operation("take", inputs=(child,), controls=(dataset.n, dataset.epoch, dataset.fixed_prefix))
        if kind in {"skip", "repeat", "shuffle", "epoch"}:
            child = self._build(dataset.src, records, occurrence + (0,))
            controls = {"skip": lambda: (dataset.n,), "repeat": lambda: (dataset.count,),
                        "shuffle": lambda: (dataset.buffer_size, dataset.seed),
                        "epoch": lambda: (dataset.length,)}[kind]()
            spec = child.spec
            records.append(MethodGraphNode(
                len(records), "method", type(dataset),
                inputs=(MethodPort("iterator", spec_node(child.spec)),),
                outputs=(MethodPort("iterator", spec_node(spec)),),
                stream_node=self._builtin_node(type(dataset).__name__, (child.spec,), spec,
                    dataset, (kind,), dataset.buffer_size if kind == "shuffle" else 0),
            ))
            return operation(kind, inputs=(child,), controls=controls)
        if kind == "zip":
            leaves = []
            tree = _freeze_zip_tree(dataset.sources, leaves)
            children = tuple(self._build(source, records, occurrence + (index,)) for index, source in enumerate(leaves))
            spec = tree.build(iter(child.spec for child in children))
            records.append(MethodGraphNode(
                len(records), "method", type(dataset),
                inputs=tuple(MethodPort("iterator", spec_node(child.spec)) for child in children),
                outputs=(MethodPort("iterator", spec_node(spec)),),
                stream_node=self._builtin_node("Zip", tuple(child.spec for child in children), spec, dataset, ("position_order",) * len(leaves), 0),
            ))
            return operation("zip", inputs=children, tree=tree)
        if kind == "chain":
            children = tuple(self._build(source, records, occurrence + (index,)) for index, source in enumerate(tuple(dataset.sources)))
            spec = merge_spec_trees([child.spec for child in children])
            records.append(MethodGraphNode(
                len(records), "method", type(dataset),
                inputs=tuple(MethodPort("iterator", spec_node(child.spec)) for child in children),
                outputs=(MethodPort("iterator", spec_node(spec)),),
                stream_node=self._builtin_node("Chain", tuple(child.spec for child in children), spec, dataset, ("declaration_order",) * len(dataset.sources), 0),
            ))
            return operation("chain", inputs=children)
        children = tuple(self._build(source, records, occurrence + (index,)) for index, source in enumerate(tuple(dataset.sources)))
        node = dataset.node
        spec = node.output_spec(tuple(child.spec for child in children))
        records.append(MethodGraphNode(
            len(records), "method", type(node),
            inputs=tuple(MethodPort("iterator", spec_node(child.spec)) for child in children),
            outputs=(MethodPort("iterator", spec_node(spec)),),
            stream_node=node,
        ))
        return operation("custom", inputs=children, node=node)

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

    Args:
        root: Frozen kernel occurrence descriptor supplied by a prepared graph.
            Applications normally obtain cursors through graph ``iterator()``.
        epoch: Logical epoch inherited by source bindings and Take controls.

    The cursor acquires lazily, closes resources once in reverse acquisition order,
    and preserves a body/acquisition/consumer failure if cleanup also fails. Use
    it as a context manager around consumer work. ``observed_eof`` records only
    successful root termination, independently of closure; cleanup errors retain
    that evidence. Unexpected non-pull StopIteration is a causal RuntimeError.
    Backend-unknown Map occurrences discover their first value once through core's
    runtime discovery, lazily activating a missing native adapter without changing
    shared Dataset specs. Unavailable dependencies raise ContextError. The cursor
    does not serialize or retain a generator frame for replay.
    The optional ``pull`` interface exposes delivery/root facts out of band;
    ordinary ``next`` returns unwrapped authored values with the same ownership.
    """

    def __init__(self, root: _StreamOp, *, epoch: int = 0) -> None:
        super().__init__()
        self._root_op = root
        self._epoch = epoch
        self._root: Iterator[Any] | None = None
        self._position = 0
        self._value: Any = None
        self._has_value = False
        self._kernel = None
        if _needs_boundary_kernel(root):
            self._kernel = self._own(_SectionExecution(root, epoch))

    def __iter__(self) -> "StreamGraphCursor":
        """Return this cursor as its own iterator."""

        return self

    @property
    def position(self) -> int:
        """Return the number of graph output elements already yielded."""

        return self._position

    @property
    def value(self) -> Any:
        """Return the borrowed output of the last successful explicit ``pull``.

        Raises RuntimeError before delivery or after a later non-delivery pull.
        Ordinary ``next`` calls do not retain a separate output reference.
        """
        if not self._has_value:
            raise RuntimeError("No explicit stream delivery is available.")
        return self._value

    def pull(self) -> PullKind:
        """Advance the flattened cursor using the optional out-of-band protocol.

        Returns DELIVERY with a borrowed ``value``, or ROOT_END after positively
        observed output EOF. This one-pass adapter emits no SECTION_END. Explicit
        closure without EOF raises RuntimeError; acquisition/execution/cleanup
        failures propagate unchanged and never become completion signals. No
        state capture or encoding is performed.
        """
        self._has_value = False
        self._value = None
        if self._closed and not self.observed_eof:
            raise RuntimeError("Closed stream has no root-completion evidence.")
        try:
            self._value = next(self)
        except StopIteration:
            return PullKind.ROOT_END
        self._has_value = True
        return PullKind.DELIVERY

    def _read(self) -> Any:
        if self._has_value:
            self._has_value = False
            self._value = None
        if self._kernel is not None:
            while True:
                kind = self._kernel.pull()
                if kind == PullKind.ROOT_END:
                    raise StopIteration
                if kind == PullKind.SECTION_END:
                    self._kernel.handoff()
                    continue
                self._position += 1
                return self._kernel.value
        if self._root is None:
            self._root = self._call(self._open, self._root_op)
        value = self._pull(self._root)
        self._position += 1
        return value

    def close(self) -> None:
        """Release graph resources and any borrowed explicit-pull value once.

        Propagates the first cleanup error after attempting all owned resources;
        closure alone never establishes root EOF. Subsequent closes are no-ops.
        """
        self._has_value = False
        self._value = None
        super().close()

    def _open(self, op: _StreamOp, *, epoch: int | None = None) -> Iterator[Any]:
        """Open one operation at an inherited epoch while retaining ownership."""

        if epoch is None:
            epoch = self._epoch
        if op.kind == "source":
            return self._own(self._call(op.source.open, epoch))
        if op.kind == "map":
            source = self._open(op.inputs[0], epoch=epoch)
            return self._own(_MapIterator(self, source, op.controls, op.selected, op.inputs[0].spec, op.spec))
        if op.kind == "batch":
            source = self._open(op.inputs[0], epoch=epoch)
            return self._own(_BatchIterator(self, source, op.controls, op.selected))
        if op.kind == "unbatch":
            source = self._open(op.inputs[0], epoch=epoch)
            return self._own(_UnbatchIterator(self, source, op.selected))
        if op.kind == "take":
            count, base_epoch, fixed_prefix = op.controls
            if count == 0:
                return self._own(iter(()))
            source_epoch = base_epoch if fixed_prefix else base_epoch + epoch
            source = self._open(op.inputs[0], epoch=source_epoch)
            return self._own(_TakeIterator(self, source, count))
        if op.kind == "zip":
            return self._own(_ZipIterator(self, op.inputs, op.tree, epoch))
        if op.kind == "chain":
            return self._own(_ChainIterator(self, op.inputs, epoch))
        if op.kind == "custom":
            return self._own(_CustomIterator(self, op.inputs, op.node, epoch))
        raise AssertionError(f"unknown stream operation {op.kind!r}")


class _MapIterator(Iterator[Any]):
    """One Map output iterator with an occurrence-local discovery buffer."""

    def __init__(self, cursor, source: Iterator[Any], controls: tuple, selected: object | None,
                 input_spec, output_spec) -> None:
        self.cursor = cursor
        self.source = source
        self.preserves_examples, self.find_implementation, self.examples_in_spec = controls
        self.selected = selected
        self.input_spec = input_spec
        self.output_spec = output_spec
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
        if self.preserves_examples:
            source_examples = self.cursor._call(self.examples_in_spec, self.input_spec, item)
            if self.cursor._call(self.examples_in_spec, self.output_spec, result) != source_examples:
                raise ValueError("Map declared preserves_examples but changed an example count.")
        return result

    def _resolve(self) -> None:
        try:
            implementation = self.cursor._call(
                self.find_implementation, input_spec=self.input_spec,
            )
        except ImplementationSelectionError as error:
            if error.reason != "unknown_traits" or error.unknown_traits != ("backend",):
                raise
            try:
                item = self.cursor._pull(self.source)
            except StopIteration:
                self.resolved = True
                return
            from .signature import _runtime_backends_for_spec

            backends = self.cursor._call(_runtime_backends_for_spec, item, self.input_spec)
            if len(backends) != 1:
                raise ImplementationSelectionError("conflict" if backends else "unknown_traits", () if backends else ("backend",))
            implementation = self.cursor._call(self.find_implementation,
                input_spec=self.input_spec,
                backend=next(iter(backends)),
            )
            self.pending = item
            self.has_pending = True
        self.selected = self.cursor._call(implementation.prepared_invoker)
        self.output_spec = self.cursor._call(implementation.infer_output_spec)
        self.resolved = True


class _BatchIterator(Iterator[Any]):
    """One bounded Batch output iterator retaining its collate selection once."""

    def __init__(self, cursor, source: Iterator[Any], controls: tuple, selected) -> None:
        self.cursor = cursor
        self.source = source
        self.batch_size, self.drop_remainder, self.ragged = controls
        self.collate = selected

    def __iter__(self) -> "_BatchIterator":
        return self

    def __next__(self) -> Any:
        batch = []
        for _ in range(self.batch_size):
            try:
                batch.append(self.cursor._pull(self.source))
            except StopIteration:
                break
        if not batch or (self.drop_remainder and len(batch) < self.batch_size):
            raise StopIteration
        return self.cursor._call(self.collate, batch)


class _UnbatchIterator(Iterator[Any]):
    """One Unbatch output iterator retaining one selected split implementation."""

    def __init__(self, cursor, source: Iterator[Any], selected) -> None:
        self.cursor = cursor
        self.source = source
        self.split = selected
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
        return self.cursor._call(self.tree.build, value_iter)


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
                self.current = self.cursor._own(StreamGraphCursor(self.inputs[self.index], epoch=self.epoch))
            try:
                return self.cursor._pull(self.current)
            except StopIteration:
                self.cursor._release(self.current)
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


class _SectionExecution(_CursorLifecycle):
    """Mutable occurrence state, with out-of-band boundaries and explicit phases.

    Child roots are local: only Repeat/Chain may replace an occurrence. A pending
    boundary is stable until handoff; no suspended operator generator encodes the
    frontier. Values are borrowed, and no continuation snapshots are constructed.
    """

    def __init__(self, op, epoch=0):
        super().__init__()
        self.op, self.epoch = op, epoch
        self.children = {}
        self.source = None
        self.value = None
        self.section_final = False
        self.root_final = False
        self.boundary = False
        self.terminal = False
        self.after_boundary_root = False
        self.count = 0
        self.within = 0
        self.index = 0
        self.produced = False
        self.selected = op.selected
        self.output_spec = op.spec
        self.input_spec = op.inputs[0].spec if op.kind == "map" else None
        self.buffer = []
        self.split_values = ()
        self.split_index = 0
        self.input_batch = None
        self.input_final = False
        self.input_root = False
        self.phase = "fill"
        self.rng = None

    def _child(self, index=0, epoch=None):
        if index not in self.children:
            child = _SectionExecution(self.op.inputs[index], self.epoch if epoch is None else epoch)
            self.children[index] = self._own(child)
        return self.children[index]

    def _forget(self, index):
        child = self.children.pop(index)
        self._release(child)

    def _deliver(self, value, *, final=False, root=False):
        self.value = value
        self.section_final, self.root_final = final, root
        self.boundary = final
        self.after_boundary_root = root and final
        if root and not final:
            self.terminal = True
        self.count += 1
        self.produced = True
        return PullKind.DELIVERY

    def _section(self, *, root=False):
        self.boundary = True
        self.after_boundary_root = root
        self.produced = True
        return PullKind.SECTION_END

    def pull(self):
        if self.boundary:
            return PullKind.SECTION_END
        if self.terminal:
            return PullKind.ROOT_END
        if self._closed:
            raise RuntimeError("Closed execution cannot advance.")
        self.section_final = self.root_final = False
        self.value = None
        try:
            kind = self._step()
            if kind == PullKind.ROOT_END:
                self.terminal = True
            return kind
        except BaseException as error:
            self._close_after_failure(error)
            raise

    def handoff(self):
        if self._closed or not self.boundary:
            raise RuntimeError("No pending section handoff.")
        self.boundary = False
        self.terminal = self.after_boundary_root
        self.after_boundary_root = False
        self.within = 0
        if self.op.kind != "zip":
            for child in self.children.values():
                if child.boundary:
                    child.handoff()
        if self.op.kind == "shuffle":
            self.phase = "fill"

    def _step(self):
        kind = self.op.kind
        if kind == "source":
            if self.source is None:
                self.source = self._own(self._call(self.op.source.open, self.epoch))
            try:
                value = self._pull(self.source)
            except StopIteration:
                return PullKind.ROOT_END
            indexed = getattr(self.op.source, "kind", None) in {"array", "npy"}
            final = indexed and self.op.cardinality.is_finite and self.count + 1 == self.op.cardinality.require_finite()
            return self._deliver(value, root=final)
        if kind == "take":
            n, base, fixed = self.op.controls
            if self.count == n:
                return PullKind.ROOT_END
            child = self._child(epoch=base if fixed else base + self.epoch)
            signal = child.pull()
            if signal == PullKind.ROOT_END:
                raise DatasetExhaustedError(n, self.count)
            if signal == PullKind.SECTION_END:
                return self._section()
            last = self.count + 1 == n
            return self._deliver(child.value, final=child.section_final or (last and self.op.sectioned),
                                 root=last)
        if kind in {"chain", "repeat"}:
            limit = len(self.op.inputs) if kind == "chain" else self.op.controls[0]
            while limit is None or self.index < limit:
                slot = self.index if kind == "chain" else 0
                child = self._child(slot, self.epoch if kind == "chain" else self.epoch + self.index)
                signal = child.pull()
                if signal == PullKind.ROOT_END:
                    nonempty = child.produced
                    self._forget(slot)
                    self.index += 1
                    if kind == "repeat" and limit is None and not nonempty:
                        return PullKind.ROOT_END
                    continue
                if signal == PullKind.SECTION_END:
                    return self._section()
                # A child root is not the root of its Chain/Repeat parent.
                return self._deliver(child.value, final=child.section_final)
            return PullKind.ROOT_END
        if kind == "zip":
            return self._zip()
        if kind == "custom":
            if self.source is None:
                inputs = tuple(_KernelIterator(self._child(i)) for i in range(len(self.op.inputs)))
                self.source = self._own(self._call(self.op.node.open, *inputs))
            try:
                return self._deliver(self._pull(self.source))
            except StopIteration:
                return PullKind.ROOT_END
        child = self._child()
        if kind == "shuffle":
            return self._shuffle(child)
        if kind == "unbatch":
            return self._unbatch(child)
        if kind == "batch":
            return self._batch(child)
        while True:
            signal = child.pull()
            if signal == PullKind.ROOT_END:
                if kind == "epoch" and self.within:
                    return self._section(root=True)
                return signal
            if signal == PullKind.SECTION_END:
                return self._section(root=child.after_boundary_root)
            value = child.value
            if kind == "skip" and self.index < self.op.controls[0]:
                self.index += 1
                if child.section_final:
                    return self._section(root=child.root_final)
                continue
            if kind == "epoch":
                self.within += 1
                return self._deliver(value, final=self.within == self.op.controls[0] or child.root_final,
                                     root=child.root_final)
            if kind == "map":
                value = self._map(value)
            return self._deliver(value, final=child.section_final, root=child.root_final)

    def _map(self, value):
        preserves, finder, examples = self.op.controls
        if self.selected is None:
            from .signature import _runtime_backends_for_spec
            try:
                implementation = self._call(finder, input_spec=self.input_spec)
            except ImplementationSelectionError as error:
                if error.reason != "unknown_traits" or error.unknown_traits != ("backend",):
                    raise
                backends = self._call(_runtime_backends_for_spec, value, self.input_spec)
                if len(backends) != 1:
                    raise error
                implementation = self._call(finder, input_spec=self.input_spec, backend=next(iter(backends)))
            self.selected = self._call(implementation.prepared_invoker)
            self.output_spec = self._call(implementation.infer_output_spec)
        result = self._call(self.selected, value)
        if preserves and self._call(examples, self.input_spec, value) != self._call(examples, self.output_spec, result):
            raise ValueError("Map declared preserves_examples but changed an example count.")
        return result

    def _batch(self, child):
        size, drop, _ = self.op.controls
        self.buffer = []
        boundary = root = False
        while len(self.buffer) < size:
            signal = child.pull()
            if signal != PullKind.DELIVERY:
                boundary, root = signal == PullKind.SECTION_END, signal == PullKind.ROOT_END
                break
            self.buffer.append(child.value)
            if child.section_final or child.root_final:
                boundary, root = child.section_final, child.root_final
                break
        if not self.buffer or (drop and len(self.buffer) < size):
            self.buffer = []
            return self._section(root=child.after_boundary_root) if boundary else PullKind.ROOT_END
        value = self._call(self.selected, self.buffer)
        self.buffer = []
        return self._deliver(value, final=boundary, root=root)

    def _unbatch(self, child):
        while self.split_index == len(self.split_values):
            self.split_values, self.input_batch, self.split_index = (), None, 0
            if self.input_final:
                self.input_final = False
                return self._section(root=self.input_root)
            if self.input_root:
                return PullKind.ROOT_END
            signal = child.pull()
            if signal != PullKind.DELIVERY:
                return self._section(root=child.after_boundary_root) if signal == PullKind.SECTION_END else signal
            self.input_batch = child.value
            self.input_final, self.input_root = child.section_final, child.root_final
            values = self._call(self.selected, self.input_batch)
            if isinstance(values, (tuple, list)):
                self.split_values = values
            else:
                splits = self._own(self._call(iter, values))
                items = []
                while True:
                    try:
                        items.append(self._pull(splits))
                    except StopIteration:
                        break
                self._release(splits)
                self.split_values = items
        value = self.split_values[self.split_index]
        self.split_index += 1
        last = self.split_index == len(self.split_values)
        final, root = last and self.input_final, last and self.input_root
        if last:
            self.input_final = False
        return self._deliver(value, final=final, root=root)

    def _shuffle(self, child):
        if self.rng is None:
            import numpy as np
            self.rng = np.random.default_rng(self.op.controls[1])
        if self.phase in {"fill", "refill"}:
            target = self.op.controls[0] if self.phase == "fill" else len(self.buffer) + 1
            while len(self.buffer) < target:
                signal = child.pull()
                if signal != PullKind.DELIVERY:
                    self.phase = "section" if signal == PullKind.SECTION_END else "root"
                    break
                self.buffer.append(child.value)
                if child.section_final or child.root_final:
                    self.phase = "section" if child.section_final else "root"
                    break
        if not self.buffer:
            return self._section(root=child.after_boundary_root) if self.phase == "section" else PullKind.ROOT_END
        index = int(self.rng.integers(0, len(self.buffer))) if len(self.buffer) > 1 else 0
        value = self.buffer.pop(index)
        final = not self.buffer and self.phase == "section"
        root = not self.buffer and (self.phase == "root" or child.root_final)
        if self.phase not in {"section", "root"}:
            self.phase = "refill"
        return self._deliver(value, final=final, root=root)

    def _zip(self):
        children = [self._child(i) for i in range(len(self.op.inputs))]
        if self.phase == "drain":
            for child in children:
                while not child.boundary:
                    if child.pull() == PullKind.ROOT_END:
                        return PullKind.ROOT_END
            for child in children:
                child.handoff()
            self.phase = "fill"
        values = []
        final = empty = False
        for child in children:
            signal = child.pull()
            if signal == PullKind.ROOT_END:
                return PullKind.ROOT_END
            if signal == PullKind.SECTION_END:
                empty = True
                continue
            values.append(child.value)
            final |= child.section_final
        if empty:
            self.phase = "drain"
            return self._section()
        if final:
            self.phase = "drain"
        return self._deliver(self._call(self.op.tree.build, iter(values)), final=final)

    def close(self):
        self.value = self.input_batch = self.source = None
        self.buffer = []
        self.split_values = ()
        try:
            super().close()
        finally:
            self.children.clear()


class _KernelIterator(Iterator[Any]):
    """Borrow a boundary-free kernel as a custom node's ordinary input."""

    def __init__(self, execution):
        self.execution = execution

    def __iter__(self):
        return self

    def __next__(self):
        signal = self.execution.pull()
        if signal == PullKind.ROOT_END:
            raise StopIteration
        if signal != PullKind.DELIVERY:
            raise RuntimeError("Custom stream received an unsupported section boundary.")
        return self.execution.value


__all__ = ["IteratorPort", "StreamGraphCursor", "StreamNode"]
