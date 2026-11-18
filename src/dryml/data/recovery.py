"""Traversal-owned source strategies and capture-on-request kernel adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Generic, Literal, TypeVar

import numpy as np

from .continuation import ContinuationValue
from .errors import UnsupportedContinuationError, ContinuationFormatError, ContinuationMismatchError

T = TypeVar("T")
ReplayRecoveryPolicy = Literal["disabled", "enabled", "if_needed"]


class RecoverableGenerator(Iterator[T], ABC, Generic[T]):
    """Explicit direct-restoration interface for the iterator actually consumed.

    Subclasses implement next, capture_state, and restore_state; construction is
    ordinary caller-controlled factory work. Recognition never constructs a class.
    Complete state must describe the next yield, including any RNG and buffers.
    Errors from hooks propagate; optional close() is owned once by the traversal.
    This class is neither a DRYML Object nor a suspended Python generator.
    """

    def __iter__(self) -> RecoverableGenerator[T]:
        """Return this actual iterator without acquisition or state extraction."""
        return self

    @abstractmethod
    def __next__(self) -> T:
        """Return the next value, or raise StopIteration on genuine exhaustion."""
        raise StopIteration

    @abstractmethod
    def capture_state(self) -> ContinuationValue:
        """Return complete next-yield state without advancing or retaining frames.

        The traversal validates and detaches the returned supported value tree
        only on requested capture. Unsupported values fail that capture, not live
        iteration. Producer failures propagate with their causal identities.
        """
        raise NotImplementedError

    @abstractmethod
    def restore_state(self, state: ContinuationValue) -> None:
        """Validate/install independent decoded state before the next resumed yield.

        Args:
            state: Supported value returned by this implementation's capture hook.
        Raises:
            ValueError: Incompatible state. A direct failure never enables replay.
        Side Effects:
            Mutates only this newly acquired producer's local next-yield state.
        """
        raise NotImplementedError


def _declared_direct(factory):
    return isinstance(factory, type) and issubclass(factory, RecoverableGenerator)


def _ops(root):
    yield root
    for child in root.inputs:
        yield from _ops(child)


def _executions(root):
    if root is not None:
        yield root
        for child in root.children.values():
            yield from _executions(child)


def _config_available(value):
    from .native import _ConfigTree, _SavedSourceArgument, _SmallArrayConfig
    from dryml.core.node import Node, require_saved
    from dryml.core.object import Object
    if isinstance(value, _SavedSourceArgument):
        return value.selection is not None and value.value.last_state_ref is value.selection
    if isinstance(value, _SmallArrayConfig):
        return True
    if isinstance(value, np.generic):
        return (not value.dtype.hasobject and value.dtype.fields is None and not value.dtype.metadata
                and value.dtype.kind in "biufcSUmM" and value.dtype.itemsize > 0)
    if isinstance(value, (Node, Object)):
        try:
            require_saved(_saved_argument(value))
        except (TypeError, ValueError, UnsupportedContinuationError):
            return False
        return True
    if isinstance(value, _ConfigTree):
        return all(_config_available(v) for v in
                   (tuple(v for _, v in value.items) if value.kind is dict else value.items))
    return value is None or type(value) in (bool, int, float, complex, str, bytes)


def _initial_available(binding):
    return all(_config_available(v) for v in binding.args) and all(_config_available(v) for _, v in binding.kwargs)


def _initial(binding, epoch):
    from .source import _epoch_seed
    if not _initial_available(binding):
        raise UnsupportedContinuationError("Original large/opaque factory argument evidence is unavailable.")
    memo = {}
    args = tuple(_initial_config(v, memo) for v in binding.args)
    kwargs = {key: _initial_config(v, memo) for key, v in binding.kwargs}
    if binding.seed_aware:
        kwargs["seed"] = _epoch_seed(binding.seed, epoch, binding.epoch_seed_version)
    return _initial_tree((args, kwargs))


def _initial_config(value, memo):
    from .native import _ConfigTree, _SavedSourceArgument, _SmallArrayConfig
    if isinstance(value, _SavedSourceArgument):
        return value.selection
    if isinstance(value, _SmallArrayConfig):
        return value.unpack(memo)
    if not isinstance(value, _ConfigTree):
        return value
    if id(value) not in memo:
        memo[id(value)] = ({key: _initial_config(v, memo) for key, v in value.items}
                          if value.kind is dict else value.kind(_initial_config(v, memo) for v in value.items))
    return memo[id(value)]


def _initial_tree(value, memo=None):
    """Encode source initialization separately from buffered values, preserving aliases."""
    from dryml.core.node import Node
    from dryml.core.object import Object
    if value is None or type(value) in (bool, int, float, complex, str, bytes):
        return ["value", value]
    memo = {} if memo is None else memo
    if id(value) in memo:
        return memo[id(value)]
    if isinstance(value, (Node, Object)):
        result = ["saved-node", _node_packet(_saved_argument(value))]
    elif type(value) is dict:
        result = ["dict", [[key, _initial_tree(v, memo)] for key, v in value.items()]]
    elif type(value) in (tuple, list):
        result = [type(value).__name__, [_initial_tree(v, memo) for v in value]]
    else:
        result = ["value", value]
    memo[id(value)] = result
    return result


def _saved_argument(value):
    from dryml.core.node import Node, require_saved
    node = value if isinstance(value, Node) else value.last_state_ref
    try:
        require_saved(node)
    except (TypeError, ValueError) as error:
        raise UnsupportedContinuationError("Source argument requires an explicitly selected retained SavedNode.") from error
    return node


def _restore_initial(capsule, binding):
    """Rebind external metadata only to the caller's supplied exact source values."""
    from dryml.core.node import Node
    from dryml.core.object import Object
    from .continuation import encode_value
    if not _initial_available(binding):
        raise ContinuationMismatchError("Supplied source binding changed after preparation.")
    bindings = {}
    def collect(value):
        if isinstance(value, (Node, Object)):
            bindings[encode_value(_node_packet(_saved_argument(value)))] = value
        elif type(value) is dict:
            for child in value.values():
                collect(child)
        elif type(value) in (list, tuple):
            for child in value:
                collect(child)
    collect((binding.factory_args, binding.factory_kwargs))
    return _initial_value(capsule, bindings=bindings)


def _initial_value(capsule, memo=None, bindings=None):
    """Validate inert capsule tags and reconstruct metadata, never source Objects."""
    from dryml.core.node import require_saved
    from .continuation import encode_value
    memo = {} if memo is None else memo
    if id(capsule) in memo:
        return memo[id(capsule)]
    if type(capsule) is not list or len(capsule) != 2 or type(capsule[0]) is not str:
        raise ContinuationFormatError("Invalid source initialization capsule.")
    tag, value = capsule
    if tag == "saved-node":
        try:
            result = _decode_node(value)
            require_saved(result)
        except (TypeError, ValueError) as error:
            raise ContinuationFormatError("Invalid required saved source selection.") from error
        if bindings is not None:
            key = encode_value(value)
            if key not in bindings:
                raise ContinuationMismatchError("Required supplied source-state binding is unavailable.")
            result = bindings[key]
    elif tag in {"tuple", "list"} and type(value) is list:
        result = (tuple if tag == "tuple" else list)(_initial_value(v, memo, bindings) for v in value)
    elif tag == "dict" and type(value) is list:
        result = {}
        for pair in value:
            if type(pair) is not list or len(pair) != 2 or type(pair[0]) is not str or pair[0] in result:
                raise ContinuationFormatError("Invalid source initialization mapping.")
            result[pair[0]] = _initial_value(pair[1], memo, bindings)
    elif tag == "value" and (value is None or type(value) in (bool, int, float, complex, str, bytes, np.ndarray) or isinstance(value, np.generic)):
        result = value
    else:
        raise ContinuationFormatError("Unknown source initialization tag/value.")
    memo[id(capsule)] = result
    return result


def _check_policy(traversal, policy):
    """Preflight every relevant branch before atomically changing one policy."""
    for op in _ops(traversal._root) if traversal._root is not None else ():
        if op.kind == "source" and op.source.kind == "generator":
            binding = op.source.binding
            active = [e for e in _executions(traversal._execution) if e.op is op and e.source is not None]
            if policy == "if_needed" and (_declared_direct(binding.gen_factory) or
                    active and all(e.source._strategy == "direct" for e in active)):
                continue
            if not _initial_available(binding):
                raise UnsupportedContinuationError("Original factory initialization is unavailable; policy unchanged.")


def _active_replay(traversal):
    return any(e.op.kind == "source" and e.op.source.kind == "generator" and e.source is not None
               and e.source._strategy == "replay" for e in _executions(traversal._execution))


def _symbol(value):
    from dryml.core.symbol import symbol_ref
    from dryml.core._value_json import _value_to_json
    try:
        return _value_to_json(symbol_ref(value), grammar=3)
    except (TypeError, ValueError) as error:
        raise UnsupportedContinuationError("Source symbol has no qualified stable representation.") from error


def _node_packet(node):
    from dryml.core.node import _as_node_view
    from dryml.core.node_prepare import NodePreparation
    from dryml.core.node_codec import encode_node_graph
    packet = encode_node_graph(NodePreparation((_as_node_view(node),)), canonical=True)
    try:
        _inert_node(packet)
    except ContinuationFormatError as error:
        raise UnsupportedContinuationError("Constructor metadata requires an import-bearing value codec.") from error
    return _node_transport(packet)


def _node_transport(value, *, restore=False):
    """Move embedded Node binary literals through the domain payload inventory.

    Node topology is explicit, so incidental sharing of JSON syntax containers
    is deliberately discarded. Core authenticates the original packet after
    its binary spelling has been reinstated, before any metadata is admitted.
    """
    import base64
    if type(value) is dict:
        binary = ("bytes" if value.get("tag") in {"array", "numpy-scalar"} or value.get("type") == "numpy_scalar"
                  else "value" if value.get("type") == "bytes" else None)
        result = {}
        for key in sorted(value):
            child = value[key]
            if key == binary:
                if restore:
                    if type(child) is not bytes:
                        raise ContinuationFormatError("Invalid Node binary transport.")
                    result[key] = base64.b64encode(child).decode("ascii")
                else:
                    result[key] = base64.b64decode(child, validate=True)
            else:
                result[key] = _node_transport(child, restore=restore)
        return result
    if type(value) in (list, tuple):
        return [_node_transport(child, restore=restore) for child in value]
    return value


def _inert_node(value):
    """Keep generic Node value-type import hooks outside inert Data metadata IO."""
    if type(value) is dict:
        if value.get("type") in {"enum", "core_type", "numpy_dtype"}:
            raise ContinuationFormatError("Import-bearing Node value codecs are not qualified for inert DatasetState IO.")
        for child in value.values():
            _inert_node(child)
    elif type(value) in (tuple, list):
        for child in value:
            _inert_node(child)


def _decode_node(packet):
    from dryml.core.node_codec import decode_node_graph
    _inert_node(packet)
    return decode_node_graph(_node_transport(packet, restore=True))


def _metadata_value(value, tags):
    """Check allowed semantic tags before Core can dispatch any type resolver."""
    from dryml.core._value_json import _value_from_json
    def check(record):
        if type(record) is not dict or record.get("type") not in tags:
            raise ContinuationFormatError("Invalid semantic continuation metadata tag.")
        if record["type"] in {"tuple", "list"}:
            for child in record.get("items", ()):
                check(child)
        elif record["type"] == "dict":
            for pair in record.get("items", ()):
                if type(pair) is not list or len(pair) != 2:
                    raise ContinuationFormatError("Invalid semantic metadata mapping.")
                check(pair[0])
                check(pair[1])
        elif record["type"] == "source_spec":
            imports = record.get("imports")
            if type(imports) is not dict or any(type(reference) is not dict or reference.get("type") != "import_ref" for reference in imports.values()):
                raise ContinuationFormatError("Invalid inert source imports.")
    check(value)
    return _value_from_json(value, grammar=3)


def _spec_value(value):
    return _metadata_value(value, {"null", "tensor_spec", "tuple", "list", "dict", "str", "int"})


def _symbol_value(value):
    return _metadata_value(value, {"import_ref", "source_spec"})


def _controls_value(row):
    controls = _metadata_value(row["controls"], {"tuple", "int", "bool", "null"})
    kind = row["kind"]
    lengths = dict(map=1, batch=3, take=3, skip=1, repeat=1, shuffle=2, epoch=1)
    if type(controls) is not tuple or len(controls) != lengths.get(kind, 0):
        raise ContinuationFormatError("Invalid frozen operator controls.")
    exact_count = lambda n, minimum=0: type(n) is int and n >= minimum
    valid = True
    if kind == "map":
        valid = type(controls[0]) is bool
    elif kind == "batch":
        valid = exact_count(controls[0], 1) and all(type(v) is bool for v in controls[1:])
    elif kind == "take":
        valid = exact_count(controls[0]) and exact_count(controls[1]) and type(controls[2]) is bool
    elif kind in {"skip", "epoch"}:
        valid = exact_count(controls[0], int(kind == "epoch"))
    elif kind == "repeat":
        valid = controls[0] is None or exact_count(controls[0])
    elif kind == "shuffle":
        valid = exact_count(controls[0], 1) and (controls[1] is None or type(controls[1]) is int)
    if not valid:
        raise ContinuationFormatError("Invalid frozen operator control values.")
    return controls


def _validate_tree(tree):
    if type(tree) is not list or len(tree) != 2 or tree[0] not in {"leaf", "list", "tuple", "dict"} or type(tree[1]) is not list:
        raise ContinuationFormatError("Invalid Zip tree evidence.")
    kind, children = tree
    if kind == "leaf":
        if children:
            raise ContinuationFormatError("Zip leaf has children.")
        return 1
    total, keys = 0, set()
    for child in children:
        if kind == "dict":
            if type(child) is not list or len(child) != 2 or type(child[0]) not in (str, int) or child[0] in keys:
                raise ContinuationFormatError("Invalid Zip mapping evidence.")
            keys.add(child[0])
            child = child[1]
        total += _validate_tree(child)
    return total


def _selection(selected):
    """Export retained selection facts, never the executable invoker/closure."""
    if selected is None:
        return None
    from dryml.methods.implementation import PreparedMethodInvoker
    from dryml.methods.signature import spec_from_node
    from dryml.core._value_json import _value_to_json
    if isinstance(selected, PreparedMethodInvoker):
        return dict(kind="method", name=selected.name,
                    receiver=_symbol(selected.adapter.receiver_type),
                    backend=None if selected.traits.backend is None else selected.traits.backend.value,
                    batch_mode=None if selected.traits.batch_mode is None else selected.traits.batch_mode.value,
                    inputs=[_value_to_json(spec_from_node(node), grammar=3) for node in selected.input_specs],
                    output=_value_to_json(None if selected.output_spec is None else spec_from_node(selected.output_spec), grammar=3),
                    children=[_selection(child) for child in selected.children])
    import inspect
    import textwrap
    try:
        source = textwrap.dedent(inspect.getsource(selected)).strip()
    except (OSError, TypeError) as error:
        raise UnsupportedContinuationError("Selected buffer adapter has no stable implementation evidence.") from error
    return dict(kind="buffer-adapter", module=selected.__module__, name=selected.__qualname__, source=source, version=1)


def _validate_selection(value):
    if value is None:
        return
    from .state import _fields
    from dryml.core.symbol import ImportRef, SourceSpec
    if type(value) is not dict or value.get("kind") not in {"method", "buffer-adapter"}:
        raise ContinuationFormatError("Invalid selected implementation tag.")
    if value["kind"] == "buffer-adapter":
        _fields(value, ("kind", "module", "name", "source", "version"), "buffer selection")
        if any(type(value[n]) is not str or not value[n] for n in ("module", "name", "source")) or type(value["version"]) is not int or value["version"] != 1:
            raise ContinuationFormatError("Invalid buffer implementation evidence.")
        return
    _fields(value, ("kind", "name", "receiver", "backend", "batch_mode", "inputs", "output", "children"), "Method selection")
    if type(value["name"]) is not str or not value["name"] or not isinstance(_symbol_value(value["receiver"]), (ImportRef, SourceSpec)):
        raise ContinuationFormatError("Invalid Method implementation evidence.")
    if value["backend"] not in {None, "numpy", "torch", "tf", "jax"} or value["batch_mode"] not in {None, "element", "batched"}:
        raise ContinuationFormatError("Invalid selected Method traits.")
    if type(value["inputs"]) is not list or type(value["children"]) is not list:
        raise ContinuationFormatError("Invalid selected Method structure.")
    for spec in (*value["inputs"], value["output"]):
        _spec_value(spec)
    for child in value["children"]:
        _validate_selection(child)


def _evidence(prepared, root=None):
    """Describe frozen execution, never a fresh CDef from edited controls."""
    from dryml.core._value_json import _value_to_json
    from .native import _unpack_config
    if prepared.execution_level != "stream":
        raise UnsupportedContinuationError("Unqualified eager streams have no exact continuation.")
    if prepared._continuation_missing_states:
        raise UnsupportedContinuationError("Stateful Method dependencies require an exact saved source-state selection.")
    root = prepared.graph._stream_plan._prepared.root if root is None else root
    aliases, rows, snapshots = {}, [], {}
    for op in _ops(root):
        if op.kind == "custom" or op.kind == "map" and op.selected is None:
            raise UnsupportedContinuationError("Continuation requires a qualified frozen operator selection.")
        alias = aliases.setdefault(id(op.dataset), len(aliases))
        recipe, method, state = prepared._continuation_nodes[op.occurrence]
        controls = op.controls[:1] if op.kind == "map" else op.controls
        # Batch's third control is a bool; Map's later controls are live helpers.
        control = _value_to_json(controls, grammar=3)
        row = dict(path=list(op.occurrence), kind=op.kind, alias=alias, controls=control,
                   spec=_value_to_json(op.spec, grammar=3),
                   cardinality=_value_to_json(op.cardinality, grammar=3),
                   sectioned=op.sectioned, sections=op.sections, section_bound=op.section_bound,
                   recipe=None if op.occurrence or recipe is None else _node_packet(recipe),
                   method=None if method is None else _node_packet(method),
                   state=None if state is None else _node_packet(state),
                   tree=None if op.tree is None else _tree_evidence(op.tree), source=None,
                   selection=_selection(op.selected))
        if op.kind == "source":
            source = op.source
            if source.kind == "array":
                tree, length = source.binding
                snapshots[str(op.occurrence)] = _unpack_config(tree)
                row["source"] = dict(kind="array", length=length)
            elif source.kind == "generator":
                binding = source.binding
                if not _initial_available(binding):
                    raise UnsupportedContinuationError("Factory configuration lacks retained original evidence.")
                row["source"] = dict(kind="generator", factory=_symbol(binding.gen_factory),
                                      initial=_initial(binding, 0), seed=binding.seed,
                                      seed_aware=binding.seed_aware, seed_version=binding.epoch_seed_version,
                                      declared_direct=_declared_direct(binding.gen_factory))
            else:
                raise UnsupportedContinuationError(f"Unsupported source continuation: {source.kind}.")
        rows.append(row)
    return rows, snapshots


def _tree_evidence(tree):
    return [tree.kind, [[key, _tree_evidence(child)] for key, child in tree.children]
            if tree.kind == "dict" else [_tree_evidence(child) for child in tree.children]]


_COUNTERS = ("epoch", "count", "within", "index", "split_index")
_FLAGS = ("section_final", "root_final", "boundary", "terminal", "after_boundary_root",
          "produced", "input_final", "input_root")
_VALUES = ("buffer", "split_values", "input_batch")


def _capture_execution(execution, policy):
    if execution is None:
        return None
    op = execution.op
    record = {name: getattr(execution, name) for name in _COUNTERS + _FLAGS + _VALUES}
    record.update(path=list(op.occurrence), kind=op.kind, phase=execution.phase,
                  rng=None if execution.rng is None else execution.rng.bit_generator.state,
                  children=[_capture_execution(child, policy) for _, child in sorted(execution.children.items())],
                  source=None)
    if op.kind == "source" and execution.source is not None:
        cursor = execution.source
        if op.source.kind == "array":
            record["source"] = dict(strategy="indexed", fetched=cursor.position, eof=cursor.observed_eof)
        elif op.source.kind == "generator":
            strategy = cursor._strategy
            # A restored cursor can itself be recaptured before normal acquisition.
            if cursor._restore is not None:
                record["source"] = cursor._restore
            elif cursor._values is None:
                record["source"] = None
            else:
                if strategy != "direct" and policy == "disabled":
                    raise UnsupportedContinuationError("Ordinary generator replay recovery is disabled.")
                record["source"] = dict(strategy=strategy, fetched=cursor.position,
                    eof=cursor.observed_eof, initial=_initial(cursor._initial_binding, execution.epoch),
                    seed=cursor._effective_seed, pass_epoch=execution.epoch,
                    examples=cursor._observed_examples, validate_examples=cursor._validate_observed,
                    producer=_symbol(type(cursor._values)) if strategy == "direct" else None,
                    state=cursor._call(cursor._values.capture_state) if strategy == "direct" and not cursor.observed_eof else None)
    return record


def _validate_execution(record, rows, parent=None):
    if record is None:
        return
    expected = set(_COUNTERS + _FLAGS + _VALUES) | {"path", "kind", "phase", "rng", "children", "source"}
    if type(record) is not dict or set(record) != expected:
        raise ContinuationFormatError("Invalid occurrence state fields.")
    path = record["path"]
    if type(path) is not list or any(type(n) is not int or n < 0 for n in path):
        raise ContinuationFormatError("Invalid occurrence path.")
    key = tuple(path)
    if key not in rows or record["kind"] != rows[key]["kind"] or parent is not None and key[:-1] != parent:
        raise ContinuationFormatError("Invalid occurrence topology/tag.")
    if any(type(record[n]) is not int or record[n] < 0 for n in _COUNTERS):
        raise ContinuationFormatError("Invalid occurrence counters.")
    if any(type(record[n]) is not bool for n in _FLAGS):
        raise ContinuationFormatError("Invalid occurrence flags.")
    if record["phase"] not in {"fill", "refill", "section", "root", "drain"}:
        raise ContinuationFormatError("Invalid occurrence phase.")
    if type(record["buffer"]) is not list or type(record["split_values"]) not in (tuple, list) or record["split_index"] > len(record["split_values"]):
        raise ContinuationFormatError("Invalid buffered frontier.")
    kind = record["kind"]
    controls = _controls_value(rows[key])
    if kind == "take" and record["count"] > controls[0] or kind == "skip" and record["index"] > controls[0] or kind == "epoch" and record["within"] > controls[0]:
        raise ContinuationFormatError("Occurrence exceeds frozen control bounds.")
    if kind == "shuffle" and len(record["buffer"]) > controls[0]:
        raise ContinuationFormatError("Shuffle buffer exceeds its frozen bound.")
    if kind != "shuffle" and record["rng"] is not None or kind not in {"shuffle", "batch"} and record["buffer"]:
        raise ContinuationFormatError("Unexpected operator buffer/RNG state.")
    if kind != "unbatch" and (record["split_values"] or record["split_index"] or record["input_batch"] is not None):
        raise ContinuationFormatError("Unexpected split frontier.")
    if kind == "source" and record["count"] > 0 and record["source"] is None:
        raise ContinuationFormatError("Missing active source frontier.")
    if record["rng"] is not None:
        try:
            rng = np.random.default_rng()
            rng.bit_generator.state = record["rng"]
        except (ValueError, TypeError, KeyError, OverflowError) as error:
            raise ContinuationFormatError("Invalid RNG state.") from error
    source = record["source"]
    if source is not None:
        if record["kind"] != "source" or type(source) is not dict:
            raise ContinuationFormatError("Source state on a non-source occurrence.")
        kind = rows[key]["source"]["kind"]
        fields = {"strategy", "fetched", "eof"}
        if kind == "generator":
            fields |= {"initial", "seed", "pass_epoch", "examples", "validate_examples", "producer", "state"}
        if set(source) != fields or type(source["fetched"]) is not int or source["fetched"] < 0 or type(source["eof"]) is not bool:
            raise ContinuationFormatError("Invalid source state fields/counters.")
        if source["fetched"] != record["count"]:
            raise ContinuationFormatError("Source fetched count disagrees with occurrence.")
        if kind == "array":
            if source["strategy"] != "indexed" or source["fetched"] > rows[key]["source"]["length"]:
                raise ContinuationFormatError("Invalid indexed position.")
        elif source["strategy"] not in {"direct", "replay"} or type(source["examples"]) is not int or source["examples"] < 0 or type(source["validate_examples"]) is not bool:
            raise ContinuationFormatError("Invalid generator strategy/state.")
        else:
            initial = _initial_value(source["initial"])
            if (type(source["pass_epoch"]) is not int or source["pass_epoch"] != record["epoch"]
                    or type(initial) is not tuple or len(initial) != 2 or type(initial[0]) is not tuple or type(initial[1]) is not dict):
                raise ContinuationFormatError("Invalid generator initialization.")
            evidence = rows[key]["source"]
            from .source import _epoch_seed
            seed = _epoch_seed(evidence["seed"], record["epoch"], evidence["seed_version"]) if evidence["seed_aware"] else None
            if source["seed"] != seed or evidence["seed_aware"] and initial[1].get("seed") != seed:
                raise ContinuationFormatError("Saved effective seed disagrees with source pass.")
            from .continuation import encode_value
            declared_args, declared_kwargs = _initial_value(evidence["initial"])
            if evidence["seed_aware"]:
                declared_kwargs["seed"] = seed
            if encode_value(source["initial"]) != encode_value(_initial_tree((declared_args, declared_kwargs))):
                raise ContinuationFormatError("Source initialization differs from frozen evidence.")
            if source["strategy"] == "direct":
                from dryml.core.symbol import ImportRef, SourceSpec
                if not isinstance(_symbol_value(source["producer"]), (ImportRef, SourceSpec)):
                    raise ContinuationFormatError("Invalid direct producer evidence.")
            elif source["producer"] is not None or source["state"] is not None:
                raise ContinuationFormatError("Replay occurrence carries direct state.")
    if type(record["children"]) is not list:
        raise ContinuationFormatError("Invalid occurrence children.")
    seen = set()
    for child in record["children"]:
        _validate_execution(child, rows, key)
        child_key = tuple(child["path"])
        if child_key in seen:
            raise ContinuationFormatError("Duplicate occurrence record.")
        seen.add(child_key)
    if record["count"] and record["kind"] not in {"source", "chain", "repeat"} and not seen:
        raise ContinuationFormatError("Missing active child frontier.")


def _restore_execution(op, record, snapshots):
    from dryml.methods.stream import _SectionExecution
    from .source import _IndexedDatasetCursor, _tree_index
    from functools import partial
    execution = _SectionExecution(op, record["epoch"])
    for name in _COUNTERS + _FLAGS + _VALUES + ("phase",):
        setattr(execution, name, record[name])
    if record["rng"] is not None:
        execution.rng = np.random.default_rng()
        execution.rng.bit_generator.state = record["rng"]
    try:
        for child in record["children"]:
            index = child["path"][-1]
            execution.children[index] = execution._own(_restore_execution(op.inputs[index], child, snapshots))
        saved = record["source"]
        if saved is not None:
            if saved["strategy"] == "indexed":
                cursor = _IndexedDatasetCursor(op.source.binding[1], partial(_tree_index, snapshots[str(op.occurrence)]))
                cursor._position = saved["fetched"]
            else:
                cursor = op.source.open(execution.epoch)
                cursor._restore = saved
                cursor._strategy = saved["strategy"]
                cursor._position = saved["fetched"]
            cursor._closed = cursor._observed_eof = saved["eof"]
            execution.source = execution._own(cursor)
        return execution
    except BaseException as error:
        execution._close_after_failure(error)
        raise


def _snapshot_root(op, snapshots):
    """Bind all restored physical occurrences, including future ones, to owned bytes."""
    from dataclasses import replace
    from functools import partial
    from .native import _SourceBinding
    from .source import _IndexedDatasetCursor, _tree_index
    children = tuple(_snapshot_root(child, snapshots) for child in op.inputs)
    source = op.source
    if op.kind == "source" and source.kind == "array":
        arrays, length = snapshots[str(op.occurrence)], source.binding[1]
        source = _SourceBinding("array", (arrays, length),
                                lambda epoch: _IndexedDatasetCursor(length, partial(_tree_index, arrays)))
    return replace(op, inputs=children, source=source)
