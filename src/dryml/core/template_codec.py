"""Closed, versioned persistence for inert template recipes.

The codec is deliberately data-only: decoding recreates symbolic references and
frozen recipe topology but never resolves a target, constructs an Object, or
invokes a factory/provider.
"""

from __future__ import annotations

import base64
import json
import math
from typing import Any, Mapping

from .errors import ParameterizationError, ParameterizationLimitError

TEMPLATE_CODEC_SCHEMA = "dryml-template"
TEMPLATE_CODEC_VERSION = 1
_MAX_BYTES = 16 * 1024 * 1024
_MAX_DEPTH = 128
_MAX_NODES = 65_536
_MAX_ENTRIES = 4_096
_MAX_INT_BITS = 1_024


class TemplateCodecError(ParameterizationError):
    """Raised when a portable template payload is malformed or unsupported.

    The error is a :class:`ParameterizationError` so public callers retain one bounded
    failure family for authoring and persistence errors.
    """


def selector_to_data(selector: Any) -> dict[str, object]:
    """Encode an exact GeneratorSelector with built-in immutable domains only.

    Args:
        selector: Exact selector produced by ``Generator``.

    Returns:
        A canonical v1 data mapping.

    Raises:
        TemplateCodecError: If the selector uses a custom runtime provider.
    """

    from .domains import UniformFromSet, UniformIntRange
    from .generator import GeneratorSelector

    if not isinstance(selector, GeneratorSelector):
        raise TypeError("selector_to_data requires a GeneratorSelector")
    generator = selector._generator
    domains = []
    encoder = _Encoder()
    for name, domain in generator.distributions.items():
        if isinstance(domain, UniformIntRange):
            domains.append({"name": name, "kind": "int-range", "lo": domain.lo, "hi": domain.hi})
        elif isinstance(domain, UniformFromSet):
            domains.append({"name": name, "kind": "set", "values": [encoder.value(value) for value in domain.values]})
        else:
            raise TemplateCodecError("template selector has a nonportable domain")
    root = encoder.value(generator.definition)
    return encoder.finish(
        "template-selector",
        root,
        extra={"domains": domains, "traverse_refs": generator._traverse_refs,
               "max_assignments": selector._max_assignments},
    )


def selector_from_data(data: Mapping[str, object]) -> Any:
    """Decode a built-in-domain GeneratorSelector without runtime providers.

    Args:
        data: Canonical mapping produced by :func:`selector_to_data`.

    Returns:
        A GeneratorSelector reconstructed from inert Definition and domain data.

    Raises:
        TemplateCodecError: If policies, domains, or graph data are invalid.
    """

    from .domains import UniformFromSet, UniformIntRange
    from .definition import Definition
    from .generator import Generator, GeneratorSelector

    kind, root, extra, state = _decode(data, extra={"domains", "traverse_refs", "max_assignments"})
    if kind != "template-selector":
        raise TemplateCodecError("template selector payload kind is invalid")
    if type(extra["traverse_refs"]) is not bool or type(extra["max_assignments"]) is not int:
        raise TemplateCodecError("template selector policy is invalid")
    domains: dict[str, object] = {}
    if not isinstance(extra["domains"], list) or len(extra["domains"]) > _MAX_ENTRIES:
        raise TemplateCodecError("template selector domains are invalid")
    for record in extra["domains"]:
        if not isinstance(record, dict) or not isinstance(record.get("name"), str) or record["name"] in domains:
            raise TemplateCodecError("template selector domain entry is invalid")
        if record.get("kind") == "int-range" and set(record) == {"name", "kind", "lo", "hi"}:
            domains[record["name"]] = UniformIntRange(record["lo"], record["hi"])
        elif record.get("kind") == "set" and set(record) == {"name", "kind", "values"} and isinstance(record["values"], list):
            # Values were decoded as part of the one shared graph.  Re-read the
            # canonical payload through a small indexed lookup kept by _decode.
            values = [_decode_value(value, state, 0) for value in record["values"]]
            domains[record["name"]] = UniformFromSet(values)
        else:
            raise TemplateCodecError("template selector domain entry is invalid")
    try:
        if not isinstance(root, Definition):
            raise TemplateCodecError("template selector root must be a Definition")
        generator = Generator(root, domains, traverse_refs=extra["traverse_refs"])
        selector = GeneratorSelector(generator, max_assignments=extra["max_assignments"])
    except ParameterizationError:
        raise
    except Exception as error:
        raise TemplateCodecError("template selector payload is invalid") from error
    if selector_to_data(selector) != dict(data):
        raise TemplateCodecError("template selector payload is not canonical")
    return selector


class _Encoder:
    def __init__(self) -> None:
        self.nodes: list[dict[str, object]] = []
        self.labels: dict[int, str] = {}
        self.active: set[int] = set()
        self.entries = 0

    def consume_entries(self, count: int) -> None:
        """Apply one cumulative container-entry charge to this payload."""

        self.entries += count
        if self.entries > _MAX_ENTRIES:
            raise ParameterizationLimitError("template aggregate entry limit exceeded")

    def finish(self, kind: str, root: object, *, extra: dict[str, object] | None = None) -> dict[str, object]:
        result: dict[str, object] = {"schema": TEMPLATE_CODEC_SCHEMA, "version": TEMPLATE_CODEC_VERSION,
                                     "kind": kind, "root": root, "nodes": self.nodes}
        if extra:
            result.update(extra)
        _check_bytes(result)
        return result

    def value(self, value: object, depth: int = 0) -> dict[str, object]:
        if depth > _MAX_DEPTH:
            raise ParameterizationLimitError("template codec depth limit exceeded")
        if value is None:
            return {"tag": "none"}
        if type(value) is bool:
            return {"tag": "bool", "value": value}
        if type(value) is int:
            if value.bit_length() > _MAX_INT_BITS:
                raise ParameterizationLimitError("template integer bit-length limit exceeded")
            return {"tag": "int", "value": str(value)}
        if type(value) is float:
            if not math.isfinite(value):
                raise TemplateCodecError("template floats must be finite")
            return {"tag": "float", "value": value.hex()}
        if isinstance(value, str):
            return {"tag": "str", "value": value}
        if isinstance(value, bytes):
            return {"tag": "bytes", "value": base64.b64encode(value).decode("ascii")}
        from .symbol import ImportRef, SourceSpec, maybe_symbol_ref
        if isinstance(value, type):
            value = maybe_symbol_ref(value)
            if value is None:
                raise TemplateCodecError("template type is not portable")
        if isinstance(value, ImportRef):
            return {"tag": "import", "module": value.module, "qualname": value.qualname}
        if isinstance(value, SourceSpec):
            self.consume_entries(len(value.imports))
            return {"tag": "source", "kind": value.kind, "source": value.source, "name": value.name,
                    "imports": [[name, self.value(ref, depth + 1)] for name, ref in value.imports.items()]}
        return self.node(value, depth)

    def node(self, value: object, depth: int) -> dict[str, object]:
        marker = id(value)
        if marker in self.active:
            raise TemplateCodecError("template graph cycle is unsupported")
        if marker in self.labels:
            return {"tag": "ref", "label": self.labels[marker]}
        if len(self.nodes) >= _MAX_NODES:
            raise ParameterizationLimitError("template codec node limit exceeded")
        # Allocate before recursively encoding children so labels stay unique
        # for a depth-first graph with aliases.
        label = f"n{len(self.labels)}"
        self.labels[marker] = label
        self.active.add(marker)
        try:
            payload = self._node_value(value, depth + 1)
            self.nodes.append({"label": label, "value": payload})
            return {"tag": "ref", "label": label}
        finally:
            self.active.remove(marker)

    def _node_value(self, value: object, depth: int) -> dict[str, object]:
        from .definition import ConcreteDefinition, Definition
        from .factory import FactorySpec, _validated_factory_target
        from .freeze import FrozenDict, FrozenList, FrozenSet, FrozenTuple
        from .links import DefLink
        from .quoted import QuotedDef, SelectorSpec
        from .reference_values import ObjectRef, StateRef
        from .selector import Selector
        from .template import Expr, Par, _BinaryExpr, _RepeatExpr, _validate_number
        if isinstance(value, Par):
            path = value.path.to_data()
            self.consume_entries(len(path["segments"]))
            return {"tag": "par", "name": value.name, "path": path}
        if isinstance(value, _BinaryExpr):
            if (
                value.operation not in {"mul", "truediv", "floordiv"}
                or not all(
                    isinstance(item, Expr) or _validate_number(item)
                    for item in (value.left, value.right)
                )
            ):
                raise TemplateCodecError("template binary expression is invalid")
            return {"tag": "binary", "op": value.operation, "left": self.value(value.left, depth), "right": self.value(value.right, depth)}
        if isinstance(value, _RepeatExpr):
            return {"tag": "repeat", "group": self.value(value.group, depth), "count": self.value(value.count, depth), "shared": value.shared}
        if isinstance(value, DefLink):
            if not value.is_finalized:
                raise TemplateCodecError("unresolved link assertion is not portable")
            return {"tag": "link", "edge": value.kind.value, "target": self.value(value.target, depth)}
        if isinstance(value, (Definition, ConcreteDefinition)):
            return {"tag": "cdef" if isinstance(value, ConcreteDefinition) else "definition",
                    "cls": self.value(value.cls, depth),
                    "args": None if isinstance(value, ConcreteDefinition) or value.args is None else self.value(value.args, depth),
                    "kwargs": self.value(value.parameters if isinstance(value, ConcreteDefinition) else value.kwargs, depth),
                    "stateful_role": value._stateful_role if isinstance(value, ConcreteDefinition) else None}
        if isinstance(value, FactorySpec):
            try:
                target = _validated_factory_target(value.target)
            except TypeError:
                raise TemplateCodecError("template factory target is invalid") from None
            return {"tag": "factory", "target": self.value(target, depth), "args": self.value(value.args, depth), "kwargs": self.value(value.kwargs, depth)}
        if isinstance(value, (FrozenDict, dict)):
            if len(value) > _MAX_ENTRIES or any(type(key) not in {str, int} for key in value):
                raise TemplateCodecError("template map is unsupported")
            self.consume_entries(len(value))
            return {"tag": "map", "items": [[self.value(key, depth), self.value(item, depth)] for key, item in value.items()]}
        if isinstance(value, (FrozenList, list, FrozenTuple, tuple)):
            if len(value) > _MAX_ENTRIES:
                raise ParameterizationLimitError("template container entry limit exceeded")
            self.consume_entries(len(value))
            return {"tag": "list" if isinstance(value, (FrozenList, list)) else "tuple", "items": [self.value(item, depth) for item in value]}
        if isinstance(value, (FrozenSet, set, frozenset)):
            if len(value) > _MAX_ENTRIES:
                raise ParameterizationLimitError("template container entry limit exceeded")
            self.consume_entries(len(value))
            if any(not _portable_set_member(item) for item in value):
                raise TemplateCodecError("template set members must be literal portable values")
            items = [self.value(item, depth) for item in value]
            items.sort(key=_canonical_bytes)
            if len({_canonical_bytes(item) for item in items}) != len(items):
                raise TemplateCodecError("template set has duplicate canonical members")
            return {"tag": "set", "items": items}
        if isinstance(value, QuotedDef):
            return {"tag": "quoted", "value": self.value(value.value, depth)}
        if isinstance(value, SelectorSpec):
            return {"tag": "selector-spec", "selector": self.value(value.selector, depth)}
        if isinstance(value, Selector):
            return {"tag": "selector", "root": self.value(value.root, depth), "strict": value.strict, "cls_policy": value.cls_policy}
        if isinstance(value, ObjectRef):
            objects = value.to_data()["objects"]
            self.consume_entries(len(objects))
            return {"tag": "object-ref", "definition": self.value(value.definition, depth), "objects": objects}
        if isinstance(value, StateRef):
            states = value.to_data()["states"]
            self.consume_entries(len(states))
            return {"tag": "state-ref", "object": self.value(value.object, depth), "states": states}
        raise TemplateCodecError("template value type is not portable")


def _encode(kind: str, root: object) -> dict[str, object]:
    encoder = _Encoder()
    return encoder.finish(kind, encoder.value(root))


class _DecodeState:
    def __init__(self, records: dict[str, Mapping[str, object]]) -> None:
        self.records, self.built, self.active = records, {}, set()
        self.entries = 0

    def consume_entries(self, count: int) -> None:
        """Apply one cumulative container-entry charge to this payload."""

        self.entries += count
        if self.entries > _MAX_ENTRIES:
            raise ParameterizationLimitError("template aggregate entry limit exceeded")


def _decode(data: Mapping[str, object], *, extra: set[str] | None = None):
    if not isinstance(data, Mapping):
        raise TemplateCodecError("template payload must be a mapping")
    expected = {"schema", "version", "kind", "root", "nodes"} | (extra or set())
    if set(data) != expected or data.get("schema") != TEMPLATE_CODEC_SCHEMA or data.get("version") != TEMPLATE_CODEC_VERSION:
        raise TemplateCodecError("template payload schema or version is invalid")
    if not isinstance(data["kind"], str) or not isinstance(data["nodes"], list) or len(data["nodes"]) > _MAX_NODES:
        raise TemplateCodecError("template payload header is invalid")
    records: dict[str, Mapping[str, object]] = {}
    for record in data["nodes"]:
        if not isinstance(record, Mapping) or set(record) != {"label", "value"} or not isinstance(record["label"], str):
            raise TemplateCodecError("template node record is invalid")
        if record["label"] in records:
            raise TemplateCodecError("template node label is duplicated")
        records[record["label"]] = record
    state = _DecodeState(records)
    root = _decode_value(data["root"], state, 0)
    if extra is not None and "domains" in extra and isinstance(data["domains"], list):
        # Domain choice values share the selector graph labels but are not
        # necessarily reachable from its template root.
        for record in data["domains"]:
            if isinstance(record, Mapping) and isinstance(record.get("values"), list):
                for value in record["values"]:
                    _decode_value(value, state, 0)
    if len(state.built) != len(records):
        raise TemplateCodecError("template graph contains unreachable labels")
    result = (data["kind"], root) if extra is None else (data["kind"], root, {key: data[key] for key in extra}, state)
    # Re-encode to reject alternate ordering, dangling shapes, and malformed
    # aliases before exposing the decoded graph.
    if extra is None:
        rebuilt = _encode(data["kind"], root)
    else:
        # Domain validation happens in selector_from_data.  Its values share this
        # graph's labels, so canonical comparison is completed by that public
        # reconstruction rather than by a second isolated decoder.
        rebuilt = None
    if rebuilt is not None and rebuilt != dict(data):
        raise TemplateCodecError("template payload is not canonical")
    _check_bytes(dict(data))
    return result


def _decode_value(data: object, state: _DecodeState, depth: int) -> object:
    if depth > _MAX_DEPTH or not isinstance(data, Mapping) or not isinstance(data.get("tag"), str):
        raise TemplateCodecError("template value is invalid")
    tag = data["tag"]
    if tag == "none" and set(data) == {"tag"}: return None
    if tag == "bool" and set(data) == {"tag", "value"} and type(data["value"]) is bool: return data["value"]
    if tag == "int" and set(data) == {"tag", "value"} and isinstance(data["value"], str):
        try: value = int(data["value"])
        except ValueError: raise TemplateCodecError("template int is invalid") from None
        if str(value) != data["value"] or value.bit_length() > _MAX_INT_BITS: raise TemplateCodecError("template int is invalid")
        return value
    if tag == "float" and set(data) == {"tag", "value"} and isinstance(data["value"], str):
        try: value = float.fromhex(data["value"])
        except (ValueError, OverflowError): raise TemplateCodecError("template float is invalid") from None
        if not math.isfinite(value) or value.hex() != data["value"]: raise TemplateCodecError("template float is invalid")
        return value
    if tag == "str" and set(data) == {"tag", "value"} and isinstance(data["value"], str): return data["value"]
    if tag == "bytes" and set(data) == {"tag", "value"} and isinstance(data["value"], str):
        try: return base64.b64decode(data["value"].encode("ascii"), validate=True)
        except Exception: raise TemplateCodecError("template bytes are invalid") from None
    if tag == "import" and set(data) == {"tag", "module", "qualname"}:
        from .symbol import ImportRef
        try: return ImportRef(data["module"], data["qualname"])
        except Exception: raise TemplateCodecError("template import is invalid") from None
    if tag == "source" and set(data) == {"tag", "kind", "source", "name", "imports"} and isinstance(data["imports"], list):
        from .symbol import SourceSpec
        state.consume_entries(len(data["imports"]))
        try: return SourceSpec(data["kind"], data["source"], data["name"], {name: _decode_value(ref, state, depth + 1) for name, ref in data["imports"]})
        except Exception: raise TemplateCodecError("template source is invalid") from None
    if tag == "ref" and set(data) == {"tag", "label"} and isinstance(data["label"], str):
        label = data["label"]
        if label in state.built: return state.built[label]
        if label not in state.records: raise TemplateCodecError("template reference label is dangling")
        if label in state.active: raise TemplateCodecError("template graph cycle is unsupported")
        state.active.add(label)
        try:
            value = _decode_node(state.records[label]["value"], state, depth + 1)
            state.built[label] = value
            return value
        finally: state.active.remove(label)
    raise TemplateCodecError("template value tag is invalid")


def _decode_node(data: object, state: _DecodeState, depth: int) -> object:
    from .bound_args import BoundArguments
    from .cdef_graph import EdgeKind
    from .definition import ConcreteDefinition, Definition
    from .factory import FactorySpec, _validated_factory_target
    from .freeze import FrozenDict, FrozenList, FrozenSet, FrozenTuple
    from .links import DefLink
    from .quoted import QuotedDef, SelectorSpec
    from .reference_values import ObjectId, ObjectRef, StateRef
    from .selector import Selector
    from .template import Expr, Par, _BinaryExpr, _RepeatExpr, _validate_number
    from .utils.graph.path import GraphPath

    if not isinstance(data, Mapping) or not isinstance(data.get("tag"), str): raise TemplateCodecError("template node is invalid")
    tag = data["tag"]
    value = lambda item: _decode_value(item, state, depth + 1)
    if tag == "par" and set(data) == {"tag", "name", "path"}:
        path = data["path"]
        if not isinstance(path, Mapping) or not isinstance(path.get("segments"), list):
            raise TemplateCodecError("template parameter is invalid")
        state.consume_entries(len(path["segments"]))
        try: return Par(data["name"], path=GraphPath.from_data(path))
        except Exception: raise TemplateCodecError("template parameter is invalid") from None
    if tag == "binary" and set(data) == {"tag", "op", "left", "right"} and data["op"] in {"mul", "truediv", "floordiv"}:
        left, right = value(data["left"]), value(data["right"])
        if not all(isinstance(item, Expr) or _validate_number(item) for item in (left, right)):
            raise TemplateCodecError("template binary operands are invalid")
        return _BinaryExpr(data["op"], left, right)
    if tag == "repeat" and set(data) == {"tag", "group", "count", "shared"} and type(data["shared"]) is bool:
        group = value(data["group"])
        if not isinstance(group, (FrozenList, FrozenTuple)): raise TemplateCodecError("template repeat group is invalid")
        try: return _RepeatExpr(list(group) if isinstance(group, FrozenList) else group, value(data["count"]), data["shared"])
        except Exception: raise TemplateCodecError("template repeat is invalid") from None
    if tag == "link" and set(data) == {"tag", "edge", "target"}:
        try: return DefLink.finalized(EdgeKind(data["edge"]), value(data["target"]))
        except Exception: raise TemplateCodecError("template link is invalid") from None
    if tag in {"definition", "cdef"} and set(data) == {"tag", "cls", "args", "kwargs", "stateful_role"}:
        cls, kwargs = value(data["cls"]), value(data["kwargs"])
        if not isinstance(kwargs, FrozenDict): raise TemplateCodecError("template definition fields are invalid")
        if tag == "cdef":
            if data["args"] is not None or type(data["stateful_role"]) is not bool: raise TemplateCodecError("template CDef is invalid")
            return ConcreteDefinition._from_bound_record(cls, BoundArguments(kwargs.items()), stateful_role=data["stateful_role"])
        if data["stateful_role"] is not None: raise TemplateCodecError("template Definition is invalid")
        args = data["args"]
        if args is not None:
            args = value(args)
            if not isinstance(args, FrozenTuple): raise TemplateCodecError("template Definition args are invalid")
        return Definition._from_symbolic_parts(cls, args, kwargs)
    if tag == "factory" and set(data) == {"tag", "target", "args", "kwargs"}:
        target, args, kwargs = value(data["target"]), value(data["args"]), value(data["kwargs"])
        if not isinstance(args, FrozenTuple) or not isinstance(kwargs, FrozenDict): raise TemplateCodecError("template factory is invalid")
        try: target = _validated_factory_target(target)
        except TypeError: raise TemplateCodecError("template factory target is invalid") from None
        return FactorySpec._from_symbolic_parts(target, tuple(args), kwargs)
    if tag == "map" and set(data) == {"tag", "items"} and isinstance(data["items"], list):
        if len(data["items"]) > _MAX_ENTRIES: raise ParameterizationLimitError("template container entry limit exceeded")
        state.consume_entries(len(data["items"]))
        items = [(value(pair[0]), value(pair[1])) for pair in data["items"] if isinstance(pair, list) and len(pair) == 2]
        if len(items) != len(data["items"]) or any(type(key) not in {str, int} for key, _ in items) or len({key for key, _ in items}) != len(items): raise TemplateCodecError("template map is invalid")
        return FrozenDict(items)
    if tag in {"list", "tuple", "set"} and set(data) == {"tag", "items"} and isinstance(data["items"], list):
        if len(data["items"]) > _MAX_ENTRIES: raise ParameterizationLimitError("template container entry limit exceeded")
        state.consume_entries(len(data["items"]))
        items = [value(item) for item in data["items"]]
        if tag == "set" and any(not _portable_set_member(item) for item in items):
            raise TemplateCodecError("template set members must be literal portable values")
        try:
            if tag == "list":
                return FrozenList(items)
            if tag == "tuple":
                return FrozenTuple(items)
            return FrozenSet(items)
        except Exception: raise TemplateCodecError("template set is invalid") from None
    if tag == "quoted" and set(data) == {"tag", "value"}: return QuotedDef(value(data["value"]))
    if tag == "selector-spec" and set(data) == {"tag", "selector"}: return SelectorSpec(value(data["selector"]))
    if tag == "selector" and set(data) == {"tag", "root", "strict", "cls_policy"}:
        try: return Selector(value(data["root"]), strict=data["strict"], cls_policy=data["cls_policy"])
        except Exception: raise TemplateCodecError("template selector is invalid") from None
    if tag == "object-ref" and set(data) == {"tag", "definition", "objects"}:
        if not isinstance(data["objects"], list):
            raise TemplateCodecError("template object reference is invalid")
        state.consume_entries(len(data["objects"]))
        try: return ObjectRef(value(data["definition"]), {GraphPath.from_data(item["path"]): ObjectId.from_data(item["object_id"]) for item in data["objects"]})
        except Exception: raise TemplateCodecError("template object reference is invalid") from None
    if tag == "state-ref" and set(data) == {"tag", "object", "states"}:
        if not isinstance(data["states"], list):
            raise TemplateCodecError("template state reference is invalid")
        state.consume_entries(len(data["states"]))
        try: return StateRef(value(data["object"]), {GraphPath.from_data(item["path"]): item["state"] for item in data["states"]})
        except Exception: raise TemplateCodecError("template state reference is invalid") from None
    raise TemplateCodecError("template node tag is invalid")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def _portable_set_member(value: object) -> bool:
    """Return whether one unordered member is a literal codec value.

    Symbolic expressions and graph-bearing values are intentionally excluded so
    a set cannot change their canonical graph address through ordering.
    """

    from .symbol import ImportRef, SourceSpec

    return value is None or type(value) in {bool, int, float, str, bytes} or isinstance(
        value, (ImportRef, SourceSpec)
    )


def _check_bytes(value: object) -> None:
    if len(_canonical_bytes(value)) > _MAX_BYTES:
        raise ParameterizationLimitError("template encoded payload size limit exceeded")
