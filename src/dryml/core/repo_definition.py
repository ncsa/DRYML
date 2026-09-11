"""Detached, bounded v1 configuration records for :class:`dryml.core.Repo`.

This module deliberately represents configuration only.  Its decoding paths do
not open Stores, resolve symbols, construct selectors, or activate sessions;
connected reconstruction is a separate boundary.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from dryml.formats.canonical import canonical_json_bytes, canonical_json_dumps


_BOUNDS = dict(
    max_depth=32, max_nodes=65536, max_entries=4096,
    max_string=1024 * 1024, max_int_bits=4096,
)
_MAX_JSON_BYTES = 16 * 1024 * 1024
_SCHEMA = "dryml-repo-definition"
_STATE_HASH = re.compile(r"[A-Za-z0-9]{1,32}-[0-9a-f]{64}\Z")
_CLASS_POLICIES = frozenset(("selector", "exact"))


class RepoDefinitionError(ValueError):
    """Raised when a portable Repo definition is invalid or unsupported.

    Error text identifies a configuration field but intentionally never includes
    user-provided values, which can contain sensitive configuration.
    """


def _error(path: str, message: str) -> RepoDefinitionError:
    return RepoDefinitionError(f"Invalid Repo definition at {path}: {message}.")


def _exact_keys(value: Any, keys: set[str], path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise _error(path, "unexpected or missing fields")
    return value


def _bounded_json_bytes(value: Any, path: str) -> bytes:
    """Return canonical bounded JSON bytes without exposing caller data in errors."""

    try:
        encoded = canonical_json_bytes(value, **_BOUNDS)
    except (RecursionError, TypeError, ValueError, OverflowError, UnicodeError):
        raise _error(path, "JSON value is invalid") from None
    if len(encoded) > _MAX_JSON_BYTES:
        raise _error(path, "encoded size bound exceeded")
    return encoded


def _json_value(value: Any, path: str) -> None:
    """Validate a JSON subtree with the shared bounded canonical codec."""

    _bounded_json_bytes(value, path)


def _symbol_data(value: Any, *, representation: str, path: str) -> dict[str, Any]:
    """Encode a symbol without resolving it or retaining a live callable."""

    from .symbol import ImportRef, SourceSpec, maybe_symbol_ref

    if value is None:
        return {"kind": "none"}
    if representation not in {"live", "symbolic"}:
        raise _error(path, "symbol representation is invalid")
    ref = value if isinstance(value, (ImportRef, SourceSpec)) else maybe_symbol_ref(value)
    if ref is None:
        raise _error(path, "symbol is not portable")
    if isinstance(ref, ImportRef):
        symbol = {"kind": "import", "module": ref.module, "qualname": ref.qualname}
    else:
        symbol = {
            "kind": "source", "source_kind": ref.kind, "source": ref.source,
            "name": ref.name,
            "imports": {
                name: {"module": item.module, "qualname": item.qualname}
                for name, item in (ref.imports or {}).items()
            },
        }
    return {"kind": "symbol", "representation": representation, "symbol": symbol}


def _validate_symbol(value: Any, path: str) -> None:
    if isinstance(value, Mapping) and set(value) == {"kind"} and value["kind"] == "none":
        return
    record = _exact_keys(value, {"kind", "representation", "symbol"}, path)
    if record["kind"] != "symbol" or record["representation"] not in {"live", "symbolic"}:
        raise _error(path, "symbol tag is invalid")
    symbol = record["symbol"]
    if not isinstance(symbol, Mapping) or not isinstance(symbol.get("kind"), str):
        raise _error(path, "symbol descriptor is invalid")
    if symbol["kind"] == "import":
        _exact_keys(symbol, {"kind", "module", "qualname"}, path)
        if not isinstance(symbol["module"], str) or not symbol["module"]:
            raise _error(path, "import module is invalid")
        if symbol["qualname"] is not None and (not isinstance(symbol["qualname"], str) or not symbol["qualname"]):
            raise _error(path, "import qualname is invalid")
        return
    if symbol["kind"] != "source":
        raise _error(path, "symbol kind is unsupported")
    _exact_keys(symbol, {"kind", "source_kind", "source", "name", "imports"}, path)
    if symbol["source_kind"] not in {"function", "class"} or not isinstance(symbol["source"], str) or not symbol["source"]:
        raise _error(path, "source symbol is invalid")
    if symbol["name"] is not None and not isinstance(symbol["name"], str):
        raise _error(path, "source name is invalid")
    if symbol["source_kind"] == "class" and not symbol["name"]:
        raise _error(path, "source class name is invalid")
    if not isinstance(symbol["imports"], Mapping):
        raise _error(path, "source imports are invalid")
    for name, item in symbol["imports"].items():
        if not isinstance(name, str):
            raise _error(path, "source import name is invalid")
        _exact_keys(item, {"module", "qualname"}, path)
        if not isinstance(item["module"], str) or not item["module"]:
            raise _error(path, "source import is invalid")
        if item["qualname"] is not None and (
            not isinstance(item["qualname"], str) or not item["qualname"]
        ):
            raise _error(path, "source import qualname is invalid")


class _SelectorEncoder:
    """Encode one selector graph while retaining object-identity topology."""

    def __init__(self) -> None:
        self.nodes: list[dict[str, Any]] = []
        self.labels: dict[int, str] = {}
        self.active: set[int] = set()

    def value(self, value: Any, path: str) -> dict[str, Any]:
        from .definition import ConcreteDefinition, Definition
        from .freeze import FrozenDict, FrozenList, FrozenSet, FrozenTuple
        from .links import DefLink
        from .params import Par
        from .quoted import QuotedDef, SelectorSpec
        from .reference_values import ObjectRef, StateRef
        from .selector import Selector
        from .symbol import ImportRef, SourceSpec

        if isinstance(value, (Definition, ConcreteDefinition)):
            return self.definition(value, path)
        if isinstance(value, QuotedDef):
            return {"kind": "quoted-definition", "value": self.value(value.value, path + ".value")}
        if isinstance(value, SelectorSpec):
            return {
                "kind": "selector-spec",
                "selector": _SelectorEncoder().selector(value.selector, path + ".selector"),
            }
        if isinstance(value, Selector):
            raise _error(path, "unquoted selector is not portable")
        if isinstance(value, (ImportRef, SourceSpec)):
            return _symbol_data(value, representation="symbolic", path=path)
        if isinstance(value, DefLink):
            return {"kind": "link", "edge": value.kind.value, "target": self.value(value.target, path + ".target")}
        if isinstance(value, ObjectRef):
            return {"kind": "object-ref", "definition": self.definition(value.definition, path + ".definition"), "objects": value.to_data()["objects"]}
        if isinstance(value, StateRef):
            return {"kind": "state-ref", "object": self.value(value.object, path + ".object"), "states": value.to_data()["states"]}
        if isinstance(value, Par):
            return self.par(value, path)
        if isinstance(value, (FrozenDict, dict)):
            if any(not isinstance(key, str) for key in value):
                raise _error(path, "map keys must be strings")
            return {"kind": "map", "items": [[key, self.value(item, f"{path}.{key}")] for key, item in sorted(value.items())]}
        if isinstance(value, (FrozenList, list)):
            return {"kind": "list", "items": [self.value(item, f"{path}[{index}]") for index, item in enumerate(value)]}
        if isinstance(value, (FrozenTuple, tuple)):
            return {"kind": "tuple", "items": [self.value(item, f"{path}[{index}]") for index, item in enumerate(value)]}
        if isinstance(value, (FrozenSet, set, frozenset)):
            from .utils.stable_hash import stable_hash_function

            # Label assignment is global, so order raw members before traversing
            # them with this encoder.  The probe has isolated labels and cannot
            # merge or alter the topology retained by the final traversal.
            try:
                ordered = sorted(
                    value,
                    key=lambda item: canonical_json_dumps(
                        _SelectorEncoder().value(item, path + ".set"), **_BOUNDS
                    ),
                )
            except (RecursionError, TypeError, ValueError, OverflowError, UnicodeError):
                raise _error(path, "set member is not portable") from None
            members = []
            for item in ordered:
                encoded = self.value(item, path + ".set")
                try:
                    fingerprint = stable_hash_function(item)
                except (RecursionError, TypeError, ValueError, OverflowError):
                    raise _error(path, "set member is not portable") from None
                members.append({"fingerprint": fingerprint, "value": encoded})
            members.sort(key=lambda member: canonical_json_dumps(member["value"], **_BOUNDS))
            if len({canonical_json_dumps(member["value"], **_BOUNDS) for member in members}) != len(members):
                raise _error(path, "set contains duplicate canonical values")
            if len({member["fingerprint"] for member in members}) != len(members):
                raise _error(path, "set contains ambiguous stable members")
            return {"kind": "set", "items": members}
        if value is None or type(value) in {bool, int, float, str}:
            _json_value(value, path)
            return {"kind": "atom", "value": value}
        raise _error(path, "value type is not portable")

    def definition(self, value: Any, path: str) -> dict[str, Any]:
        from .definition import ConcreteDefinition
        key = id(value)
        if key in self.active:
            raise _error(path, "definition cycle is unsupported")
        if key in self.labels:
            return {"kind": "definition-ref", "label": self.labels[key]}
        label = f"n{len(self.labels)}"
        self.labels[key] = label
        self.active.add(key)
        try:
            from .symbol import ImportRef, SourceSpec

            representation = (
                "symbolic"
                if isinstance(value.cls, (ImportRef, SourceSpec))
                else "live"
            )
            node: dict[str, Any] = {"label": label, "node_kind": "cdef" if isinstance(value, ConcreteDefinition) else "definition", "cls": _symbol_data(value.cls, representation=representation, path=path + ".cls")}
            if isinstance(value, ConcreteDefinition):
                node["parameters"] = [[name, self.value(item, f"{path}.{name}")] for name, item in value.parameters.items()]
                node["stateful_role"] = value._stateful_role
            else:
                node["args"] = None if value.args is None else [self.value(item, f"{path}.args[{index}]") for index, item in enumerate(value.args)]
                node["kwargs"] = [[name, self.value(item, f"{path}.{name}")] for name, item in value.kwargs.items()]
            self.nodes.append(node)
            return {"kind": "definition-ref", "label": label}
        finally:
            self.active.remove(key)

    def par(self, value: Any, path: str) -> dict[str, Any]:
        from .params import (
            AnyMatcher, ChoiceMatcher, ExactMatcher, IntRangeMatcher,
            MissingMatcher, PresentMatcher, SubclassMatcher,
            UniformFromSetGenerator, UniformIntRangeGenerator,
        )
        matcher = value.matcher
        if type(matcher) is PresentMatcher:
            match = {"kind": "present"}
        elif type(matcher) is MissingMatcher:
            match = {"kind": "missing"}
        elif type(matcher) is AnyMatcher:
            match = {"kind": "any"}
        elif type(matcher) is ExactMatcher:
            match = {"kind": "exact", "value": self.value(matcher.value, path + ".exact")}
        elif type(matcher) is ChoiceMatcher:
            match = {"kind": "choice", "values": [self.value(item, path + ".choice") for item in matcher.values]}
        elif type(matcher) is IntRangeMatcher:
            match = {"kind": "int-range", "lo": matcher.lo, "hi": matcher.hi}
        elif type(matcher) is SubclassMatcher:
            match = {"kind": "subclass", "cls": _symbol_data(matcher.cls, representation="live", path=path + ".subclass")}
        else:
            raise _error(path, "matcher is not portable")
        generator = value.generator
        if generator is None:
            gen = None
        elif type(generator) is UniformIntRangeGenerator:
            gen = {"kind": "uniform-int-range", "lo": generator.lo, "hi": generator.hi}
        elif type(generator) is UniformFromSetGenerator:
            gen = {"kind": "uniform-from-set", "values": [self.value(item, path + ".generator") for item in generator.values]}
        else:
            raise _error(path, "generator is not portable")
        return {"kind": "par", "name": value.name, "matcher": match, "generator": gen}

    def selector(self, selector: Any, path: str) -> dict[str, Any]:
        return {"root": self.definition(selector.root, path + ".root"), "strict": selector.strict, "cls_policy": selector.cls_policy, "nodes": self.nodes}


def _validate_par(value: Mapping[str, Any], path: str, validate_value: Any) -> None:
    """Validate one closed Par descriptor without invoking matcher behavior."""

    _exact_keys(value, {"kind", "name", "matcher", "generator"}, path)
    if value["kind"] != "par" or (value["name"] is not None and not isinstance(value["name"], str)):
        raise _error(path, "parameter descriptor is invalid")
    matcher = value["matcher"]
    if not isinstance(matcher, Mapping) or not isinstance(matcher.get("kind"), str):
        raise _error(path, "matcher is invalid")
    matcher_kind = matcher["kind"]
    if matcher_kind in {"present", "missing", "any"}:
        _exact_keys(matcher, {"kind"}, path + ".matcher")
    elif matcher_kind == "exact":
        _exact_keys(matcher, {"kind", "value"}, path + ".matcher")
        validate_value(matcher["value"], path + ".matcher.value")
    elif matcher_kind == "choice":
        _exact_keys(matcher, {"kind", "values"}, path + ".matcher")
        if not isinstance(matcher["values"], list) or len(matcher["values"]) > 4096:
            raise _error(path, "choice values are invalid")
        for index, child in enumerate(matcher["values"]):
            validate_value(child, f"{path}.matcher.values[{index}]")
    elif matcher_kind == "int-range":
        _exact_keys(matcher, {"kind", "lo", "hi"}, path + ".matcher")
        if type(matcher["lo"]) is not int or type(matcher["hi"]) is not int or matcher["lo"] > matcher["hi"]:
            raise _error(path, "range bounds are invalid")
    elif matcher_kind == "subclass":
        _exact_keys(matcher, {"kind", "cls"}, path + ".matcher")
        _validate_symbol(matcher["cls"], path + ".matcher.cls")
        if matcher["cls"].get("representation") != "live":
            raise _error(path, "subclass matcher requires a live type")
    else:
        raise _error(path, "matcher is invalid")

    generator = value["generator"]
    if generator is None:
        return
    if not isinstance(generator, Mapping) or not isinstance(generator.get("kind"), str):
        raise _error(path, "generator is invalid")
    if generator["kind"] == "uniform-int-range":
        _exact_keys(generator, {"kind", "lo", "hi"}, path + ".generator")
        if type(generator["lo"]) is not int or type(generator["hi"]) is not int or generator["lo"] > generator["hi"]:
            raise _error(path, "generator bounds are invalid")
        return
    if generator["kind"] == "uniform-from-set":
        _exact_keys(generator, {"kind", "values"}, path + ".generator")
        if not isinstance(generator["values"], list) or len(generator["values"]) > 4096:
            raise _error(path, "generator values are invalid")
        for index, child in enumerate(generator["values"]):
            validate_value(child, f"{path}.generator.values[{index}]")
        return
    raise _error(path, "generator is invalid")


def _validate_selector(value: Any, path: str) -> None:
    record = _exact_keys(value, {"root", "strict", "cls_policy", "nodes"}, path)
    if type(record["strict"]) is not bool or record["cls_policy"] not in _CLASS_POLICIES:
        raise _error(path, "selector policy is invalid")
    if not isinstance(record["nodes"], list) or len(record["nodes"]) > 65536:
        raise _error(path, "selector nodes are invalid")

    labels: dict[str, Mapping[str, Any]] = {}
    for index, node in enumerate(record["nodes"]):
        node_path = f"{path}.nodes[{index}]"
        if not isinstance(node, Mapping) or not isinstance(node.get("label"), str):
            raise _error(node_path, "definition label is invalid")
        if node["label"] in labels:
            raise _error(node_path, "definition label is duplicated")
        if node.get("node_kind") == "definition":
            _exact_keys(node, {"label", "node_kind", "cls", "args", "kwargs"}, node_path)
            if node["args"] is not None and not isinstance(node["args"], list):
                raise _error(node_path, "definition args are invalid")
            if (
                    isinstance(node["cls"], Mapping)
                    and node["cls"].get("kind") == "none"
                    and node["args"]
            ):
                raise _error(node_path, "classless definitions cannot have positional args")
            pairs = node["kwargs"]
        elif node.get("node_kind") == "cdef":
            _exact_keys(node, {"label", "node_kind", "cls", "parameters", "stateful_role"}, node_path)
            if type(node["stateful_role"]) is not bool:
                raise _error(node_path, "CDef role is invalid")
            pairs = node["parameters"]
        else:
            raise _error(node_path, "definition node kind is invalid")
        _validate_symbol(node["cls"], node_path + ".cls")
        if not isinstance(pairs, list) or len(pairs) > 4096:
            raise _error(node_path, "definition fields are invalid")
        names = set()
        for pair in pairs:
            if not isinstance(pair, list) or len(pair) != 2 or not isinstance(pair[0], str):
                raise _error(node_path, "definition field is invalid")
            if pair[0] in names:
                raise _error(node_path, "definition field is duplicated")
            names.add(pair[0])
        labels[node["label"]] = node

    seen: set[str] = set()
    active: set[str] = set()

    def definition_ref(current: Any, item_path: str, *, cdef: bool | None = None) -> str:
        _exact_keys(current, {"kind", "label"}, item_path)
        if current["kind"] != "definition-ref" or not isinstance(current["label"], str):
            raise _error(item_path, "definition reference is invalid")
        label = current["label"]
        node = labels.get(label)
        if node is None:
            raise _error(item_path, "definition reference is missing")
        if cdef is not None and (node["node_kind"] == "cdef") != cdef:
            raise _error(item_path, "definition reference has the wrong kind")
        if label in active:
            raise _error(item_path, "definition cycle is unsupported")
        if label not in seen:
            active.add(label)
            values = [] if node.get("args") is None else node.get("args", [])
            values = [*values, *(pair[1] for pair in node.get("kwargs", node.get("parameters", [])))]
            for child in values:
                item(child, item_path)
            active.remove(label)
            seen.add(label)
        return label

    def reference_entries(entries: Any, item_path: str, *, state: bool) -> dict[Any, Any]:
        if not isinstance(entries, list) or len(entries) > 4096:
            raise _error(item_path, "reference entries are invalid")
        from .reference_values import ObjectId
        from .utils.graph.path import GraphPath

        result = {}
        for index, entry in enumerate(entries):
            entry_path = f"{item_path}[{index}]"
            _exact_keys(entry, {"path", "state"} if state else {"path", "object_id"}, entry_path)
            try:
                graph_path = GraphPath.from_data(entry["path"])
                if graph_path in result:
                    raise ValueError()
                value = entry["state"] if state else ObjectId.from_data(entry["object_id"])
                if state and (not isinstance(value, str) or not _STATE_HASH.fullmatch(value)):
                    raise ValueError()
            except (RecursionError, TypeError, ValueError, OverflowError):
                raise _error(entry_path, "reference path or identity is invalid") from None
            result[graph_path] = value
        if not state and len(set(result.values())) != len(result):
            raise _error(item_path, "independent reference identities are duplicated")
        return result

    def validate_object_topology(current: Mapping[str, Any], item_path: str) -> None:
        """Check ObjectRef paths against the complete descriptor-only CDef graph."""

        from .utils.graph.path import (
            GraphPath,
            Index,
            Key,
            Parameter,
            SetMember,
            graph_path_sort_key,
        )

        root_label = definition_ref(current["definition"], item_path + ".definition", cdef=True)
        objects = reference_entries(current["objects"], item_path + ".objects", state=False)
        expected: dict[tuple[Any, ...], tuple[GraphPath, Any | None]] = {}
        def add(key: tuple[Any, ...], graph_path: GraphPath, object_id: Any | None) -> None:
            previous = expected.get(key)
            if previous is None or graph_path_sort_key(graph_path) < graph_path_sort_key(previous[0]):
                expected[key] = (graph_path, object_id)

        def visit_reference(reference: Mapping[str, Any], graph_path: GraphPath, reference_path: str) -> None:
            object_descriptor = reference if reference["kind"] == "object-ref" else reference["object"]
            embedded = reference_entries(
                object_descriptor["objects"], reference_path + ".objects", state=False
            )
            for child_path, object_id in embedded.items():
                add(("object-id", object_id), graph_path.join(child_path), object_id)

        def visit_value(value: Mapping[str, Any], graph_path: GraphPath, value_path: str, active_labels: set[str]) -> None:
            kind = value["kind"]
            if kind == "definition-ref":
                label = value["label"]
                if labels[label]["node_kind"] != "cdef":
                    raise _error(value_path, "CDef graph contains a partial Definition")
                visit_cdef(label, graph_path, value_path, active_labels)
                return
            if kind in {"object-ref", "state-ref"}:
                visit_reference(value, graph_path, value_path)
                return
            if kind == "link":
                if value["edge"] == "materialize":
                    visit_value(value["target"], graph_path, value_path + ".target", active_labels)
                return
            if kind in {"list", "tuple"}:
                for index, child in enumerate(value["items"]):
                    visit_value(child, graph_path.child(Index(index)), f"{value_path}[{index}]", active_labels)
                return
            if kind == "map":
                for index, pair in enumerate(value["items"]):
                    visit_value(pair[1], graph_path.child(Key(pair[0])), f"{value_path}[{index}]", active_labels)
                return
            if kind == "set":
                for member in value["items"]:
                    visit_value(
                        member["value"],
                        graph_path.child(SetMember(member["fingerprint"])),
                        value_path + ".set",
                        active_labels,
                    )
                return
            if kind == "par":
                raise _error(value_path, "CDef graph contains a parameter placeholder")

        def visit_cdef(label: str, graph_path: GraphPath, value_path: str, active_labels: set[str]) -> None:
            if label in active_labels:
                raise _error(value_path, "definition cycle is unsupported")
            node = labels[label]
            if node["stateful_role"]:
                add(("node", label), graph_path, None)
            active_labels.add(label)
            for name, child in node["parameters"]:
                visit_value(
                    child,
                    graph_path.child(Parameter(name)),
                    f"{value_path}.parameters",
                    active_labels,
                )
            active_labels.remove(label)

        visit_cdef(root_label, GraphPath(), item_path + ".definition", set())
        expected_paths = {path for path, _ in expected.values()}
        if set(objects) != expected_paths:
            raise _error(item_path, "reference paths do not match CDef topology")
        for key, (graph_path, object_id) in expected.items():
            if object_id is not None and objects[graph_path] != object_id:
                raise _error(item_path, "reference identity does not match imported topology")

    def item(current: Any, item_path: str) -> None:
        if not isinstance(current, Mapping) or not isinstance(current.get("kind"), str):
            raise _error(item_path, "value descriptor is invalid")
        kind = current["kind"]
        if kind == "atom":
            _exact_keys(current, {"kind", "value"}, item_path)
            if current["value"] is not None and type(current["value"]) not in {bool, int, float, str}:
                raise _error(item_path, "atom is not a JSON primitive")
            _json_value(current["value"], item_path)
            return
        if kind == "symbol":
            _validate_symbol(current, item_path)
            return
        if kind == "definition-ref":
            definition_ref(current, item_path)
            return
        if kind == "quoted-definition":
            _exact_keys(current, {"kind", "value"}, item_path)
            item(current["value"], item_path + ".value")
            return
        if kind == "selector-spec":
            _exact_keys(current, {"kind", "selector"}, item_path)
            _validate_selector(current["selector"], item_path + ".selector")
            return
        if kind in {"list", "tuple"}:
            _exact_keys(current, {"kind", "items"}, item_path)
            if not isinstance(current["items"], list) or len(current["items"]) > 4096:
                raise _error(item_path, "container is invalid")
            for index, child in enumerate(current["items"]):
                item(child, f"{item_path}[{index}]")
            return
        if kind == "set":
            _exact_keys(current, {"kind", "items"}, item_path)
            if not isinstance(current["items"], list) or len(current["items"]) > 4096:
                raise _error(item_path, "container is invalid")
            canonical_members = set()
            fingerprints = set()
            for index, member in enumerate(current["items"]):
                member_path = f"{item_path}[{index}]"
                _exact_keys(member, {"fingerprint", "value"}, member_path)
                if not isinstance(member["fingerprint"], str) or not re.fullmatch(r"[0-9a-f]{64}", member["fingerprint"]):
                    raise _error(member_path, "set fingerprint is invalid")
                if member["fingerprint"] in fingerprints:
                    raise _error(item_path, "set contains ambiguous stable members")
                fingerprints.add(member["fingerprint"])
                item(member["value"], member_path + ".value")
                encoded = _bounded_json_bytes(member["value"], member_path + ".value")
                if encoded in canonical_members:
                    raise _error(item_path, "set contains duplicate canonical values")
                canonical_members.add(encoded)
            return
        if kind == "map":
            _exact_keys(current, {"kind", "items"}, item_path)
            if not isinstance(current["items"], list) or len(current["items"]) > 4096:
                raise _error(item_path, "map is invalid")
            names = set()
            for index, pair in enumerate(current["items"]):
                pair_path = f"{item_path}[{index}]"
                if not isinstance(pair, list) or len(pair) != 2 or not isinstance(pair[0], str):
                    raise _error(pair_path, "map entry is invalid")
                if pair[0] in names:
                    raise _error(pair_path, "map key is duplicated")
                names.add(pair[0])
                item(pair[1], pair_path)
            return
        if kind == "link":
            _exact_keys(current, {"kind", "edge", "target"}, item_path)
            if current["edge"] not in {"ref", "materialize"}:
                raise _error(item_path, "link edge is invalid")
            item(current["target"], item_path + ".target")
            return
        if kind == "par":
            _validate_par(current, item_path, item)
            return
        if kind == "object-ref":
            _exact_keys(current, {"kind", "definition", "objects"}, item_path)
            definition_ref(current["definition"], item_path + ".definition", cdef=True)
            reference_entries(current["objects"], item_path + ".objects", state=False)
            validate_object_topology(current, item_path)
            return
        if kind == "state-ref":
            _exact_keys(current, {"kind", "object", "states"}, item_path)
            if not isinstance(current["object"], Mapping) or current["object"].get("kind") != "object-ref":
                raise _error(item_path, "state reference object is invalid")
            item(current["object"], item_path + ".object")
            objects = reference_entries(current["object"]["objects"], item_path + ".object.objects", state=False)
            states = reference_entries(current["states"], item_path + ".states", state=True)
            if set(objects) != set(states):
                raise _error(item_path, "state paths do not match object reference")
            return
        raise _error(item_path, "value tag is unsupported")

    if not isinstance(record["root"], Mapping) or record["root"].get("kind") != "definition-ref":
        raise _error(path + ".root", "selector root must be a Definition")
    definition_ref(record["root"], path + ".root", cdef=False)
    if len(seen) != len(labels):
        raise _error(path, "definition graph contains unreachable nodes")


def _validate_data(data: Any) -> dict[str, Any]:
    record = _exact_keys(data, {"schema", "version", "stores", "default_store", "routing", "settings"}, "$")
    if record["schema"] != _SCHEMA or type(record["version"]) is not int or record["version"] != 1:
        raise _error("$", "schema or version is unsupported")
    if not isinstance(record["stores"], list):
        raise _error("$.stores", "must be a list")
    stores = set()
    for index, store in enumerate(record["stores"]):
        store = _exact_keys(store, {"kind", "path", "query_index"} if isinstance(store, Mapping) and store.get("kind") == "dir" else {"kind", "path"}, f"$.stores[{index}]")
        if store["kind"] == "dir":
            if store["query_index"] not in {"auto", "sqlite", "memory", "none"}:
                raise _error(f"$.stores[{index}]", "query policy is invalid")
        elif store["kind"] != "zip":
            raise _error(f"$.stores[{index}]", "store kind is unsupported")
        if not isinstance(store["path"], str) or not os.path.isabs(store["path"]):
            raise _error(f"$.stores[{index}]", "store path must be absolute")
        key = (store["kind"], store["path"])
        if key in stores: raise _error("$.stores", "stores must be distinct")
        stores.add(key)
    default = record["default_store"]
    if (not record["stores"] and default is not None) or (record["stores"] and (type(default) is not int or default != 0)):
        raise _error("$.default_store", "must select the first store or be null")
    routing = record["routing"]
    if routing is not None:
        routing = _exact_keys(routing, {"graph_mode", "match_mode", "routes"}, "$.routing")
        if (
            routing["graph_mode"] not in {"per-object", "closure"}
            or routing["match_mode"] not in {"first", "all"}
            or not isinstance(routing["routes"], list)
        ):
            raise _error("$.routing", "routing modes are invalid")
        for index, route in enumerate(routing["routes"]):
            route = _exact_keys(route, {"selector", "store"}, f"$.routing.routes[{index}]")
            if type(route["store"]) is not int or not 0 <= route["store"] < len(record["stores"]):
                raise _error(f"$.routing.routes[{index}]", "store index is invalid")
            _validate_selector(route["selector"], f"$.routing.routes[{index}].selector")
    settings = _exact_keys(record["settings"], {"config", "lease_duration", "save_objs_on_deletion"}, "$.settings")
    _json_value(settings["config"], "$.settings.config")
    if not isinstance(settings["config"], Mapping):
        raise _error("$.settings.config", "must be a mapping")
    if (
        type(settings["lease_duration"]) not in {int, float}
        or not math.isfinite(settings["lease_duration"])
        or not 0 < settings["lease_duration"] <= 3600
    ):
        raise _error("$.settings.lease_duration", "is invalid")
    if type(settings["save_objs_on_deletion"]) is not bool:
        raise _error("$.settings.save_objs_on_deletion", "must be bool")
    return json.loads(_bounded_json_bytes(record, "$"))


@dataclass(frozen=True, slots=True, init=False)
class RepoDefinition:
    """An immutable, inert v1 Repo configuration snapshot.

    ``to_data`` and ``from_data`` only inspect detached descriptors.  They never
    open persistent resources, resolve symbols, or construct a live Repo.
    """

    _data: dict[str, Any]

    def __init__(self, data: Mapping[str, Any]) -> None:
        """Validate and retain one inert v1 configuration mapping.

        Args:
            data: Complete v1 envelope with only supported JSON descriptors.

        Raises:
            TypeError: If ``data`` is not a mapping.
            RepoDefinitionError: If the envelope violates the closed v1 schema.
        """
        if not isinstance(data, Mapping):
            raise TypeError("Repo definition data must be a mapping.")
        try:
            validated = _validate_data(data)
        except RepoDefinitionError:
            raise
        except (RecursionError, TypeError, ValueError, OverflowError, UnicodeError):
            raise RepoDefinitionError("Invalid Repo definition.") from None
        object.__setattr__(self, "_data", validated)

    def to_data(self) -> dict[str, Any]:
        """Return a fresh bounded JSON-compatible copy for caller inspection.

        Returns:
            A detached mapping that cannot mutate this definition.

        Raises:
            RepoDefinitionError: If retained data cannot be encoded safely.
        """
        try:
            return json.loads(_bounded_json_bytes(self._data, "$"))
        except RepoDefinitionError:
            raise RepoDefinitionError("Repo definition could not be encoded.") from None

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "RepoDefinition":
        """Validate and detach one v1 mapping without activating resources.

        Args:
            data: Complete JSON-compatible v1 envelope.

        Returns:
            An immutable inert configuration value.

        Raises:
            TypeError: If ``data`` is not a mapping.
            RepoDefinitionError: If fields, bounds, or descriptor grammar fail.
        """
        if not isinstance(data, Mapping):
            raise TypeError("Repo definition data must be a mapping.")
        try:
            return cls(data)
        except RepoDefinitionError:
            raise
        except (RecursionError, ValueError, TypeError, OverflowError):
            raise RepoDefinitionError("Invalid Repo definition.") from None

    def to_json(self) -> str:
        """Return deterministic bounded JSON for caller-owned transport.

        Returns:
            Canonical v1 JSON text no larger than the encoded-size limit.

        Raises:
            RepoDefinitionError: If retained data cannot be encoded safely.
        """
        try:
            return _bounded_json_bytes(self._data, "$").decode("utf-8")
        except (RepoDefinitionError, UnicodeError):
            raise RepoDefinitionError("Repo definition could not be encoded.") from None

    @classmethod
    def from_json(cls, data: str) -> "RepoDefinition":
        """Parse and validate bounded JSON without opening Stores or symbols.

        Args:
            data: JSON text containing one complete v1 envelope.

        Returns:
            An immutable inert configuration value.

        Raises:
            TypeError: If ``data`` is not text.
            RepoDefinitionError: If parsing, bounds, duplicate keys, or grammar
                validation fails.
        """
        if not isinstance(data, str):
            raise TypeError("Repo definition JSON must be a string.")
        try:
            if len(data.encode("utf-8")) > _MAX_JSON_BYTES:
                raise RepoDefinitionError("Repo definition JSON exceeds encoded size bound.")
            decoded = json.loads(data, parse_constant=lambda _: (_ for _ in ()).throw(ValueError()), object_pairs_hook=_duplicate_free_mapping)
        except RepoDefinitionError:
            raise
        except (ValueError, TypeError, RecursionError, UnicodeError):
            raise RepoDefinitionError("Repo definition JSON is malformed.") from None
        return cls.from_data(decoded)


def _symbol_from_data(value: Mapping[str, Any], *, require_live: bool = False) -> Any:
    """Rebuild one validated symbol while preserving its representation tag."""

    from .symbol import ImportRef, SourceSpec

    if value["kind"] == "none":
        if require_live:
            raise _error("$.routing", "a live type is required")
        return None
    symbol = value["symbol"]
    if symbol["kind"] == "import":
        result = ImportRef(symbol["module"], symbol["qualname"])
    else:
        result = SourceSpec(
            symbol["source_kind"], symbol["source"], symbol["name"],
            {
                name: ImportRef(item["module"], item["qualname"])
                for name, item in symbol["imports"].items()
            },
        )
    if value["representation"] == "symbolic" and not require_live:
        return result
    try:
        result = result.resolve()
    except Exception as error:
        raise RepoDefinitionError("Repo definition could not resolve a live selector symbol.") from error
    if require_live and not isinstance(result, type):
        raise RepoDefinitionError("Repo definition subclass matcher did not resolve to a type.")
    return result


def _selector_from_data(value: Mapping[str, Any]):
    """Construct one validated Selector graph at the explicit live boundary."""

    from .bound_args import BoundArguments
    from .cdef_graph import EdgeKind
    from .definition import ConcreteDefinition, Definition, SKIP_ARGS
    from .freeze import FrozenDict, FrozenList, FrozenSet, FrozenTuple
    from .links import DefLink
    from .params import (
        AnyMatcher, ChoiceMatcher, ExactMatcher, IntRangeMatcher, MissingMatcher,
        Par, PresentMatcher, SubclassMatcher, UniformFromSetGenerator,
        UniformIntRangeGenerator,
    )
    from .quoted import QuotedDef, SelectorSpec
    from .reference_values import ObjectId, ObjectRef, StateRef
    from .selector import Selector
    from .utils.graph.path import GraphPath

    nodes = {node["label"]: node for node in value["nodes"]}
    definitions: dict[str, Any] = {}

    def item(current: Mapping[str, Any]) -> Any:
        kind = current["kind"]
        if kind == "atom":
            return current["value"]
        if kind == "symbol":
            return _symbol_from_data(current)
        if kind == "definition-ref":
            return definition(current["label"])
        if kind == "quoted-definition":
            return QuotedDef(item(current["value"]))
        if kind == "selector-spec":
            return SelectorSpec(_selector_from_data(current["selector"]))
        if kind == "list":
            return FrozenList(item(child) for child in current["items"])
        if kind == "tuple":
            return FrozenTuple(item(child) for child in current["items"])
        if kind == "set":
            return FrozenSet(item(member["value"]) for member in current["items"])
        if kind == "map":
            return FrozenDict((name, item(child)) for name, child in current["items"])
        if kind == "link":
            return DefLink(EdgeKind(current["edge"]), item(current["target"]))
        if kind == "par":
            matcher_data = current["matcher"]
            matcher_kind = matcher_data["kind"]
            if matcher_kind == "present":
                matcher = PresentMatcher()
            elif matcher_kind == "missing":
                matcher = MissingMatcher()
            elif matcher_kind == "any":
                matcher = AnyMatcher()
            elif matcher_kind == "exact":
                matcher = ExactMatcher(item(matcher_data["value"]))
            elif matcher_kind == "choice":
                matcher = ChoiceMatcher(item(child) for child in matcher_data["values"])
            elif matcher_kind == "int-range":
                matcher = IntRangeMatcher(matcher_data["lo"], matcher_data["hi"])
            else:
                matcher = SubclassMatcher(_symbol_from_data(matcher_data["cls"], require_live=True))
            generator_data = current["generator"]
            if generator_data is None:
                generator = None
            elif generator_data["kind"] == "uniform-int-range":
                generator = UniformIntRangeGenerator(generator_data["lo"], generator_data["hi"])
            else:
                generator = UniformFromSetGenerator(
                    item(child) for child in generator_data["values"]
                )
            return Par(current["name"], matcher, generator)
        if kind == "object-ref":
            objects = {
                GraphPath.from_data(entry["path"]): ObjectId.from_data(entry["object_id"])
                for entry in current["objects"]
            }
            return ObjectRef(item(current["definition"]), objects)
        if kind == "state-ref":
            states = {
                GraphPath.from_data(entry["path"]): entry["state"]
                for entry in current["states"]
            }
            return StateRef(item(current["object"]), states)
        raise AssertionError(f"validated selector value has unknown kind {kind!r}")

    def definition(label: str) -> Any:
        existing = definitions.get(label)
        if existing is not None:
            return existing
        node = nodes[label]
        cls = _symbol_from_data(node["cls"])
        if node["node_kind"] == "cdef":
            result = ConcreteDefinition._from_bound_record(
                cls,
                BoundArguments(tuple((name, item(child)) for name, child in node["parameters"])),
                stateful_role=node["stateful_role"],
            )
        else:
            args = node["args"]
            kwargs = {name: item(child) for name, child in node["kwargs"]}
            if cls is None:
                result = Definition(SKIP_ARGS, **kwargs) if args is None else Definition(**kwargs)
            elif args is None:
                result = Definition(cls, SKIP_ARGS, **kwargs)
            else:
                result = Definition(cls, *(item(child) for child in args), **kwargs)
        definitions[label] = result
        return result

    root = item(value["root"])
    return Selector(root, strict=value["strict"], cls_policy=value["cls_policy"])


def repo_from_definition(definition: RepoDefinition):
    """Reconnect a fully validated definition using only existing Store authority.

    Reconstruction deliberately resolves live selector operands and opens Stores
    only after all inert grammar and persistent path/type checks have succeeded.
    The returned Repo owns the fresh Store handles; any failure closes only the
    handles opened by this call without committing their buffered state.
    """

    if not isinstance(definition, RepoDefinition):
        raise TypeError("Repo.from_definition requires a RepoDefinition.")
    # Revalidate a detached copy so live reconstruction never trusts a retained
    # implementation detail or a future subclass's mutable backing object.
    data = RepoDefinition.from_data(definition.to_data()).to_data()
    routing_data = data["routing"]
    try:
        routing_parts = None if routing_data is None else _reconstruct_routing(routing_data)
        _preflight_store_descriptors(data["stores"])
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as error:
        raise RepoDefinitionError("Repo definition reconstruction preflight failed.") from error

    from .repo import Repo
    from .store.dir import DirStore
    from .store.zip import ZipStore

    opened = []
    repo = None
    try:
        for descriptor in data["stores"]:
            if descriptor["kind"] == "dir":
                opened.append(DirStore.open_existing(
                    descriptor["path"], query_index=descriptor["query_index"],
                ))
            else:
                opened.append(ZipStore.open_existing(descriptor["path"]))
        if routing_parts is None:
            routing = None
        else:
            selectors, store_indexes, routing_data = routing_parts
            from .repo_plan import SaveRouting
            routing = SaveRouting(
                tuple((selector, opened[index]) for selector, index in zip(selectors, store_indexes)),
                routing_data["match_mode"], routing_data["graph_mode"],
            )
        repo = Repo(
            opened,
            config=data["settings"]["config"],
            lease_duration=data["settings"]["lease_duration"],
            save_routing=routing,
        )
        repo.save_objs_on_deletion = data["settings"]["save_objs_on_deletion"]
        repo._adopt_owned_stores(opened)
        return repo
    except BaseException as error:
        if repo is not None:
            repo._adopt_owned_stores(opened)
            try:
                repo.close(flush=False)
            except BaseException:
                pass
        else:
            for store in reversed(opened):
                try:
                    store.close()
                except BaseException:
                    pass
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        if isinstance(error, RepoDefinitionError):
            raise
        raise RepoDefinitionError("Repo definition could not open required Store authority.") from error


def _reconstruct_routing(data: Mapping[str, Any]):
    """Construct validated live selectors and bind their table destinations."""

    # Store indexes remain inert until the caller binds them to fresh handles.
    selectors = tuple(_selector_from_data(route["selector"]) for route in data["routes"])
    return (selectors, tuple(route["store"] for route in data["routes"]), data)


def _preflight_store_descriptors(stores: list[Mapping[str, Any]]) -> None:
    """Validate every required persistent path/type before opening any Store."""

    from .store.dir import DirStore
    from .store.zip import ZipStore

    for descriptor in stores:
        if descriptor["kind"] == "dir":
            DirStore._validate_existing_root(descriptor["path"])
        else:
            ZipStore._validate_existing_archive(descriptor["path"])


def _duplicate_free_mapping(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate key")
        result[key] = value
    return result


def definition_from_repo(repo: Any) -> RepoDefinition:
    """Capture one Repo configuration snapshot without traversing Store data."""

    from .store.dir import DirStore
    from .store.zip import ZipStore

    with repo._configuration_lock:
        if repo._closing or repo._closed:
            raise RepoDefinitionError("Repo definition cannot be exported after close begins.")
        repo._save_context_leases += 1
        if repo._clock_explicit or repo._owner_token_factory_explicit:
            repo._save_context_leases -= 1
            raise RepoDefinitionError("Repo definition does not support explicit clock or owner-token factory.")
        try:
            connected = repo._normalize_store_handles(repo.stores, reject_physical=True)
        except (TypeError, ValueError, OSError):
            repo._save_context_leases -= 1
            raise RepoDefinitionError(
                "Repo definition Store configuration is ambiguous or invalid."
            ) from None
        stores_snapshot = tuple(connected)
        routing_snapshot = repo._save_routing
        config_snapshot = dict(repo.config)
        lease_duration = repo._lease_duration
        save_objs_on_deletion = repo.save_objs_on_deletion

    try:
        stores = []
        table: dict[int, int] = {}
        for index, store in enumerate(stores_snapshot):
            if type(store) is ZipStore:
                with store.transaction_fence():
                    path = store.archive_path
                    if (
                        store._archive_dirty
                        or path is None
                        or not os.path.isfile(path)
                        or os.path.getsize(path) == 0
                    ):
                        raise RepoDefinitionError(
                            "Repo definition store configuration has no clean committed archive."
                        )
                descriptor = {"kind": "zip", "path": os.path.abspath(path)}
            elif type(store) is DirStore:
                if not isinstance(store.query_index_policy, str) or store._query_index_config is not None:
                    raise RepoDefinitionError("Repo definition query policy is not portable.")
                descriptor = {
                    "kind": "dir",
                    "path": os.path.abspath(store.base_dir),
                    "query_index": store.query_index_policy,
                }
            else:
                raise RepoDefinitionError("Repo definition store type is not portable.")
            stores.append(descriptor)
            table[id(store)] = index
        if routing_snapshot is None:
            routing_data = None
        else:
            routes = []
            for index, (selector, store) in enumerate(routing_snapshot.routes):
                if id(store) not in table:
                    raise RepoDefinitionError("Repo definition routing destination is disconnected.")
                encoder = _SelectorEncoder()
                routes.append(
                    {
                        "selector": encoder.selector(selector, f"$.routing.routes[{index}].selector"),
                        "store": table[id(store)],
                    }
                )
            routing_data = {
                "graph_mode": routing_snapshot.graph_mode,
                "match_mode": routing_snapshot.match_mode,
                "routes": routes,
            }
        data = {
            "schema": _SCHEMA,
            "version": 1,
            "stores": stores,
            "default_store": 0 if stores else None,
            "routing": routing_data,
            "settings": {
                "config": config_snapshot,
                "lease_duration": lease_duration,
                "save_objs_on_deletion": save_objs_on_deletion,
            },
        }
        return RepoDefinition.from_data(data)
    finally:
        with repo._configuration_lock:
            repo._save_context_leases -= 1


__all__ = ["RepoDefinition", "RepoDefinitionError"]
