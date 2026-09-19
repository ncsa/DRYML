"""Detached, bounded v1 configuration records for :class:`dryml.core.Repo`.

This module deliberately represents configuration only.  Its decoding paths do
not open Stores, resolve symbols, construct selectors, or activate sessions;
connected reconstruction is a separate boundary.
"""

from __future__ import annotations

import json
import hashlib
import math
import os
import re
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from threading import RLock
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

    The error covers malformed closed-grammar data, unsupported portable values,
    and existing-authority reconstruction failures. Error text identifies a
    configuration field but intentionally never includes user-provided values,
    which can contain sensitive configuration. It has no side effects and does
    not imply that a partially reconstructed Repo is usable.
    """


class RepoReconstructionError(RepoDefinitionError):
    """Report reconstruction cleanup that still owns fresh Repo or Store handles.

    Args:
        message: Sanitized reconstruction failure summary.
        cleanup_issues: Sanitized bounded cleanup issue labels.

    The original reconstruction failure is retained as ``__cause__``. This
    error privately owns only fresh reconstruction Repos and Store handles whose
    non-flushing close failed; it never owns borrowed caller handles. :meth:`cleanup`
    closes retained Repos before their dependent Stores, serializes concurrent
    callers, and removes resources that close successfully. If ordinary close
    failures remain, it re-raises this same error with updated bounded issues.
    Control flow that derives directly from :class:`BaseException` propagates
    with this error attached as ``repo_cleanup_error`` so the caller retains
    ownership for a later retry.
    """

    _MAX_CLEANUP_ISSUES = 16

    def __init__(
            self, message: str, *, _retained_repos=(), _retained_stores=(),
            cleanup_issues=()):
        super().__init__(message)
        self._cleanup_lock = RLock()
        self._retained_repos = []
        self._retained_stores = []
        self._cleanup_issues = ()
        self._retain_cleanup(repos=_retained_repos, stores=_retained_stores)
        self._cleanup_issues = tuple(cleanup_issues[:self._MAX_CLEANUP_ISSUES])

    def _retain_cleanup(self, *, repos=(), stores=(), issues=()) -> None:
        """Add unique resource dependencies to this error's retry ownership."""

        with self._cleanup_lock:
            for repo in repos:
                if not any(existing is repo for existing in self._retained_repos):
                    self._retained_repos.append(repo)
            for store in stores:
                if not any(existing is store for existing in self._retained_stores):
                    self._retained_stores.append(store)
            self._cleanup_issues = tuple((
                *self._cleanup_issues,
                *tuple(issues),
            )[:self._MAX_CLEANUP_ISSUES])

    @property
    def cleanup_issues(self) -> tuple[str, ...]:
        """Return immutable bounded diagnostics from failed cleanup attempts.

        Returns:
            Sanitized failure type names in occurrence order. The tuple is a
            snapshot and never exposes retained Store handles.
        """

        with self._cleanup_lock:
            return self._cleanup_issues

    def cleanup(self) -> None:
        """Retry non-flushing close once for every privately retained resource.

        Side Effects:
            Closes only this error's fresh reconstruction Repos and Store handles.
            Repos close before Stores. Successful closes are removed, making
            cleanup idempotent after all resources close. Concurrent callers are
            serialized.

        Raises:
            RepoReconstructionError: This same error, if a non-control-flow
                close failure leaves any retained handle.
            BaseException: If cleanup receives control flow outside
                :class:`Exception`. This error remains attached as
                ``repo_cleanup_error`` for a later retry.
        """

        with self._cleanup_lock:
            repos = tuple(self._retained_repos)
            stores = tuple(self._retained_stores)
            remaining = []
            issues = list(self._cleanup_issues)
            retained_repos = []
            for index, repo in enumerate(repos):
                try:
                    repo.close(flush=False)
                except BaseException as error:
                    retained_repos.append(repo)
                    if len(issues) < self._MAX_CLEANUP_ISSUES:
                        issues.append(type(error).__name__)
                    if not isinstance(error, Exception):
                        retained_repos.extend(repos[index + 1:])
                        self._retained_repos = retained_repos
                        self._retained_stores = list(stores)
                        self._cleanup_issues = tuple(issues)
                        error.repo_cleanup_error = self
                        raise
            if retained_repos:
                self._retained_repos = retained_repos
                self._retained_stores = list(stores)
                self._cleanup_issues = tuple(issues)
                raise self
            for index, store in enumerate(stores):
                try:
                    store.close()
                except BaseException as error:
                    remaining.append(store)
                    if len(issues) < self._MAX_CLEANUP_ISSUES:
                        issues.append(type(error).__name__)
                    if not isinstance(error, Exception):
                        remaining.extend(stores[index + 1:])
                        self._retained_stores = remaining
                        self._cleanup_issues = tuple(issues)
                        error.repo_cleanup_error = self
                        raise
            self._retained_repos = []
            self._retained_stores = remaining
            self._cleanup_issues = tuple(issues)
            if remaining:
                raise self


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


class _SemanticBudget:
    """Bound descriptor walks independently of their compact JSON representation."""

    def __init__(self) -> None:
        self.visits = 0

    def visit(self, path: str, depth: int) -> None:
        if depth > _BOUNDS["max_depth"]:
            raise _error(path, "semantic depth bound exceeded")
        self.visits += 1
        if self.visits > _BOUNDS["max_nodes"]:
            raise _error(path, "semantic visit bound exceeded")


def _stable_leaf_hash(value: Any) -> str:
    """Return the stable-hash leaf digest for a closed JSON primitive."""

    if value is None:
        payload = b"N"
    elif type(value) is bool:
        payload = b"B1" if value else b"B0"
    elif type(value) is int:
        payload = b"I" + str(value).encode("ascii")
    elif type(value) is float:
        payload = b"F" + struct.pack(">d", value)
    elif type(value) is str:
        payload = b"S" + value.encode("utf-8")
    else:
        raise AssertionError("validated descriptor atom has unsupported type")
    return hashlib.sha256(payload).hexdigest()


def _stable_sequence_hash(type_marker: str, values: list[str]) -> str:
    hasher = hashlib.sha256()
    hasher.update(b"T" + type_marker.encode("utf-8"))
    hasher.update(b"|" + str(len(values)).encode("ascii"))
    for index, value in enumerate(values):
        hasher.update(b"I" + str(index).encode("ascii"))
        hasher.update(b"V" + value.encode("ascii"))
    return hasher.hexdigest()


def _stable_mapping_hash(type_marker: str, values: list[tuple[Any, str]]) -> str:
    hasher = hashlib.sha256()
    hasher.update(b"T" + type_marker.encode("utf-8"))
    hasher.update(b"|" + str(len(values)).encode("ascii"))
    items = sorted((_stable_leaf_hash(key), value) for key, value in values)
    for key, value in items:
        hasher.update(b"K" + key.encode("ascii"))
        hasher.update(b"V" + value.encode("ascii"))
    return hasher.hexdigest()


def _stable_set_hash(values: list[str]) -> str:
    hasher = hashlib.sha256()
    hasher.update(b"Tbuiltins.set")
    hasher.update(b"|" + str(len(values)).encode("ascii"))
    for value in sorted(values):
        hasher.update(b"V" + value.encode("ascii"))
    return hasher.hexdigest()


def _descriptor_set_fingerprint(
        value: Mapping[str, Any], labels: Mapping[str, Mapping[str, Any]],
        budget: _SemanticBudget, path: str) -> str:
    """Derive a stable set-member hash from closed data without reconstruction."""

    definitions: dict[str, str] = {}
    active: set[str] = set()

    def symbol(current: Mapping[str, Any]) -> str:
        record = current["symbol"]
        if current["representation"] == "live":
            if record["kind"] == "import":
                module, qualname = record["module"], record["qualname"]
            elif "live_class" in record:
                module, qualname = record["live_class"]["module"], record["live_class"]["qualname"]
            else:
                raise _error(path, "live source symbols cannot identify set members")
            return hashlib.sha256(
                f"class:{module}.{qualname}".encode("utf-8")
            ).hexdigest()
        if record["kind"] == "import":
            payload = json.dumps(
                {"kind": "import", "module": record["module"], "qualname": record["qualname"]},
                sort_keys=True, separators=(",", ":"),
            ).encode("utf-8")
        else:
            payload = json.dumps(
                {
                    "kind": record["source_kind"], "source": record["source"],
                    "name": record["name"],
                    "imports": {
                        name: f"{item['module']}:{item['qualname']}" if item["qualname"] is not None else item["module"]
                        for name, item in sorted(record["imports"].items())
                    },
                },
                sort_keys=True, separators=(",", ":"),
            ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def definition(label: str, depth: int) -> str:
        budget.visit(path, depth)
        existing = definitions.get(label)
        if existing is not None:
            return existing
        if label in active:
            raise _error(path, "definition cycle is unsupported")
        active.add(label)
        try:
            node = labels[label]
            cls = _stable_leaf_hash(None) if node["cls"]["kind"] == "none" else symbol(node["cls"])
            if node["node_kind"] == "cdef":
                params = _stable_mapping_hash(
                    "builtins.dict",
                    [(name, item(child, depth + 1)) for name, child in node["parameters"]],
                )
                result = _stable_mapping_hash(
                    "dryml.core.definition.ConcreteDefinition:identity-v2",
                    [("cls", cls), ("parameters", params)],
                )
            else:
                entries = [("cls", cls)]
                if node["args"] is not None:
                    entries.append(("args", _stable_sequence_hash(
                        "builtins.tuple", [item(child, depth + 1) for child in node["args"]],
                    )))
                entries.append(("kwargs", _stable_mapping_hash(
                    "builtins.dict",
                    [(name, item(child, depth + 1)) for name, child in node["kwargs"]],
                )))
                result = _stable_mapping_hash("dryml.core.definition.Definition", entries)
            definitions[label] = result
            return result
        finally:
            active.remove(label)

    def par(current: Mapping[str, Any], depth: int) -> str:
        matcher = current["matcher"]
        kind = matcher["kind"]
        if kind in {"present", "missing", "any"}:
            matcher_key = _stable_sequence_hash("builtins.tuple", [_stable_leaf_hash(kind)])
        elif kind == "exact":
            matcher_key = _stable_sequence_hash("builtins.tuple", [_stable_leaf_hash(kind), item(matcher["value"], depth + 1)])
        elif kind == "choice":
            matcher_key = _stable_sequence_hash("builtins.tuple", [_stable_leaf_hash(kind), _stable_sequence_hash("builtins.tuple", [item(child, depth + 1) for child in matcher["values"]])])
        elif kind == "int-range":
            matcher_key = _stable_sequence_hash("builtins.tuple", [_stable_leaf_hash(kind), _stable_leaf_hash(matcher["lo"]), _stable_leaf_hash(matcher["hi"])])
        else:
            matcher_key = _stable_sequence_hash("builtins.tuple", [_stable_leaf_hash(kind), symbol(matcher["cls"])])
        generator = current["generator"]
        if generator is None:
            generator_key = _stable_leaf_hash(None)
        elif generator["kind"] == "uniform-int-range":
            generator_key = _stable_sequence_hash("builtins.tuple", [_stable_leaf_hash(generator["kind"]), _stable_leaf_hash(generator["lo"]), _stable_leaf_hash(generator["hi"])])
        else:
            generator_key = _stable_sequence_hash("builtins.tuple", [_stable_leaf_hash(generator["kind"]), _stable_sequence_hash("builtins.tuple", [item(child, depth + 1) for child in generator["values"]])])
        stable_key = _stable_sequence_hash("builtins.tuple", [_stable_leaf_hash("par"), _stable_leaf_hash(current["name"]), matcher_key, generator_key])
        return _stable_mapping_hash("dryml.core.params.Par", [("stable_key", stable_key)])

    def cdef_graph_hash(root: str) -> str:
        """Recreate the token-free CDef graph digest from descriptor edges."""

        from .utils.graph.path import GraphPath, Index, Key, Parameter, SetMember, graph_path_sort_key

        edges: dict[str, list[tuple[GraphPath, str, str]]] = {}
        reached: set[str] = set()

        def direct(current: Mapping[str, Any], graph_path: GraphPath):
            kind = current["kind"]
            if kind == "definition-ref":
                yield graph_path, current["label"], "materialize"
            elif kind == "link":
                target = current["target"]
                if target["kind"] == "definition-ref":
                    yield graph_path, target["label"], current["edge"]
            elif kind in {"list", "tuple"}:
                for index, child in enumerate(current["items"]):
                    yield from direct(child, graph_path.child(Index(index)))
            elif kind == "map":
                for name, child in current["items"]:
                    yield from direct(child, graph_path.child(Key(name)))
            elif kind == "set":
                for member in current["items"]:
                    yield from direct(member["value"], graph_path.child(SetMember(member["fingerprint"])))

        def visit(label: str) -> None:
            if label in reached:
                return
            reached.add(label)
            node = labels[label]
            node_edges = []
            for name, child in node["parameters"]:
                node_edges.extend(direct(child, GraphPath((Parameter(name),))))
            edges[label] = node_edges
            for _, child, _ in node_edges:
                visit(child)

        visit(root)
        stable_hashes = {label: definition(label, 0) for label in reached}
        minimum_paths = {root: GraphPath()}
        pending = [(root, GraphPath())]
        while pending:
            parent, parent_path = pending.pop()
            for edge_path, child, _ in edges[parent]:
                candidate = parent_path.join(edge_path)
                previous = minimum_paths.get(child)
                if previous is None or graph_path_sort_key(candidate) < graph_path_sort_key(previous):
                    minimum_paths[child] = candidate
                    pending.append((child, candidate))
        ordered = sorted(reached, key=lambda label: (graph_path_sort_key(minimum_paths[label]), stable_hashes[label]))
        graph_labels = {label: f"n{index}" for index, label in enumerate(ordered)}
        node_order: list[str] = []
        seen: set[str] = set()

        def order(label: str) -> None:
            if label in seen:
                return
            seen.add(label)
            node_order.append(label)
            for _, child, _ in sorted(edges[label], key=lambda edge: (graph_path_sort_key(edge[0]), stable_hashes[edge[1]], edge[2])):
                order(child)

        order(root)
        projection = {
            "root": graph_labels[root],
            "nodes": [
                {
                    "label": graph_labels[label],
                    "stable_hash": stable_hashes[label],
                    "edges": [
                        (kind, edge_path.to_bytes().hex(), graph_labels[child])
                        for edge_path, child, kind in sorted(
                            edges[label],
                            key=lambda edge: (graph_path_sort_key(edge[0]), graph_labels[edge[1]], edge[2]),
                        )
                    ],
                }
                for label in node_order
            ],
        }
        payload = json.dumps(projection, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(b"dryml-cdef-graph-v1\x00" + payload).hexdigest()

    def object_ref(current: Mapping[str, Any]) -> str:
        graph_hash = cdef_graph_hash(current["definition"]["label"])
        from .utils.graph.path import GraphPath, graph_path_sort_key

        entries = sorted(
            current["objects"],
            key=lambda entry: graph_path_sort_key(GraphPath.from_data(entry["path"])),
        )
        payload = {
            "definition_graph": graph_hash,
            "objects": [[entry["path"], entry["object_id"]] for entry in entries],
        }
        digest = hashlib.sha256(
            b"dryml-object-ref-v1\x00" + json.dumps(
                payload, separators=(",", ":"), ensure_ascii=True,
            ).encode("ascii")
        ).hexdigest()
        return hashlib.sha256(b"dryml-object-ref-v1\x00" + digest.encode("ascii")).hexdigest()

    def item(current: Mapping[str, Any], depth: int) -> str:
        budget.visit(path, depth)
        kind = current["kind"]
        if kind == "atom":
            return _stable_leaf_hash(current["value"])
        if kind == "symbol":
            return symbol(current)
        if kind == "definition-ref":
            return definition(current["label"], depth + 1)
        if kind in {"list", "tuple"}:
            return _stable_sequence_hash(
                f"builtins.{kind}", [item(child, depth + 1) for child in current["items"]],
            )
        if kind == "set":
            return _stable_set_hash([item(member["value"], depth + 1) for member in current["items"]])
        if kind == "map":
            return _stable_mapping_hash("builtins.dict", [(name, item(child, depth + 1)) for name, child in current["items"]])
        if kind == "link":
            return _stable_mapping_hash("dryml.core.links.DefLink", [("kind", _stable_leaf_hash(current["edge"])), ("target", item(current["target"], depth + 1))])
        if kind == "quoted-definition":
            return _stable_mapping_hash("dryml.core.quoted.QuotedDef", [("value", item(current["value"], depth + 1))])
        if kind == "par":
            return par(current, depth + 1)
        if kind == "object-ref":
            return object_ref(current)
        if kind == "state-ref":
            graph_hash = cdef_graph_hash(current["object"]["definition"]["label"])
            from .utils.graph.path import GraphPath, graph_path_sort_key

            objects = sorted(
                current["object"]["objects"],
                key=lambda entry: graph_path_sort_key(GraphPath.from_data(entry["path"])),
            )
            object_identity = {
                "definition_graph": graph_hash,
                "objects": [[entry["path"], entry["object_id"]] for entry in objects],
            }
            object_digest = hashlib.sha256(
                b"dryml-object-ref-v1\x00" + json.dumps(
                    object_identity, separators=(",", ":"), ensure_ascii=True,
                ).encode("ascii")
            ).hexdigest()
            states = sorted(
                current["states"],
                key=lambda entry: graph_path_sort_key(GraphPath.from_data(entry["path"])),
            )
            state_digest = hashlib.sha256(
                b"dryml-state-ref-v1\x00" + json.dumps(
                    {"object": object_digest, "states": [[entry["path"], entry["state"]] for entry in states]},
                    separators=(",", ":"), ensure_ascii=True,
                ).encode("ascii")
            ).hexdigest()
            return hashlib.sha256(b"dryml-state-ref-v1\x00" + state_digest.encode("ascii")).hexdigest()
        raise _error(path, "set member has unsupported stable semantics")

    return item(value, 0)


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
        if representation == "live" and isinstance(value, type):
            symbol["live_class"] = {
                "module": value.__module__, "qualname": value.__qualname__,
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
    source_keys = {"kind", "source_kind", "source", "name", "imports"}
    if "live_class" in symbol:
        source_keys.add("live_class")
    _exact_keys(symbol, source_keys, path)
    if symbol["source_kind"] not in {"function", "class"} or not isinstance(symbol["source"], str) or not symbol["source"]:
        raise _error(path, "source symbol is invalid")
    if symbol["name"] is not None and not isinstance(symbol["name"], str):
        raise _error(path, "source name is invalid")
    if symbol["source_kind"] == "class" and not symbol["name"]:
        raise _error(path, "source class name is invalid")
    if "live_class" in symbol:
        if value["representation"] != "live" or not isinstance(symbol["live_class"], Mapping):
            raise _error(path, "source live class identity is invalid")
        _exact_keys(symbol["live_class"], {"module", "qualname"}, path)
        if not all(isinstance(symbol["live_class"][field], str) and symbol["live_class"][field] for field in ("module", "qualname")):
            raise _error(path, "source live class identity is invalid")
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

    def __init__(self, budget: _SemanticBudget | None = None) -> None:
        self.nodes: list[dict[str, Any]] = []
        self.labels: dict[int, str] = {}
        self.active: set[int] = set()
        self.budget = _SemanticBudget() if budget is None else budget

    def value(self, value: Any, path: str, depth: int = 0) -> dict[str, Any]:
        from .definition import ConcreteDefinition, Definition
        from .freeze import FrozenDict, FrozenList, FrozenSet, FrozenTuple
        from .links import DefLink
        from .params import Par
        from .quoted import QuotedDef, SelectorSpec
        from .reference_values import ObjectRef, StateRef
        from .selector import Selector
        from .symbol import ImportRef, SourceSpec

        self.budget.visit(path, depth)
        if isinstance(value, (Definition, ConcreteDefinition)):
            return self.definition(value, path, depth + 1)
        if isinstance(value, QuotedDef):
            return {"kind": "quoted-definition", "value": self.value(value.value, path + ".value", depth + 1)}
        if isinstance(value, SelectorSpec):
            return {
                "kind": "selector-spec",
                "selector": _SelectorEncoder(self.budget).selector(value.selector, path + ".selector", depth + 1),
            }
        if isinstance(value, Selector):
            raise _error(path, "unquoted selector is not portable")
        if isinstance(value, (ImportRef, SourceSpec)):
            return _symbol_data(value, representation="symbolic", path=path)
        if isinstance(value, DefLink):
            if not value.is_finalized:
                raise _error(path, "unresolved link assertion is not portable")
            return {"kind": "link", "edge": value.kind.value, "target": self.value(value.target, path + ".target", depth + 1)}
        if isinstance(value, ObjectRef):
            return {"kind": "object-ref", "definition": self.definition(value.definition, path + ".definition", depth + 1), "objects": value.to_data()["objects"]}
        if isinstance(value, StateRef):
            return {"kind": "state-ref", "object": self.value(value.object, path + ".object", depth + 1), "states": value.to_data()["states"]}
        if isinstance(value, Par):
            return self.par(value, path, depth + 1)
        if isinstance(value, (FrozenDict, dict)):
            from .utils.graph.path import canonical_key_bytes

            if any(type(key) not in {str, int} for key in value):
                raise _error(path, "map keys must be strings or integers")
            items = sorted(value.items(), key=lambda item: canonical_key_bytes(item[0]))
            return {"kind": "map", "items": [[key, self.value(item, f"{path}.items[{index}]", depth + 1)] for index, (key, item) in enumerate(items)]}
        if isinstance(value, (FrozenList, list)):
            return {"kind": "list", "items": [self.value(item, f"{path}.items[{index}]", depth + 1) for index, item in enumerate(value)]}
        if isinstance(value, (FrozenTuple, tuple)):
            return {"kind": "tuple", "items": [self.value(item, f"{path}.items[{index}]", depth + 1) for index, item in enumerate(value)]}
        if isinstance(value, (FrozenSet, set, frozenset)):
            from .utils.stable_hash import stable_hash_function

            # Label assignment is global, so order raw members before traversing
            # them with this encoder.  The probe has isolated labels and cannot
            # merge or alter the topology retained by the final traversal.
            try:
                ordered = sorted(
                    value,
                    key=lambda item: canonical_json_dumps(
                        _SelectorEncoder(self.budget).value(item, path + ".set", depth + 1), **_BOUNDS
                    ),
                )
            except (RecursionError, TypeError, ValueError, OverflowError, UnicodeError):
                raise _error(path, "set member is not portable") from None
            members = []
            for item in ordered:
                encoded = self.value(item, path + ".set", depth + 1)
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

    def definition(self, value: Any, path: str, depth: int = 0) -> dict[str, Any]:
        from .definition import ConcreteDefinition
        self.budget.visit(path, depth)
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
                node["parameters"] = [[name, self.value(item, f"{path}.parameters[{index}]", depth + 1)] for index, (name, item) in enumerate(value.parameters.items())]
                node["stateful_role"] = value._stateful_role
            else:
                node["args"] = None if value.args is None else [self.value(item, f"{path}.args[{index}]", depth + 1) for index, item in enumerate(value.args)]
                node["kwargs"] = [[name, self.value(item, f"{path}.kwargs[{index}]", depth + 1)] for index, (name, item) in enumerate(value.kwargs.items())]
            self.nodes.append(node)
            return {"kind": "definition-ref", "label": label}
        finally:
            self.active.remove(key)

    def par(self, value: Any, path: str, depth: int = 0) -> dict[str, Any]:
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
            match = {"kind": "exact", "value": self.value(matcher.value, path + ".exact", depth + 1)}
        elif type(matcher) is ChoiceMatcher:
            match = {"kind": "choice", "values": [self.value(item, f"{path}.choice[{index}]", depth + 1) for index, item in enumerate(matcher.values)]}
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
            gen = {"kind": "uniform-from-set", "values": [self.value(item, f"{path}.generator[{index}]", depth + 1) for index, item in enumerate(generator.values)]}
        else:
            raise _error(path, "generator is not portable")
        return {"kind": "par", "name": value.name, "matcher": match, "generator": gen}

    def selector(self, selector: Any, path: str, depth: int = 0) -> dict[str, Any]:
        return {"root": self.definition(selector.root, path + ".root", depth + 1), "strict": selector.strict, "cls_policy": selector.cls_policy, "nodes": self.nodes}


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


def _validate_reference_graph_path(value: Any, path: str) -> None:
    """Validate the canonical closed GraphPath form used inside references."""

    from .utils.graph.path import GRAPH_PATH_SCHEMA_VERSION

    record = _exact_keys(value, {"schema_version", "segments"}, path)
    if record["schema_version"] != GRAPH_PATH_SCHEMA_VERSION or not isinstance(record["segments"], list):
        raise _error(path, "reference path is invalid")
    for index, segment in enumerate(record["segments"]):
        segment_path = f"{path}.segments[{index}]"
        if not isinstance(segment, Mapping) or not isinstance(segment.get("kind"), str):
            raise _error(segment_path, "reference path segment is invalid")
        kind = segment["kind"]
        if kind in {"parameter", "kwarg"}:
            _exact_keys(segment, {"kind", "name"}, segment_path)
            if not isinstance(segment["name"], str):
                raise _error(segment_path, "reference path segment is invalid")
        elif kind in {"arg", "index"}:
            _exact_keys(segment, {"kind", "index"}, segment_path)
            if type(segment["index"]) is not int or segment["index"] < 0:
                raise _error(segment_path, "reference path segment is invalid")
        elif kind == "key":
            _exact_keys(segment, {"kind", "value"}, segment_path)
            if type(segment["value"]) not in {str, int}:
                raise _error(segment_path, "reference path segment is invalid")
        elif kind == "set_member":
            _exact_keys(segment, {"kind", "fingerprint", "ordinal"}, segment_path)
            if (
                    not isinstance(segment["fingerprint"], str)
                    or type(segment["ordinal"]) is not int
                    or segment["ordinal"] < 0):
                raise _error(segment_path, "reference path segment is invalid")
        else:
            raise _error(segment_path, "reference path segment is invalid")


def _validate_selector(value: Any, path: str) -> None:
    record = _exact_keys(value, {"root", "strict", "cls_policy", "nodes"}, path)
    if type(record["strict"]) is not bool or record["cls_policy"] not in _CLASS_POLICIES:
        raise _error(path, "selector policy is invalid")
    if not isinstance(record["nodes"], list) or len(record["nodes"]) > 65536:
        raise _error(path, "selector nodes are invalid")

    labels: dict[str, Mapping[str, Any]] = {}
    budget = _SemanticBudget()
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

    def definition_ref(
            current: Any, item_path: str, depth: int = 0,
            *, cdef: bool | None = None) -> str:
        budget.visit(item_path, depth)
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
                item(child, item_path, depth + 1)
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
                _validate_reference_graph_path(entry["path"], entry_path + ".path")
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
            if len(expected) >= _BOUNDS["max_entries"] and key not in expected:
                raise _error(item_path, "reference topology entry bound exceeded")
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

        def visit_value(value: Mapping[str, Any], graph_path: GraphPath, value_path: str, active_labels: set[str], depth: int) -> None:
            budget.visit(value_path, depth)
            kind = value["kind"]
            if kind == "definition-ref":
                label = value["label"]
                if labels[label]["node_kind"] != "cdef":
                    raise _error(value_path, "CDef graph contains a partial Definition")
                visit_cdef(label, graph_path, value_path, active_labels, depth + 1)
                return
            if kind in {"object-ref", "state-ref"}:
                visit_reference(value, graph_path, value_path)
                return
            if kind == "link":
                if value["edge"] == "materialize":
                    visit_value(value["target"], graph_path, value_path + ".target", active_labels, depth + 1)
                return
            if kind in {"list", "tuple"}:
                for index, child in enumerate(value["items"]):
                    visit_value(child, graph_path.child(Index(index)), f"{value_path}[{index}]", active_labels, depth + 1)
                return
            if kind == "map":
                for index, pair in enumerate(value["items"]):
                    visit_value(pair[1], graph_path.child(Key(pair[0])), f"{value_path}[{index}]", active_labels, depth + 1)
                return
            if kind == "set":
                for member in value["items"]:
                    visit_value(
                        member["value"],
                        graph_path.child(SetMember(member["fingerprint"])),
                        value_path + ".set",
                        active_labels, depth + 1,
                    )
                return
            if kind == "par":
                raise _error(value_path, "CDef graph contains a parameter placeholder")

        def visit_cdef(label: str, graph_path: GraphPath, value_path: str, active_labels: set[str], depth: int) -> None:
            budget.visit(value_path, depth)
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
                    active_labels, depth + 1,
                )
            active_labels.remove(label)

        visit_cdef(root_label, GraphPath(), item_path + ".definition", set(), 0)
        expected_paths = {path for path, _ in expected.values()}
        if set(objects) != expected_paths:
            raise _error(item_path, "reference paths do not match CDef topology")
        for key, (graph_path, object_id) in expected.items():
            if object_id is not None and objects[graph_path] != object_id:
                raise _error(item_path, "reference identity does not match imported topology")

    def item(current: Any, item_path: str, depth: int = 0) -> None:
        budget.visit(item_path, depth)
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
            definition_ref(current, item_path, depth + 1)
            return
        if kind == "quoted-definition":
            _exact_keys(current, {"kind", "value"}, item_path)
            item(current["value"], item_path + ".value", depth + 1)
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
                item(child, f"{item_path}[{index}]", depth + 1)
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
                item(member["value"], member_path + ".value", depth + 1)
                encoded = _bounded_json_bytes(member["value"], member_path + ".value")
                if encoded in canonical_members:
                    raise _error(item_path, "set contains duplicate canonical values")
                canonical_members.add(encoded)
                actual = _descriptor_set_fingerprint(
                    member["value"], labels, budget, member_path + ".value",
                )
                if member["fingerprint"] != actual:
                    raise _error(member_path, "set fingerprint does not match member semantics")
                if member["fingerprint"] in fingerprints:
                    raise _error(item_path, "set contains ambiguous stable members")
                fingerprints.add(member["fingerprint"])
            return
        if kind == "map":
            _exact_keys(current, {"kind", "items"}, item_path)
            if not isinstance(current["items"], list) or len(current["items"]) > 4096:
                raise _error(item_path, "map is invalid")
            names = set()
            for index, pair in enumerate(current["items"]):
                pair_path = f"{item_path}[{index}]"
                if not isinstance(pair, list) or len(pair) != 2 or type(pair[0]) not in {str, int}:
                    raise _error(pair_path, "map entry is invalid")
                key = pair[0]
                if key in names:
                    raise _error(pair_path, "map key is duplicated")
                names.add(key)
                item(pair[1], pair_path, depth + 1)
            return
        if kind == "link":
            _exact_keys(current, {"kind", "edge", "target"}, item_path)
            if current["edge"] not in {"ref", "materialize"}:
                raise _error(item_path, "link edge is invalid")
            item(current["target"], item_path + ".target", depth + 1)
            return
        if kind == "par":
            _validate_par(current, item_path, lambda child, child_path: item(child, child_path, depth + 1))
            return
        if kind == "object-ref":
            _exact_keys(current, {"kind", "definition", "objects"}, item_path)
            definition_ref(current["definition"], item_path + ".definition", depth + 1, cdef=True)
            reference_entries(current["objects"], item_path + ".objects", state=False)
            validate_object_topology(current, item_path)
            return
        if kind == "state-ref":
            _exact_keys(current, {"kind", "object", "states"}, item_path)
            if not isinstance(current["object"], Mapping) or current["object"].get("kind") != "object-ref":
                raise _error(item_path, "state reference object is invalid")
            item(current["object"], item_path + ".object", depth + 1)
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
        store = _validate_store_descriptor(store, f"$.stores[{index}]")
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


def _validate_store_descriptor(value: Any, path: str = "$") -> dict[str, Any]:
    """Validate and detach one closed portable existing-Store descriptor.

    Args:
        value: Mapping describing a supported existing ``DirStore`` or path-backed
            ``ZipStore``.
        path: Sanitized descriptor location used only in validation diagnostics.

    Returns:
        A detached JSON-compatible descriptor.

    Raises:
        RepoDefinitionError: If the descriptor's kind, fields, settings, or path
            is unsupported. This function never opens or creates Store authority.
    """

    record = _exact_keys(
        value,
        {"kind", "path", "query_index"}
        if isinstance(value, Mapping) and value.get("kind") == "dir"
        else {"kind", "path"},
        path,
    )
    if record["kind"] == "dir":
        if (
                not isinstance(record["query_index"], str)
                or record["query_index"] not in {"auto", "sqlite", "memory", "none"}):
            raise _error(path, "query policy is invalid")
    elif record["kind"] != "zip":
        raise _error(path, "store kind is unsupported")
    if not isinstance(record["path"], str) or not os.path.isabs(record["path"]):
        raise _error(path, "store path must be absolute")
    return json.loads(_bounded_json_bytes(record, path))


@dataclass(frozen=True, slots=True, init=False)
class RepoDefinition:
    """An immutable, inert v1 Repo configuration snapshot.

    Args:
        data: Complete JSON-compatible v1 envelope containing supported Store,
            routing, and declarative-setting descriptors.

    ``to_data`` and ``from_data`` only inspect detached descriptors. They never
    open persistent resources, resolve symbols, activate a session, materialize
    Objects, or construct a live Repo. Use :meth:`Repo.from_definition` for the
    explicit existing-Store reconstruction boundary. Retained data is detached
    and bounded; callers still own transport and any sensitive configuration.
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
    if symbol["kind"] == "source" and "live_class" in symbol and not require_live:
        # SourceSpec equality preserves the original live-class identity for
        # default, exact, and strict Selector matching after reconstruction.
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
            return DefLink.finalized(EdgeKind(current["edge"]), item(current["target"]))
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
    Outside an active Session cache, the returned Repo owns fresh Store handles.
    With caching, a semantic Repo hit is reused and a completed miss is published
    atomically with cache-owned borrowed Stores; any failure closes only newly
    staged misses without committing buffered state.
    """

    if not isinstance(definition, RepoDefinition):
        raise TypeError("Repo.from_definition requires a RepoDefinition.")
    # Revalidate a detached copy so live reconstruction never trusts a retained
    # implementation detail or a future subclass's mutable backing object.
    data = _validate_data(definition.to_data())
    routing_data = data["routing"]
    try:
        _preflight_store_descriptors(data["stores"])
        _validate_live_store_identities(data["stores"])
        routing_parts = None if routing_data is None else _reconstruct_routing(routing_data)
    except (KeyboardInterrupt, SystemExit):
        raise
    except RepoDefinitionError:
        raise
    except Exception as error:
        raise RepoDefinitionError("Repo definition reconstruction preflight failed.") from error

    from . import session as core_session

    cache = core_session._current_resource_cache()
    if cache is not None:
        key = _repo_cache_key_from_data(data)
        return cache._acquire_repo(
            key, lambda: _reconstruct_repo_from_data(data, routing_parts, cache_active=True),
        )
    return _reconstruct_repo_from_data(data, routing_parts, cache_active=False)


def _reconstruct_repo_from_data(data: Mapping[str, Any], routing_parts, *, cache_active: bool):
    """Construct one Repo after preflight, borrowing active-cache Store handles."""

    from .repo import Repo
    from .store.dir import DirStore
    from .store.zip import ZipStore
    opened = []
    owned = []
    repo = None
    try:
        for descriptor in data["stores"]:
            if descriptor["kind"] == "dir":
                store = DirStore.open_existing(
                    descriptor["path"], query_index=descriptor["query_index"],
                )
            else:
                store = ZipStore.open_existing(descriptor["path"])
            opened.append(store)
            if not cache_active:
                owned.append(store)
        if routing_parts is None:
            routing = None
        else:
            selectors, store_indexes, routing_data = routing_parts
            from .repo_plan import SaveRouting
            routing = SaveRouting(
                tuple((selector, opened[index]) for selector, index in zip(selectors, store_indexes)),
                routing_data["match_mode"], routing_data["graph_mode"],
            )
        repo = Repo.__new__(Repo)
        Repo.__init__(
            repo,
            opened,
            config=data["settings"]["config"],
            lease_duration=data["settings"]["lease_duration"],
            save_routing=routing,
        )
        repo.save_objs_on_deletion = data["settings"]["save_objs_on_deletion"]
        repo._adopt_owned_stores(owned)
        return repo
    except BaseException as error:
        issues = []
        cleanup_control_flow = None
        retained_repos = []
        if repo is not None:
            # This Repo is not returned. Prevent its destructor from publishing
            # cached Objects while reconstruction cleanup unwinds.
            repo_owned = tuple(repo._owned_stores)
            repo._closing = True
            try:
                repo.close(flush=False)
            except BaseException as cleanup_error:
                repo._closing = True
                retained_repos.append(repo)
                issues.append(type(cleanup_error).__name__)
                if not isinstance(cleanup_error, Exception):
                    cleanup_control_flow = cleanup_error
            # Stores adopted before failure remain under the Repo's one retry
            # owner. A hook can fail before adoption, leaving only those handles
            # for this reconstruction path to close directly.
            owned = [
                store for store in owned
                if not any(store is adopted for adopted in repo_owned)
            ]
        failed_stores = []
        for store in reversed(owned):
            try:
                store.close()
            except BaseException as cleanup_error:
                failed_stores.append(store)
                issues.append(type(cleanup_error).__name__)
                if (
                        cleanup_control_flow is None
                        and not isinstance(cleanup_error, Exception)):
                    cleanup_control_flow = cleanup_error
        cleanup_error = None
        if retained_repos or failed_stores:
            cleanup_error = RepoReconstructionError(
                "Repo definition reconstruction cleanup requires retry.",
                _retained_repos=retained_repos,
                _retained_stores=failed_stores,
                cleanup_issues=issues,
            )
        if cleanup_control_flow is not None:
            if cleanup_error is not None:
                cleanup_control_flow.repo_cleanup_error = cleanup_error
            raise cleanup_control_flow
        if not isinstance(error, Exception):
            if cleanup_error is not None:
                error.repo_cleanup_error = cleanup_error
            raise
        if cleanup_error is not None:
            raise cleanup_error from error
        if isinstance(error, RepoDefinitionError):
            raise
        raise RepoDefinitionError("Repo definition could not open required Store authority.") from error


def _repo_cache_key_from_data(data: Mapping[str, Any]) -> tuple[Any, ...]:
    """Return the complete current semantic key for one validated Repo request."""

    stores = tuple(_store_cache_key_from_descriptor(store) for store in data["stores"])
    try:
        routing = canonical_json_bytes(data["routing"], **_BOUNDS)
        config = canonical_json_bytes(data["settings"]["config"], **_BOUNDS)
    except (RecursionError, TypeError, ValueError, OverflowError, UnicodeError):
        raise RepoDefinitionError("Repo definition cache key is invalid.") from None
    return (
        "repo", stores, data["default_store"], routing, config,
        float(data["settings"]["lease_duration"]),
        data["settings"]["save_objs_on_deletion"],
    )


def _repo_cache_key_from_repo(repo: Any) -> tuple[Any, ...] | None:
    """Snapshot a live Repo for matching without exporting buffered Store state.

    The snapshot intentionally uses live physical evidence instead of portable
    Store export, so a dirty ZipStore can remain the same cache transaction while
    no longer being eligible for transport.
    """

    with repo._configuration_lock:
        if repo._closing or repo._closed:
            return None
        try:
            stores = tuple(repo._normalize_store_handles(repo.stores, reject_physical=True))
            store_keys = tuple(_store_cache_key_from_store(store) for store in stores)
            if any(key is None for key in store_keys):
                return None
            table = {id(store): index for index, store in enumerate(stores)}
            if repo._save_routing is None:
                routing = None
            else:
                routes = []
                for index, (selector, store) in enumerate(repo._save_routing.routes):
                    if id(store) not in table:
                        return None
                    routes.append({
                        "selector": _SelectorEncoder().selector(
                            selector, f"$.routing.routes[{index}].selector",
                        ),
                        "store": table[id(store)],
                    })
                routing = {
                    "graph_mode": repo._save_routing.graph_mode,
                    "match_mode": repo._save_routing.match_mode,
                    "routes": routes,
                }
            config = canonical_json_bytes(repo.config, **_BOUNDS)
            routing_bytes = canonical_json_bytes(routing, **_BOUNDS)
            lease_duration = float(repo._lease_duration)
            deletion_save = repo.save_objs_on_deletion
        except (RecursionError, TypeError, ValueError, OverflowError, UnicodeError, OSError):
            return None
    if type(deletion_save) is not bool:
        return None
    return (
        "repo", store_keys, 0 if store_keys else None, routing_bytes, config,
        lease_duration, deletion_save,
    )


def _reconstruct_routing(data: Mapping[str, Any]):
    """Construct validated live selectors and bind their table destinations."""

    # Store indexes remain inert until the caller binds them to fresh handles.
    selectors = tuple(_selector_from_data(route["selector"]) for route in data["routes"])
    return (selectors, tuple(route["store"] for route in data["routes"]), data)


def _preflight_store_descriptors(stores: list[Mapping[str, Any]]) -> None:
    """Validate every required persistent path/type before opening any Store."""

    for descriptor in stores:
        _preflight_store_descriptor(descriptor)


def _preflight_store_descriptor(descriptor: Mapping[str, Any]) -> None:
    """Validate existing authority for one already-validated Store descriptor."""

    from .store.dir import DirStore
    from .store.zip import ZipStore

    if descriptor["kind"] == "dir":
        DirStore._validate_existing_root(descriptor["path"])
    else:
        ZipStore._validate_existing_archive(descriptor["path"])


def _store_cache_key_from_descriptor(descriptor: Mapping[str, Any]) -> tuple[Any, ...]:
    """Return the physical identity and opening settings for an existing request."""

    _preflight_store_descriptor(descriptor)
    try:
        evidence = os.stat(descriptor["path"])
    except OSError as error:
        raise RepoDefinitionError("Store definition could not inspect required authority.") from error
    if descriptor["kind"] == "dir":
        return ("dir", evidence.st_dev, evidence.st_ino, descriptor["query_index"])
    return ("zip", evidence.st_dev, evidence.st_ino)


def _store_cache_key_from_store(store: Any) -> tuple[Any, ...] | None:
    """Return live Store cache evidence without exporting or committing buffers."""

    from .store.dir import DirStore
    from .store.zip import ZipStore

    if type(store) is DirStore:
        if not isinstance(store.query_index_policy, str) or store._query_index_config is not None:
            return None
        try:
            evidence = os.stat(store.base_dir)
        except OSError:
            return None
        if (evidence.st_dev, evidence.st_ino) != getattr(store, "_authority_evidence", None):
            return None
        return ("dir", evidence.st_dev, evidence.st_ino, store.query_index_policy)
    if type(store) is ZipStore and store.archive_path is not None:
        if getattr(store, "_closed_handle", False):
            return None
        try:
            evidence = os.stat(store.archive_path)
        except OSError:
            return None
        if (evidence.st_dev, evidence.st_ino) != getattr(store, "_archive_evidence", None):
            return None
        return ("zip", evidence.st_dev, evidence.st_ino)
    return None


def _open_store_descriptor(definition: Mapping[str, Any]):
    """Open one existing Store descriptor through the active Session cache, if any."""

    descriptor = _validate_store_descriptor(definition)
    key = _store_cache_key_from_descriptor(descriptor)

    def open_fresh():
        from .store.dir import DirStore
        from .store.zip import ZipStore

        if descriptor["kind"] == "dir":
            return DirStore(
                descriptor["path"], query_index=descriptor["query_index"], _existing_only=True,
            )
        return ZipStore(
            descriptor["path"], _existing_only=True, _authority_prevalidated=True,
        )

    from . import session as core_session

    cache = core_session._current_resource_cache()
    if cache is None:
        return open_fresh()
    return cache._acquire_store(key, open_fresh)


def _resource_cache_retains_store(store: Any) -> bool:
    """Return whether the active cache retains the Store lifetime for this request."""

    from . import session as core_session

    cache = core_session._current_resource_cache()
    return cache is not None and cache._retains_store(store)


def _validate_live_store_identities(stores: list[Mapping[str, Any]]) -> None:
    """Reject equivalent existing Store destinations before opening any handles."""

    identities = set()
    for descriptor in stores:
        if descriptor["kind"] == "dir":
            evidence = os.stat(descriptor["path"])
            identity = ("dir", evidence.st_dev, evidence.st_ino)
        else:
            identity = ("zip", os.path.normcase(os.path.realpath(descriptor["path"])))
        if identity in identities:
            raise RepoDefinitionError(
                "Repo definition names duplicate physical Store destinations."
            )
        identities.add(identity)


def _duplicate_free_mapping(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate key")
        result[key] = value
    return result


def definition_from_repo(repo: Any) -> RepoDefinition:
    """Capture one Repo configuration snapshot without traversing Store data."""

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
            descriptor = definition_from_store(store)
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


def definition_from_store(store: Any) -> dict[str, Any]:
    """Export one Store's detached portable descriptor without committing it.

    Args:
        store: Supported live ``DirStore`` or clean path-backed ``ZipStore``.

    Returns:
        A detached closed descriptor containing backend kind, absolute location,
        and supported opening settings.

    Raises:
        RepoDefinitionError: If the Store type, opening settings, or archive
            transaction cannot be transported. This function never saves or commits.
    """

    from .store.dir import DirStore
    from .store.zip import ZipStore

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
        return {"kind": "zip", "path": os.path.abspath(path)}
    if type(store) is DirStore:
        if not isinstance(store.query_index_policy, str) or store._query_index_config is not None:
            raise RepoDefinitionError("Repo definition query policy is not portable.")
        return {
            "kind": "dir",
            "path": os.path.abspath(store.base_dir),
            "query_index": store.query_index_policy,
        }
    raise RepoDefinitionError("Repo definition store type is not portable.")


__all__ = ["RepoDefinition", "RepoDefinitionError", "RepoReconstructionError"]
