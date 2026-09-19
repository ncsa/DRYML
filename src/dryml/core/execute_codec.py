"""Bounded, worker-local transport for one core-aware execution call.

The codec deliberately transports authority records instead of live core resources.
It is internal to :mod:`dryml.core.execute`: its byte representation is version-local
and is not a Store or network interchange format.
"""

from __future__ import annotations

import builtins
import ast
import dis
import inspect
import pathlib
import sys
import types
from collections.abc import Mapping
from typing import Any

import dill

from .cdef_codec import decode_cdef_graph, encode_cdef_graph
from .cdef_graph import has_stateful_materialization
from .definition import ConcreteDefinition
from .links import DefLink
from .object import Object
from .reference_values import ObjectRef, StateRef
from .repo import Repo
from .store.store import Store
from .signatures import ReferenceSelection, SignatureError, compile_signature
from .symbol import ImportRef
from ._callable_inspection import describe_callable


_VERSION = 2
_DEFAULT_LIMIT = 67_108_864
_MAX_NODES = 65_536
_MAX_DEPTH = 64
_OUTCOME_VERSION = 1
_PATH_TYPES = {
    cls.__name__: cls
    for cls in (
        pathlib.PosixPath,
        pathlib.WindowsPath,
        pathlib.PurePosixPath,
        pathlib.PureWindowsPath,
    )
}


class CoreCallCodecError(TypeError):
    """Raised when a call graph cannot safely cross the core execution boundary.

    The exception names only a structural path and category. It never includes a
    caller value representation, Store location, or serialized payload.
    """


def _fail(reason: str, path: str) -> None:
    raise CoreCallCodecError(f"core execution transport rejected {reason} at {path}")


def _dill_code(value: types.CodeType, path: str) -> bytes:
    """Encode only native code objects after structural capture has lowered data."""
    try:
        return dill.dumps(value, protocol=5, byref=False, recurse=False)
    except Exception as error:
        raise CoreCallCodecError(f"core execution transport rejected function code at {path}") from error


def _dill_result(value: Any, path: str) -> bytes:
    """Encode an already recursively resource-free ordinary result."""
    try:
        return dill.dumps(value, protocol=5, byref=False, recurse=True)
    except Exception as error:
        raise CoreCallCodecError(f"core execution transport rejected ordinary result at {path}") from error


def _result_graph(value: Any, *, limit_bytes: int,
                  automatic_references: set[int]) -> bytes:
    """Encode an already-published result graph with the call graph grammar."""
    encoder = _Encoder(limit_bytes=limit_bytes, automatic_references=automatic_references)
    return encoder.finish(encoder.value(value, "$.result"))


def _load_result_graph(data: bytes, *, repo: Repo, limit_bytes: int) -> tuple[Any, frozenset[int]]:
    """Decode one result graph after its outcome envelope has been validated."""
    decoder = _Decoder(data, repo=repo, limit_bytes=limit_bytes)
    return decoder.value(decoder.graph["root"], "$.result"), frozenset(decoder.automatic_references)


def _outcome(success: bool, *, result: bytes | None = None,
             updates: list[dict[str, Any]] | None = None,
             publications: list[dict[str, Any]] | None = None,
             reason: str | None = None, limit_bytes: int) -> bytes:
    """Encode one bounded tagged worker outcome without live core values."""
    value = {
        "version": _OUTCOME_VERSION,
        "tag": "core-outcome",
        "success": success,
        "result": result,
        "updates": updates or [],
        "publications": publications or [],
        "reason": reason,
    }
    try:
        encoded = dill.dumps(value, protocol=5, byref=False, recurse=True)
    except Exception as error:
        raise CoreCallCodecError("core execution transport could not encode result outcome") from error
    if len(encoded) > limit_bytes:
        # Publication/update evidence is authority already made durable; only the
        # result graph may be dropped after a completed publication.
        value["success"] = False
        value["result"] = None
        value["reason"] = "result outcome exceeds configured bound after publication"
        encoded = dill.dumps(value, protocol=5, byref=False, recurse=True)
        if len(encoded) > limit_bytes:
            raise CoreCallCodecError("core execution transport rejected oversized result outcome evidence")
    return encoded


def _failure_reason(error: Exception) -> str:
    """Project expected owner failures into a bounded transport-safe category."""
    if isinstance(error, SignatureError):
        return f"SignatureError:{error.reason}" + ("" if error.slot is None else f":{error.slot}")
    return type(error).__name__


def decode_outcome(data: bytes, *, repo: Repo, limit_bytes: int = _DEFAULT_LIMIT) -> dict[str, Any]:
    """Decode one closed worker outcome into portable result and update authority.

    The returned mapping contains only ordinary values, immutable references, and
    Store-table-relative publication facts. It never reconstructs a StoreReport.
    """
    if not isinstance(data, bytes) or len(data) > limit_bytes:
        raise CoreCallCodecError("core execution transport rejected oversized result")
    value = _load(data, "$.outcome")
    fields = {"version", "tag", "success", "result", "updates", "publications", "reason"}
    if not isinstance(value, Mapping) or set(value) != fields or value.get("version") != _OUTCOME_VERSION or value.get("tag") != "core-outcome":
        raise CoreCallCodecError("core execution transport rejected malformed result outcome")
    if type(value["success"]) is not bool or not isinstance(value["updates"], list) or not isinstance(value["publications"], list):
        raise CoreCallCodecError("core execution transport rejected malformed result outcome")
    if value["reason"] is not None and not isinstance(value["reason"], str):
        raise CoreCallCodecError("core execution transport rejected malformed result outcome")
    if value["success"]:
        if not isinstance(value["result"], bytes):
            raise CoreCallCodecError("core execution transport rejected malformed successful outcome")
        value = dict(value)
        value["result"], value["automatic_references"] = _load_result_graph(
            value["result"], repo=repo, limit_bytes=limit_bytes,
        )
    elif value["result"] is not None:
        raise CoreCallCodecError("core execution transport rejected malformed failed outcome")
    return value


def _load(data: Any, path: str) -> Any:
    if not isinstance(data, bytes):
        _fail("malformed dill payload", path)
    try:
        return dill.loads(data)
    except Exception as error:
        raise CoreCallCodecError(f"core execution transport rejected malformed dill payload at {path}") from error


def _import_ref(value: Any) -> ImportRef | None:
    """Return a reference without importing a module in the coordinator."""
    module_name = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", None)
    if not isinstance(module_name, str) or not isinstance(qualname, str) or "<locals>" in qualname:
        return None
    module = sys.modules.get(module_name)
    if module is None:
        return None
    candidate: Any = module
    try:
        for part in qualname.split("."):
            candidate = getattr(candidate, part)
    except Exception:
        return None
    return ImportRef(module_name, qualname) if candidate is value else None


def _annotation_names(annotations: Mapping[str, Any]) -> set[str]:
    """Collect raw future-annotation names without evaluating their expressions."""
    names: set[str] = set()
    for annotation in annotations.values():
        if not isinstance(annotation, str):
            continue
        try:
            names.update(node.id for node in ast.walk(ast.parse(annotation, mode="eval")) if isinstance(node, ast.Name))
        except SyntaxError:
            continue
    return names


def _missing_global_names(function: types.FunctionType, unbound: set[str]) -> set[str]:
    """Separate true missing globals from attribute names reported as unbound."""
    return {
        str(instruction.argval)
        for instruction in dis.get_instructions(function)
        if instruction.opname in {"LOAD_GLOBAL", "LOAD_NAME"} and instruction.argval in unbound
    }


def _is_resource(value: Any) -> bool:
    """Return whether a value is a live resource rather than portable value data."""
    return isinstance(value, (Repo, Store))


def _instance_fields(value: Any) -> dict[str, Any] | None:
    """Return declared instance fields, including slots, without invoking hooks."""
    fields: dict[str, Any] = {}
    mapping = getattr(value, "__dict__", None)
    if isinstance(mapping, Mapping):
        fields.update(mapping)
    for cls in type(value).__mro__:
        slots = cls.__dict__.get("__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        if not isinstance(slots, tuple):
            continue
        for name in slots:
            if isinstance(name, str) and name not in {"__dict__", "__weakref__"} and hasattr(value, name):
                fields[name] = getattr(value, name)
    return fields if fields or hasattr(value, "__dict__") or any(
        "__slots__" in cls.__dict__ for cls in type(value).__mro__
    ) else None


def _class_has_resource(value: type) -> bool:
    """Reject import shortcuts whose class attributes retain a live resource."""
    return any(_is_resource(item) or any(_is_resource(field) for field in (_instance_fields(item) or {}).values())
               for item in value.__dict__.values())


class _Encoder:
    """Encode one whole call graph while preserving aliases and rejecting cycles."""

    def __init__(self, *, limit_bytes: int,
                 automatic_references: set[int] | None = None) -> None:
        self.nodes: list[dict[str, Any]] = []
        self.memo: dict[int, int] = {}
        self.active: set[int] = set()
        self.limit_bytes = limit_bytes
        self.automatic_references = automatic_references or set()
        self.imported_captures: dict[int, ImportRef] = {}

    def _capture(self, value: Any, path: str, depth: int) -> int:
        """Import stable DRYML dependencies and capture caller helpers by value."""
        module_name = getattr(value, "__module__", None)
        if (
            inspect.isfunction(value)
            and isinstance(module_name, str)
            and (module_name == "dryml" or module_name.startswith("dryml."))
        ):
            ref = _import_ref(value)
            if ref is not None:
                self.imported_captures[id(value)] = ref
        return self.value(value, path, depth)

    def value(self, value: Any, path: str, depth: int = 0) -> int:
        if depth > _MAX_DEPTH:
            _fail("graph depth exceeds the transport bound", path)
        if len(self.nodes) >= _MAX_NODES:
            _fail("graph node count exceeds the transport bound", path)
        identity = id(value)
        if identity in self.active:
            _fail("cyclic graph", path)
        if identity in self.memo:
            return self.memo[identity]
        index = len(self.nodes)
        self.memo[identity] = index
        self.nodes.append({})
        self.active.add(identity)
        try:
            self.nodes[index] = self._node(value, path, depth)
        finally:
            self.active.remove(identity)
        return index

    def _node(self, value: Any, path: str, depth: int) -> dict[str, Any]:
        from dryml.managed.config import ManagedConfig
        from dryml.managed.descriptor import (
            ManagedOperation, _BoundComposite, _BoundOperation, _ManagedComposite,
        )

        if type(value) is ManagedConfig:
            return self._managed_config_node(value, path, depth)
        if type(value) is ManagedOperation:
            owner = value._owner
            member = value.member
            if not isinstance(owner, type) or not isinstance(member, str):
                _fail("unbound managed declaration", path)
            return {
                "tag": "managed_declaration",
                "authored": self.value(value._target, f"{path}.authored", depth + 1),
                "executable": self.value(value._executable, f"{path}.executable", depth + 1),
                "owner": self.value(owner, f"{path}.owner", depth + 1),
                "member": member,
                "resumable": value.resumable,
            }
        if type(value) is _ManagedComposite:
            descriptor = value._descriptor
            owner = descriptor._owner
            member = descriptor.member
            if not isinstance(owner, type) or not isinstance(member, str):
                _fail("unbound managed composite", path)
            return {
                "tag": "managed_composite",
                "declaration": self.value(descriptor, f"{path}.declaration", depth + 1),
                "outer": self.value(value._outer, f"{path}.outer", depth + 1),
                "owner": self.value(owner, f"{path}.owner", depth + 1),
                "member": member,
            }
        if type(value) is _BoundOperation:
            return {
                "tag": "managed_target",
                "receiver": self.value(value._instance, f"{path}.receiver", depth + 1),
                "declaration": self.value(value._descriptor, f"{path}.declaration", depth + 1),
            }
        if type(value) is _BoundComposite:
            return {
                "tag": "managed_composite_target",
                "receiver": self.value(value._instance, f"{path}.receiver", depth + 1),
                "composite": self.value(value._composite, f"{path}.composite", depth + 1),
            }
        if _is_resource(value):
            _fail("live core resource", path)
        imported_capture = self.imported_captures.get(id(value))
        if imported_capture is not None:
            return {
                "tag": "import", "module": imported_capture.module,
                "qualname": imported_capture.qualname,
            }
        automatic = id(value) in self.automatic_references
        if isinstance(value, ConcreteDefinition):
            return {"tag": "auto_cdef" if automatic else "cdef", "value": encode_cdef_graph(value)}
        if isinstance(value, Object):
            if has_stateful_materialization(value.definition):
                state = value.last_state_ref
                if state is None or state.object != value.object_ref:
                    _fail("unsaved live Object", path)
                return {"tag": "state", "value": state.to_data()}
            return {"tag": "cdef", "value": encode_cdef_graph(value.definition)}
        if isinstance(value, StateRef):
            return {"tag": "auto_state" if automatic else "state", "value": value.to_data()}
        if isinstance(value, ObjectRef):
            return {"tag": "auto_object_ref" if automatic else "object_ref", "value": value.to_data()}
        if isinstance(value, DefLink):
            if not value.is_finalized:
                _fail("unresolved reference assertion", path)
            return {"tag": "link", "kind": value.kind.value, "target": self.value(value.target, f"{path}.target", depth + 1)}
        if isinstance(value, tuple):
            return {"tag": "tuple", "items": [self.value(item, f"{path}[{index}]", depth + 1) for index, item in enumerate(value)]}
        if isinstance(value, list):
            return {"tag": "list", "items": [self.value(item, f"{path}[{index}]", depth + 1) for index, item in enumerate(value)]}
        if isinstance(value, Mapping):
            return {
                "tag": "dict",
                "items": [[self.value(key, f"{path}.key[{index}]", depth + 1), self.value(item, f"{path}.value[{index}]", depth + 1)] for index, (key, item) in enumerate(value.items())],
            }
        if isinstance(value, (set, frozenset)):
            return {"tag": "frozenset" if isinstance(value, frozenset) else "set", "items": [self.value(item, f"{path}.member[{index}]", depth + 1) for index, item in enumerate(value)]}
        if type(value) in _PATH_TYPES.values():
            return {"tag": "path", "kind": type(value).__name__, "value": str(value)}
        if inspect.ismodule(value):
            return {"tag": "import", "module": value.__name__, "qualname": None}
        if inspect.ismethod(value):
            return {"tag": "bound_method", "function": self.value(value.__func__, f"{path}.function", depth + 1), "receiver": self.value(value.__self__, f"{path}.receiver", depth + 1)}
        if inspect.isfunction(value):
            description = describe_callable(value)
            if description.owner == "function":
                return {
                    "tag": "function_owner",
                    "target": self.value(
                        description.raw_target,
                        f"{path}.function_target",
                        depth + 1,
                    ),
                }
            closure = inspect.getclosurevars(value)
            missing = _missing_global_names(value, closure.unbound)
            if missing:
                _fail("missing captured name", path)
            capture_values = {**closure.globals, **closure.nonlocals}
            for name in _annotation_names(value.__annotations__):
                if name in value.__globals__:
                    capture_values.setdefault(name, value.__globals__[name])
            captures = {name: self._capture(item, f"{path}.capture[{index}]", depth + 1) for index, (name, item) in enumerate(capture_values.items())}
            return {
                "tag": "function", "code": _dill_code(value.__code__, path), "name": value.__name__,
                "defaults": self.value(value.__defaults__, f"{path}.defaults", depth + 1),
                "kwdefaults": self.value(value.__kwdefaults__, f"{path}.kwdefaults", depth + 1),
                "annotations": self.value(value.__annotations__, f"{path}.annotations", depth + 1),
                "captures": captures, "freevars": list(value.__code__.co_freevars),
            }
        if inspect.isclass(value):
            ref = _import_ref(value)
            if ref is not None and not _class_has_resource(value):
                return {"tag": "import", "module": ref.module, "qualname": ref.qualname}
            namespace = {
                name: self.value(item, f"{path}.class_field[{index}]", depth + 1)
                for index, (name, item) in enumerate(value.__dict__.items())
                if name not in {"__dict__", "__weakref__", "__module__", "__doc__", "__qualname__"}
                and not isinstance(item, types.MemberDescriptorType)
            }
            return {"tag": "class", "name": value.__name__, "bases": [self.value(base, f"{path}.base[{index}]", depth + 1) for index, base in enumerate(value.__bases__)], "namespace": namespace}
        if not isinstance(value, type) and (_instance_fields(value) is not None):
            ref = _import_ref(type(value))
            type_node = (
                {"module": ref.module, "qualname": ref.qualname}
                if ref is not None else self.value(type(value), f"{path}.type", depth + 1)
            )
            fields = _instance_fields(value)
            assert fields is not None
            return {"tag": "instance", "type": type_node, "fields": {name: self.value(item, f"{path}.field[{index}]", depth + 1) for index, (name, item) in enumerate(fields.items())}}
        if value is None or type(value) in {bool, int, float, str, bytes}:
            return {"tag": "atom", "value": value}
        _fail("unsupported ordinary value", path)

    def _managed_config_resource(self, value: Any, path: str, *, allow_repo: bool) -> dict[str, Any] | None:
        """Detach one supported ManagedConfig authority without retaining its handle."""
        from .store.dir import DirStore

        if value is None:
            return None
        if allow_repo and isinstance(value, Repo):
            definition = value.to_definition().to_data()
            role = "repo"
            descriptors = definition["stores"]
        elif type(value) is DirStore:
            definition = value.to_definition()
            role = "store"
            descriptors = [definition]
        else:
            _fail("unsupported managed config resource", path)
        if any(descriptor.get("kind") != "dir" for descriptor in descriptors):
            _fail("unsupported managed config resource", path)
        return {"role": role, "definition": definition}

    def _managed_config_node(self, value: Any, path: str, depth: int) -> dict[str, Any]:
        """Snapshot one exact ManagedConfig policy and its callback membership."""
        if type(value.rerun) is not bool:
            _fail("malformed managed config", path)
        callbacks = value.callbacks
        if callbacks is not None:
            if type(callbacks) is not list or len(callbacks) > 64:
                _fail("malformed managed config", path)
            for callback in callbacks:
                try:
                    modality = describe_callable(callback).native_modality
                except Exception:
                    _fail("unsupported managed callback", path)
                if modality != "sync":
                    _fail("unsupported managed callback", path)
            callback_nodes = [
                self.value(callback, f"{path}.callback[{index}]", depth + 1)
                for index, callback in enumerate(callbacks)
            ]
        else:
            callback_nodes = []
        return {
            "tag": "managed_config",
            "state_repo": self._managed_config_resource(
                value.state_repo, f"{path}.state_repo", allow_repo=True,
            ),
            "control_store": self._managed_config_resource(
                value.control_store, f"{path}.control_store", allow_repo=False,
            ),
            "rerun": value.rerun,
            "callbacks": callback_nodes,
        }

    def finish(self, root: int) -> bytes:
        try:
            payload = dill.dumps({"version": _VERSION, "root": root, "nodes": self.nodes}, protocol=5)
        except Exception as error:
            raise CoreCallCodecError("core execution transport could not encode call graph") from error
        if len(payload) > self.limit_bytes:
            raise CoreCallCodecError("core execution transport rejected oversized call graph")
        return payload


class _Decoder:
    """Decode one validated graph after worker setup, preserving alias identity."""

    def __init__(self, data: bytes, *, repo: Repo, limit_bytes: int) -> None:
        if not isinstance(data, bytes) or len(data) > limit_bytes:
            raise CoreCallCodecError("core execution transport rejected oversized invocation")
        graph = _load(data, "$")
        if (
            not isinstance(graph, Mapping) or set(graph) != {"version", "root", "nodes"}
            or graph.get("version") != _VERSION or not isinstance(graph.get("nodes"), list)
        ):
            raise CoreCallCodecError("core execution transport rejected malformed call graph")
        self.graph = graph
        self.nodes = graph["nodes"]
        if len(self.nodes) > _MAX_NODES:
            raise CoreCallCodecError("core execution transport rejected oversized call graph")
        self._validate_graph()
        self.repo = repo
        self.memo: dict[int, Any] = {}
        self.active: set[int] = set()
        self.automatic_references: set[int] = set()

    def _validate_graph(self) -> None:
        """Reject malformed closed nodes before imports or Repo materialization."""
        if isinstance(self.graph.get("root"), bool) or not isinstance(self.graph.get("root"), int):
            _fail("malformed graph root", "$")
        allowed = {
            "atom": {"tag", "value"}, "cdef": {"tag", "value"},
            "auto_cdef": {"tag", "value"}, "state": {"tag", "value"},
            "auto_state": {"tag", "value"}, "object_ref": {"tag", "value"},
            "auto_object_ref": {"tag", "value"},
            "import": {"tag", "module", "qualname"}, "link": {"tag", "kind", "target"},
            "tuple": {"tag", "items"}, "list": {"tag", "items"}, "set": {"tag", "items"},
            "frozenset": {"tag", "items"}, "dict": {"tag", "items"},
            "path": {"tag", "kind", "value"},
            "bound_method": {"tag", "function", "receiver"},
            "function_owner": {"tag", "target"},
            "function": {"tag", "code", "name", "defaults", "kwdefaults", "annotations", "captures", "freevars"},
            "class": {"tag", "name", "bases", "namespace"},
            "instance": {"tag", "type", "fields"},
            "managed_declaration": {"tag", "authored", "executable", "owner", "member", "resumable"},
            "managed_composite": {"tag", "declaration", "outer", "owner", "member"},
            "managed_target": {"tag", "receiver", "declaration"},
            "managed_composite_target": {"tag", "receiver", "composite"},
            "managed_config": {"tag", "state_repo", "control_store", "rerun", "callbacks"},
        }
        for index, node in enumerate(self.nodes):
            if not isinstance(node, Mapping) or node.get("tag") not in allowed or set(node) != allowed[node["tag"]]:
                _fail("malformed graph node", f"$.node[{index}]")
            tag = node["tag"]
            if tag in {"tuple", "list", "set", "frozenset"} and not isinstance(node["items"], list):
                _fail("malformed container", f"$.node[{index}]")
            if tag == "path" and (
                not isinstance(node["kind"], str)
                or node["kind"] not in _PATH_TYPES
                or not isinstance(node["value"], str)
            ):
                _fail("malformed path", f"$.node[{index}]")
            if tag == "dict" and (not isinstance(node["items"], list) or any(not isinstance(item, list) or len(item) != 2 for item in node["items"])):
                _fail("malformed mapping", f"$.node[{index}]")
            if tag == "managed_config":
                self._validate_managed_config_node(node, f"$.node[{index}]")
            if tag in {"function", "class", "instance"}:
                mapping = node.get("captures", node.get("namespace", node.get("fields")))
                if not isinstance(mapping, Mapping) or not all(isinstance(name, str) for name in mapping):
                    _fail("malformed structural descriptor", f"$.node[{index}]")
            references: list[Any] = []
            if tag in {"tuple", "list", "set", "frozenset"}:
                references.extend(node["items"])
            elif tag == "dict":
                references.extend(part for pair in node["items"] for part in pair)
            elif tag == "link":
                references.append(node["target"])
            elif tag == "bound_method":
                references.extend((node["function"], node["receiver"]))
            elif tag == "function_owner":
                references.append(node["target"])
            elif tag == "function":
                references.extend((node["defaults"], node["kwdefaults"], node["annotations"], *node["captures"].values()))
                if not isinstance(node["freevars"], list) or not all(isinstance(name, str) for name in node["freevars"]):
                    _fail("malformed function descriptor", f"$.node[{index}]")
            elif tag == "class":
                references.extend((*node["bases"], *node["namespace"].values()))
            elif tag == "instance":
                if isinstance(node["type"], Mapping):
                    if set(node["type"]) != {"module", "qualname"}:
                        _fail("malformed instance type", f"$.node[{index}]")
                else:
                    references.append(node["type"])
                references.extend(node["fields"].values())
            elif tag == "managed_declaration":
                if not isinstance(node["member"], str) or type(node["resumable"]) is not bool:
                    _fail("malformed managed target", f"$.node[{index}]")
                references.extend((node["authored"], node["executable"], node["owner"]))
            elif tag == "managed_composite":
                if not isinstance(node["member"], str):
                    _fail("malformed managed composite", f"$.node[{index}]")
                references.extend((node["declaration"], node["outer"], node["owner"]))
            elif tag == "managed_target":
                references.extend((node["receiver"], node["declaration"]))
            elif tag == "managed_composite_target":
                references.extend((node["receiver"], node["composite"]))
            elif tag == "managed_config":
                references.extend(node["callbacks"])
            if any(isinstance(reference, bool) or not isinstance(reference, int) or not 0 <= reference < len(self.nodes) for reference in references):
                _fail("malformed graph reference", f"$.node[{index}]")

    @staticmethod
    def _validate_managed_config_resource(
            value: Any, path: str, *, allow_repo: bool) -> None:
        """Validate an inert direct-DirStore config role before any resource opens."""
        from .repo_definition import RepoDefinition, _validate_store_descriptor

        if value is None:
            return
        if not isinstance(value, Mapping) or set(value) != {"role", "definition"}:
            _fail("malformed managed config resource", path)
        role = value["role"]
        if role == "repo" and allow_repo:
            try:
                definition = RepoDefinition.from_data(value["definition"])
                descriptors = definition.to_data()["stores"]
            except (RecursionError, TypeError, ValueError, OverflowError, UnicodeError):
                _fail("malformed managed config resource", path)
        elif role == "store":
            try:
                descriptors = [_validate_store_descriptor(value["definition"])]
            except (RecursionError, TypeError, ValueError, OverflowError, UnicodeError):
                _fail("malformed managed config resource", path)
        else:
            _fail("malformed managed config resource", path)
        if any(descriptor["kind"] != "dir" for descriptor in descriptors):
            _fail("unsupported managed config resource", path)

    def _validate_managed_config_node(self, node: Mapping[str, Any], path: str) -> None:
        """Validate closed config policy and inert resource roles before decoding."""
        if type(node["rerun"]) is not bool or not isinstance(node["callbacks"], list) or len(node["callbacks"]) > 64:
            _fail("malformed managed config", path)
        self._validate_managed_config_resource(node["state_repo"], f"{path}.state_repo", allow_repo=True)
        self._validate_managed_config_resource(node["control_store"], f"{path}.control_store", allow_repo=False)

    def value(self, index: Any, path: str = "$") -> Any:
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(self.nodes):
            _fail("malformed graph reference", path)
        if index in self.memo:
            return self.memo[index]
        if index in self.active:
            _fail("cyclic graph", path)
        node = self.nodes[index]
        if not isinstance(node, Mapping) or not isinstance(node.get("tag"), str):
            _fail("malformed graph node", path)
        self.active.add(index)
        try:
            value = self._node(node, path)
            self.memo[index] = value
            return value
        finally:
            self.active.remove(index)

    def _node(self, node: Mapping[str, Any], path: str) -> Any:
        tag = node["tag"]
        if tag == "atom":
            value = node["value"]
            if value is not None and type(value) not in {bool, int, float, str, bytes}:
                _fail("malformed atom", path)
            return value
        if tag in {"cdef", "auto_cdef"}:
            value = decode_cdef_graph(node.get("value"))
        elif tag in {"state", "auto_state"}:
            value = StateRef.from_data(node.get("value"))
        elif tag in {"object_ref", "auto_object_ref"}:
            value = ObjectRef.from_data(node.get("value"))
        else:
            value = None
        if tag in {"cdef", "auto_cdef", "state", "auto_state", "object_ref", "auto_object_ref"}:
            if tag.startswith("auto_"):
                self.automatic_references.add(id(value))
            return value
        if tag in {"tuple", "list", "set", "frozenset"}:
            items = node.get("items")
            if not isinstance(items, list):
                _fail("malformed container", path)
            values = [self.value(item, f"{path}[{position}]") for position, item in enumerate(items)]
            return tuple(values) if tag == "tuple" else values if tag == "list" else set(values) if tag == "set" else frozenset(values)
        if tag == "dict":
            items = node.get("items")
            if not isinstance(items, list):
                _fail("malformed mapping", path)
            result = {}
            for item in items:
                if not isinstance(item, list) or len(item) != 2:
                    _fail("malformed mapping entry", path)
                result[self.value(item[0], f"{path}.key")] = self.value(item[1], f"{path}.value")
            return result
        if tag == "path":
            kind, value = node.get("kind"), node.get("value")
            if (
                not isinstance(kind, str)
                or kind not in _PATH_TYPES
                or not isinstance(value, str)
            ):
                _fail("malformed path", path)
            try:
                return _PATH_TYPES[kind](value)
            except (NotImplementedError, OSError, ValueError) as error:
                raise CoreCallCodecError(
                    f"core execution transport rejected incompatible path at {path}"
                ) from error
        if tag == "import":
            module, qualname = node.get("module"), node.get("qualname")
            if not isinstance(module, str) or (qualname is not None and not isinstance(qualname, str)):
                _fail("malformed import reference", path)
            return ImportRef(module, qualname).resolve()
        if tag == "link":
            from .cdef_graph import EdgeKind
            try:
                kind = EdgeKind(node.get("kind"))
            except Exception as error:
                raise CoreCallCodecError(f"core execution transport rejected malformed link at {path}") from error
            return DefLink.finalized(kind, self.value(node.get("target"), f"{path}.target"))
        if tag == "bound_method":
            return types.MethodType(self.value(node.get("function"), f"{path}.function"), self.value(node.get("receiver"), f"{path}.receiver"))
        if tag == "function_owner":
            from .signatures import function

            return function(self.value(node.get("target"), f"{path}.target"))
        if tag == "function":
            captures = node.get("captures")
            freevars = node.get("freevars")
            if not isinstance(captures, Mapping) or not isinstance(freevars, list) or not all(isinstance(name, str) for name in freevars):
                _fail("malformed function descriptor", path)
            namespace = {"__builtins__": builtins.__dict__}
            namespace.update({name: self.value(index, f"{path}.capture[{name!r}]") for name, index in captures.items()})
            cells = tuple(_cell(namespace[name]) for name in freevars)
            code = _load(node.get("code"), f"{path}.code")
            if not isinstance(code, types.CodeType) or not isinstance(node.get("name"), str):
                _fail("malformed function code", path)
            function = types.FunctionType(code, namespace, node["name"], self.value(node.get("defaults"), f"{path}.defaults"), cells)
            function.__kwdefaults__ = self.value(node.get("kwdefaults"), f"{path}.kwdefaults")
            function.__annotations__ = self.value(node.get("annotations"), f"{path}.annotations")
            return function
        if tag == "class":
            bases = node["bases"]
            if not isinstance(node["name"], str) or not isinstance(bases, list):
                _fail("malformed class descriptor", path)
            decoded_bases = tuple(self.value(index, f"{path}.base[{position}]") for position, index in enumerate(bases))
            if not all(isinstance(base, type) for base in decoded_bases):
                _fail("malformed class base", path)
            namespace = {name: self.value(index, f"{path}.class_field[{position}]") for position, (name, index) in enumerate(node["namespace"].items())}
            return type(node["name"], decoded_bases, namespace)
        if tag == "instance":
            descriptor = node.get("type")
            if isinstance(descriptor, Mapping):
                cls = ImportRef(descriptor.get("module"), descriptor.get("qualname")).resolve()
            else:
                cls = self.value(descriptor, f"{path}.type")
            if not isinstance(cls, type):
                _fail("malformed instance type", path)
            fields = node["fields"]
            instance = object.__new__(cls)
            for position, (name, index) in enumerate(fields.items()):
                if not isinstance(name, str):
                    _fail("malformed instance field", path)
                setattr(instance, name, self.value(index, f"{path}.field[{position}]"))
            return instance
        if tag == "managed_declaration":
            from dryml.managed.descriptor import ManagedOperation, _ManagedComposite

            authored = self.value(node["authored"], f"{path}.authored")
            executable = self.value(node["executable"], f"{path}.executable")
            owner = self.value(node["owner"], f"{path}.owner")
            if (
                type(authored) is not types.FunctionType
                or type(executable) is not types.FunctionType
                or not isinstance(owner, type)
            ):
                _fail("malformed managed declaration", path)
            try:
                declaration = ManagedOperation(executable, resumable=node["resumable"])
                declaration.__set_name__(owner, node["member"])
            except (TypeError, ValueError) as error:
                raise CoreCallCodecError(
                    f"core execution transport rejected malformed managed declaration at {path}"
                ) from error
            if declaration.author_signature != inspect.signature(authored):
                _fail("malformed managed declaration", path)
            # The executable wrapper can reconstruct an equivalent raw function
            # through a function-owner node. Keep the separately captured authored
            # function as the declaration identity used for binding and signatures.
            declaration._target = authored
            return declaration
        if tag == "managed_composite":
            from dryml.managed.descriptor import ManagedOperation, _ManagedComposite

            declaration = self.value(node["declaration"], f"{path}.declaration")
            outer = self.value(node["outer"], f"{path}.outer")
            owner = self.value(node["owner"], f"{path}.owner")
            if (
                type(declaration) is not ManagedOperation
                or type(outer) is not types.FunctionType
                or not isinstance(owner, type)
                or declaration._owner is not owner
                or declaration.member != node["member"]
            ):
                _fail("malformed managed composite", path)
            return _ManagedComposite(declaration, outer)
        if tag in {"managed_target", "managed_composite_target"}:
            receiver = self.value(node["receiver"], f"{path}.receiver")
            # Managed owns lifecycle admission and currently requires its bound
            # receiver before it can expose the owner seam.
            if isinstance(receiver, (StateRef, ObjectRef)):
                receiver = self.repo.materialize_boundary((receiver,), reuse_live="never")[0]
            evidence = self.value(
                node["declaration"] if tag == "managed_target" else node["composite"],
                f"{path}.declaration" if tag == "managed_target" else f"{path}.composite",
            )
            from dryml.managed.descriptor import (
                ManagedOperation, _BoundComposite, _BoundOperation, _ManagedComposite,
            )

            declaration = evidence if tag == "managed_target" else (
                evidence._descriptor if type(evidence) is _ManagedComposite else None
            )
            if type(declaration) is not ManagedOperation or not isinstance(receiver, declaration._owner):
                _fail("malformed managed target", path)
            if tag == "managed_target":
                return _BoundOperation(declaration, receiver)
            if type(evidence) is not _ManagedComposite:
                _fail("malformed managed target", path)
            return _BoundComposite(evidence, receiver)
        if tag == "managed_config":
            from dryml.managed import ManagedConfig

            def decode_resource(value: Any, *, state_repo: bool) -> Any:
                if value is None:
                    return None
                role = value["role"]
                definition = value["definition"]
                if role == "repo" and state_repo:
                    from .repo_definition import RepoDefinition

                    return Repo.from_definition(RepoDefinition.from_data(definition))
                if role == "store":
                    return Store.from_definition(definition)
                _fail("malformed managed config resource", path)

            callbacks = [
                self.value(index, f"{path}.callback[{position}]")
                for position, index in enumerate(node["callbacks"])
            ]
            try:
                return ManagedConfig(
                    state_repo=decode_resource(node["state_repo"], state_repo=True),
                    control_store=decode_resource(node["control_store"], state_repo=False),
                    rerun=node["rerun"],
                    callbacks=callbacks,
                )
            except (TypeError, ValueError) as error:
                raise CoreCallCodecError(
                    f"core execution transport rejected malformed managed config at {path}"
                ) from error
        _fail("unknown graph tag", path)


def _cell(value: Any):
    return (lambda: value).__closure__[0]


def _selection_data(
        selections: Mapping[Any, Any], repo: Repo,
        store_table: tuple[Store, ...] | None = None,
) -> dict[Any, Any]:
    """Detach selection identity and Store intent without transporting a handle."""
    result = {}
    for key, selection in selections.items():
        if selection is None:
            result[key] = None
            continue
        if isinstance(selection, ObjectRef):
            selection = ReferenceSelection(selection)
        elif isinstance(selection, tuple) and len(selection) == 2:
            selection = ReferenceSelection(selection[0], selection[1])
        if not isinstance(selection, ReferenceSelection) or not isinstance(selection.object_ref, ObjectRef):
            _fail("malformed selection", "$.selections")
        store = selection.store
        table = tuple(repo.stores) if store_table is None else store_table
        if store is not None and not any(store is candidate for candidate in table):
            _fail("unsupported selected Store", "$.selections")
        evidence = repo.reference_evidence(selection.object_ref.definition)
        if not any(
            item.object_ref == selection.object_ref and (store is None or store in item.stores)
            for item in evidence.declarations
        ) and not any(
            item.state_ref.object == selection.object_ref and (store is None or store in item.stores)
            for item in evidence.states
        ):
            _fail("selected authority is unavailable", "$.selections")
        result[key] = {
            "object_ref": selection.object_ref,
            "store": None if store is None else next(
                index for index, candidate in enumerate(table) if candidate is store
            ),
        }
    return result


def _selection_controls(data: Mapping[Any, Any], repo: Repo) -> Mapping[Any, Any]:
    """Rebind frozen Store-table identifiers to exactly one worker Repo handle."""
    result = {}
    for key, value in data.items():
        if value is None:
            result[key] = None
            continue
        if not isinstance(value, Mapping) or set(value) != {"object_ref", "store"} or not isinstance(value["object_ref"], ObjectRef):
            _fail("malformed selection", "$.selections")
        index = value["store"]
        if index is not None and (isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(repo.stores)):
            _fail("malformed selected Store", "$.selections")
        store = None if index is None else repo.stores[index]
        result[key] = ReferenceSelection(value["object_ref"], store)
    return result


def encode_invocation(
        fn: Any, args: tuple[Any, ...], kwargs: Mapping[str, Any], *, repo: Repo,
        selections: Mapping[Any, Any] | None = None,
        store_table: tuple[Store, ...] | None = None,
        limit_bytes: int = _DEFAULT_LIMIT,
        update_targets: tuple[Mapping[str, Any], ...] = (),
) -> bytes:
    """Encode one detached call graph without activating annotations or saving input state."""
    if not callable(fn) or not isinstance(args, tuple) or not isinstance(kwargs, Mapping) or not all(isinstance(key, str) for key in kwargs):
        raise CoreCallCodecError("core execution transport requires a callable, tuple arguments, and string keyword mapping")
    description = describe_callable(fn)
    target = description.raw_target
    owner = description.owner
    if description.native_modality != "sync":
        raise CoreCallCodecError("core execution transport rejected async or generator target")
    encoder = _Encoder(limit_bytes=limit_bytes)
    if not isinstance(update_targets, tuple):
        raise CoreCallCodecError("core execution transport requires tuple update targets")
    root = encoder.value({"target": target, "args": args, "kwargs": dict(kwargs), "owner": owner, "selections": _selection_data(selections or {}, repo, store_table), "updates": list(update_targets)}, "$")
    return encoder.finish(root)


def decode_invocation(data: bytes, *, repo: Repo, limit_bytes: int = _DEFAULT_LIMIT) -> tuple[Any, tuple[Any, ...], dict[str, Any], str, Mapping[Any, Any], tuple[Mapping[str, Any], ...]]:
    """Reconstruct a call graph after worker setup and before one local boundary."""
    decoder = _Decoder(data, repo=repo, limit_bytes=limit_bytes)
    decoded = decoder.value(decoder.graph["root"])
    if not isinstance(decoded, Mapping) or set(decoded) != {"target", "args", "kwargs", "owner", "selections", "updates"}:
        raise CoreCallCodecError("core execution transport rejected malformed call descriptor")
    if not callable(decoded["target"]) or not isinstance(decoded["args"], tuple) or not isinstance(decoded["kwargs"], dict) or not all(isinstance(key, str) for key in decoded["kwargs"]):
        raise CoreCallCodecError("core execution transport rejected malformed call descriptor")
    if decoded["owner"] not in {"ordinary", "function", "method", "managed"} or not isinstance(decoded["selections"], Mapping) or not isinstance(decoded["updates"], list):
        raise CoreCallCodecError("core execution transport rejected malformed call descriptor")
    from .definition import ConcreteDefinition, Definition
    if decoded["owner"] in {"method", "managed"} and isinstance(
            decoded["target"], (ConcreteDefinition, Definition, ObjectRef, StateRef)):
        decoded["target"] = repo.materialize_boundary((decoded["target"],), reuse_live="never")[0]
    targets = []
    for item in decoded["updates"]:
        if not isinstance(item, Mapping) or set(item) != {"target", "object_digest", "path"} or not all(isinstance(item[name], str) and item[name] for name in item):
            raise CoreCallCodecError("core execution transport rejected malformed update target")
        targets.append(item)
    return decoded["target"], decoded["args"], decoded["kwargs"], decoded["owner"], _selection_controls(decoded["selections"], repo), tuple(targets)


def _target_materializing_roots(target: Any) -> tuple[tuple[Any, ...], Any]:
    """Return callable receiver/capture roots and their post-admission installer.

    Captures have no Ref annotation position of their own, so a callable that
    directly dereferences a transported authority needs its materialized value.
    The roots enter the selected argument boundary as extras rather than being
    independently restored. Explicit Ref arguments therefore remain references,
    while Mat arguments and callable captures share the same Repo transaction.
    """
    roots: list[Any] = []
    setters: list[Any] = []

    def add(value: Any, setter: Any) -> None:
        roots.append(value)
        setters.append(setter)

    def instance_fields(value: Any) -> None:
        fields = _instance_fields(value)
        if fields is None:
            return
        for name, field in fields.items():
            add(field, lambda result, receiver=value, field_name=name: setattr(receiver, field_name, result))

    def function_captures(value: types.FunctionType) -> None:
        for name, captured in value.__globals__.items():
            if name != "__builtins__":
                add(captured, lambda result, namespace=value.__globals__, capture=name: namespace.__setitem__(capture, result))
        if value.__closure__ is not None:
            for cell in value.__closure__:
                add(cell.cell_contents, lambda result, destination=cell: setattr(destination, "cell_contents", result))

    if inspect.ismethod(target):
        receiver = target.__self__
        if not isinstance(receiver, type):
            instance_fields(receiver)
        function = target.__func__
        if isinstance(function, types.FunctionType):
            function_captures(function)
    elif isinstance(target, types.FunctionType):
        function_captures(target)
    elif not inspect.isclass(target):
        instance_fields(target)

    def install(values: tuple[Any, ...]) -> None:
        if len(values) != len(setters):
            raise CoreCallCodecError("core execution transport received invalid capture materialization")
        for setter, value in zip(setters, values):
            setter(value)

    return tuple(roots), install


def _walk_live_objects(value: Any, found: list[Object], seen: set[int], active: set[int]) -> None:
    """Collect supported live Object leaves once while rejecting result cycles."""
    if isinstance(value, Object):
        if id(value) not in seen:
            found.append(value)
            seen.add(id(value))
        return
    if isinstance(value, (ObjectRef, StateRef)):
        return
    if isinstance(value, (tuple, list)):
        identity = id(value)
        if identity in active:
            raise CoreCallCodecError("core execution transport rejected cyclic graph at $.result")
        if identity in seen:
            return
        seen.add(identity)
        active.add(identity)
        try:
            for item in value:
                _walk_live_objects(item, found, seen, active)
        finally:
            active.remove(identity)
        return
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active:
            raise CoreCallCodecError("core execution transport rejected cyclic graph at $.result")
        if identity in seen:
            return
        seen.add(identity)
        active.add(identity)
        try:
            for key, item in value.items():
                _walk_live_objects(key, found, seen, active)
                _walk_live_objects(item, found, seen, active)
        finally:
            active.remove(identity)


def _maximal_roots(values: list[Object]) -> list[Object]:
    """Coalesce repeated selected Objects and descendants into first maximal roots."""
    roots: list[Object] = []
    for value in values:
        if any(value is candidate for candidate in roots):
            continue
        ancestors = [
            root for root in roots
            if any(bound is value for bound in getattr(root, "_runtime_projection", {}).values())
        ]
        if ancestors:
            continue
        descendants = [
            root for root in roots
            if any(bound is root for bound in getattr(value, "_runtime_projection", {}).values())
        ]
        if descendants:
            roots = [root for root in roots if root not in descendants]
        roots.append(value)
    return roots


def _report_data(report: Any, repo: Repo, state_ref: StateRef | None) -> list[dict[str, Any]]:
    """Project ephemeral StoreReport facts into Store-table-relative evidence."""
    if report is None:
        return []
    indexes = {id(store): index for index, store in enumerate(repo.stores)}
    facts = []
    for item in report.publications:
        index = indexes.get(id(item.store))
        reference = item.state_ref if isinstance(item.state_ref, StateRef) else state_ref
        if index is None or not isinstance(reference, StateRef):
            continue
        facts.append({
            "state": reference.to_data(), "store": index, "phase": item.phase,
            "status": item.status, "path": str(item.path),
        })
    return facts


class _ResultPublisher:
    """Publish live result/update graphs once and replace them with exact authority."""

    def __init__(self, repo: Repo, updates: list[tuple[str, Object]], *, result_limit_bytes: int) -> None:
        self.repo = repo
        self._result_limit_bytes = result_limit_bytes
        self.updates = self._maximal_updates(updates)
        self.references: dict[int, StateRef] = {}
        self.publications: list[dict[str, Any]] = []
        self.update_states: list[dict[str, Any]] = []
        self._published_updates = False
        self._memo: dict[int, Any] = {}
        self._active: set[int] = set()
        self.automatic_references: set[int] = set()
        self._prospective_states: dict[int, StateRef] = {}

    @staticmethod
    def _maximal_updates(values: list[tuple[str, Object]]) -> list[tuple[str, Object]]:
        """Coalesce selected roots while retaining the winning caller token."""
        roots: list[tuple[str, Object]] = []
        for target, value in values:
            if any(value is root for _, root in roots):
                continue
            if any(any(bound is value for bound in getattr(root, "_runtime_projection", {}).values()) for _, root in roots):
                continue
            roots = [item for item in roots if not any(
                bound is item[1] for bound in getattr(value, "_runtime_projection", {}).values()
            )]
            roots.append((target, value))
        return roots

    def _remember_graph(self, root: Object, state: StateRef) -> None:
        self.references[id(root)] = state
        for path, bound in getattr(root, "_runtime_projection", {}).items():
            if isinstance(bound, Object):
                if bound is root:
                    continue
                try:
                    self.references[id(bound)] = state.at(path)
                except ValueError:
                    # Non-materializing runtime values never carry StateRef state.
                    continue

    def _save(self, root: Object, *, update: bool, target: str | None = None) -> StateRef:
        try:
            state, report = self.repo.save(root, deep_capture=True, report_stores=True)
        except BaseException as error:
            report = getattr(error, "report", None)
            state = getattr(root, "last_state_ref", None)
            self.publications.extend(_report_data(
                report, self.repo, state if isinstance(state, StateRef) else None,
            ))
            raise
        self.publications.extend(_report_data(report, self.repo, state))
        self._remember_graph(root, state)
        if update:
            assert target is not None
            self.update_states.append({"state": state.to_data(), "object": state.object.to_data(), "target": target})
        return state

    def publish_updates(self) -> None:
        """Deep-save selected maximal worker argument roots exactly once."""
        if self._published_updates:
            return
        self._published_updates = True
        for target, root in self.updates:
            self._save(root, update=True, target=target)

    def _prospective_state(self, value: Object) -> StateRef:
        """Build a shape-accurate future receipt without saving ``value``."""
        state = self._prospective_states.get(id(value))
        if state is None:
            state = StateRef(value.object_ref, {
                path: "core-" + "0" * 64 for path in value.object_ref.objects
            })
            self._prospective_states[id(value)] = state
        return state

    def _preflight_graph(self, value: Any, path: str, memo: dict[int, Any],
                         active: set[int], depth: int = 0) -> Any:
        """Detach live Objects and reject unsupported result data before a save."""
        if depth > _MAX_DEPTH:
            _fail("graph depth exceeds the transport bound", path)
        if _is_resource(value):
            _fail("live core resource", path)
        if isinstance(value, Object):
            return self._prospective_state(value)
        if isinstance(value, (ObjectRef, StateRef, ConcreteDefinition)):
            return value
        identity = id(value)
        if isinstance(value, tuple):
            if identity in active:
                _fail("cyclic graph", path)
            if identity not in memo:
                active.add(identity)
                try:
                    memo[identity] = tuple(
                        self._preflight_graph(item, f"{path}[{index}]", memo, active, depth + 1)
                        for index, item in enumerate(value)
                    )
                finally:
                    active.remove(identity)
            return memo[identity]
        if isinstance(value, list):
            if identity in active:
                _fail("cyclic graph", path)
            if identity not in memo:
                active.add(identity)
                try:
                    memo[identity] = [
                        self._preflight_graph(item, f"{path}[{index}]", memo, active, depth + 1)
                        for index, item in enumerate(value)
                    ]
                finally:
                    active.remove(identity)
            return memo[identity]
        if isinstance(value, Mapping):
            if identity in active:
                _fail("cyclic graph", path)
            if identity not in memo:
                active.add(identity)
                try:
                    memo[identity] = {
                        self._preflight_graph(key, f"{path}.key[{index}]", memo, active, depth + 1):
                        self._preflight_graph(item, f"{path}.value[{index}]", memo, active, depth + 1)
                        for index, (key, item) in enumerate(value.items())
                    }
                finally:
                    active.remove(identity)
            return memo[identity]
        if isinstance(value, (set, frozenset)):
            if identity in active:
                _fail("cyclic graph", path)
            if identity not in memo:
                active.add(identity)
                try:
                    items = [
                        self._preflight_graph(item, f"{path}.member[{index}]", memo, active, depth + 1)
                        for index, item in enumerate(value)
                    ]
                    memo[identity] = frozenset(items) if isinstance(value, frozenset) else set(items)
                finally:
                    active.remove(identity)
            return memo[identity]
        return value

    def _validate_before_publication(self, value: Any) -> None:
        """Validate the result and every selected update root without side effects."""
        graph = self._preflight_graph(
            {"result": value, "updates": tuple(root for _, root in self.updates)},
            "$.preflight", {}, set(),
        )
        _result_graph(graph, limit_bytes=self._result_limit_bytes, automatic_references=set())

    def _reserve_evidence(self, value: Any) -> None:
        """Prove the actual outcome budget can retain all possible save evidence."""
        if len(self.repo.stores) > _MAX_NODES:
            raise CoreCallCodecError("core execution transport rejected oversized Store table")
        result_objects: list[Object] = []
        _walk_live_objects(value, result_objects, set(), set())
        roots = [root for _, root in self.updates] + _maximal_roots(result_objects)
        if len(roots) > _MAX_NODES:
            raise CoreCallCodecError("core execution transport rejected oversized publication roots")
        updates = [
            {"state": self._prospective_state(root).to_data(),
             "object": root.object_ref.to_data(), "target": target}
            for target, root in self.updates
        ]
        phases = ("definition", "state", "snapshot", "membership", "index", "main", "alias", "commit")
        publications = [
            {"state": state.to_data(), "store": store_index, "phase": phase,
             "status": "completed", "path": str(path)}
            for root in roots
            for state in (self._prospective_state(root),)
            for path in state.states
            for store_index in range(len(self.repo.stores))
            for phase in phases
        ]
        # The ordinary result can be dropped after publication, but durable
        # evidence cannot. Reserve exactly the fallback outcome before saving.
        _outcome(
            False, updates=updates, publications=publications,
            reason="result outcome exceeds configured bound after publication",
            limit_bytes=self._result_limit_bytes,
        )

    def raw_result(self, value: Any) -> Any:
        """Validate and reserve the full result/update graph before publication."""
        self._validate_before_publication(value)
        self._reserve_evidence(value)
        self.publish_updates()
        if isinstance(value, (ObjectRef, StateRef, ConcreteDefinition)):
            return value
        return self.value(value)

    def value(self, value: Any) -> Any:
        """Replace supported nested live Object leaves with automatic references."""
        if isinstance(value, (ObjectRef, StateRef, ConcreteDefinition)):
            return value
        if isinstance(value, Object):
            state = self.references.get(id(value))
            if state is None:
                state = self._save(value, update=False)
            from .signatures import _select_automatic
            selected = state if has_stateful_materialization(value.definition) else _select_automatic(value, "return")
            self.automatic_references.add(id(selected))
            return selected
        if isinstance(value, tuple):
            return self._container(value, tuple)
        if isinstance(value, list):
            return self._container(value, list)
        if isinstance(value, Mapping):
            identity = id(value)
            if identity in self._active:
                raise CoreCallCodecError("core execution transport rejected cyclic graph at $.result")
            if identity in self._memo:
                return self._memo[identity]
            self._active.add(identity)
            try:
                result = {self.value(key): self.value(item) for key, item in value.items()}
                self._memo[identity] = result
                return result
            finally:
                self._active.remove(identity)
        return value

    def _container(self, value: tuple[Any, ...] | list[Any], build: Any) -> Any:
        identity = id(value)
        if identity in self._active:
            raise CoreCallCodecError("core execution transport rejected cyclic graph at $.result")
        if identity in self._memo:
            return self._memo[identity]
        self._active.add(identity)
        try:
            result = build(self.value(item) for item in value)
            self._memo[identity] = result
            return result
        finally:
            self._active.remove(identity)


def invoke_invocation(data: bytes, *, repo: Repo, invocation_limit_bytes: int = _DEFAULT_LIMIT,
                        result_limit_bytes: int = _DEFAULT_LIMIT, update_args: bool = False) -> bytes:
    """Invoke once, publish selected state, and return a tagged portable outcome.

    Result publication happens at each owner's raw-return seam, before its one
    existing return-normalization boundary. Expected invocation, publication, and
    result-encoding failures retain any completed publication ledger in the
    returned outcome rather than escaping as a type-only worker exception.
    """
    target, args, kwargs, owner, selections, update_targets = decode_invocation(
        data, repo=repo, limit_bytes=invocation_limit_bytes,
    )
    publisher: _ResultPublisher | None = None
    delivered: list[Object] = []

    def on_delivered(call_args: tuple[Any, ...], call_kwargs: Mapping[str, Any]) -> None:
        nonlocal publisher
        # Only delivered argument positions count: captures and Ref values remain
        # outside the update set, while nested materialized Object values are kept.
        seen: set[int] = set()
        active: set[int] = set()
        for value in (*call_args, *call_kwargs.values()):
            _walk_live_objects(value, delivered, seen, active)
        selected: list[tuple[str, Object]] = []
        unmatched = list(update_targets)
        for value in delivered:
            for index, item in enumerate(unmatched):
                if value.object_ref.digest() == item["object_digest"]:
                    selected.append((item["target"], value))
                    unmatched.pop(index)
                    break
        publisher = _ResultPublisher(
            repo, selected if update_args else [], result_limit_bytes=result_limit_bytes,
        )

    def on_raw_result(value: Any) -> Any:
        if publisher is None:
            raise CoreCallCodecError("core execution transport did not deliver arguments")
        return publisher.raw_result(value)

    try:
        capture_roots, install_captures = _target_materializing_roots(target)
        if owner == "function":
            from .signatures import _invoke_function_with_raw_result
            plan = compile_signature(target)
            result = _invoke_function_with_raw_result(
                plan, target, args, kwargs, repo=repo, reuse_live="never",
                selections=selections, on_raw_result=on_raw_result,
                on_delivered_args=on_delivered,
                extra_mat_roots=capture_roots, on_extra_materialized=install_captures,
            )
        elif owner in {"method", "managed"}:
            invoke = getattr(target, "__dryml_execute_invoke__", None)
            if not callable(invoke):
                raise CoreCallCodecError("core execution transport rejected missing owner invocation seam")
            if owner == "managed":
                publisher = _ResultPublisher(repo, [], result_limit_bytes=result_limit_bytes)
                result = invoke(args, kwargs, repo=repo, on_raw_result=publisher.raw_result)
            else:
                publisher = _ResultPublisher(repo, [], result_limit_bytes=result_limit_bytes)
                result = invoke(args, kwargs, repo=repo, on_raw_result=publisher.raw_result)
        else:
            plan = compile_signature(target)
            call_args, call_kwargs = plan.prepare_args(
                args, kwargs, repo=repo, reuse_live="never", selections=selections,
            ).deliver_args(
                extra_mat_roots=capture_roots, on_extra_materialized=install_captures,
            )
            on_delivered(call_args, call_kwargs)
            result = on_raw_result(target(*call_args, **call_kwargs))
            result = plan.prepare_return(
                result, repo=repo, reuse_live="never", preserve_reference_data=True,
            ).deliver_return(preserve_reference_data=True)
        assert publisher is not None
        # A default materializing return can load a fresh local Object after raw
        # publication. Its saved receipt maps it back to the exact first snapshot.
        adapted = publisher.value(result)
        return _outcome(
            True, result=_result_graph(
                adapted, limit_bytes=result_limit_bytes,
                automatic_references=publisher.automatic_references,
            ),
            updates=publisher.update_states, publications=publisher.publications,
            limit_bytes=result_limit_bytes,
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as error:
        publications = [] if publisher is None else publisher.publications
        updates = [] if publisher is None else publisher.update_states
        return _outcome(
            False, updates=updates, publications=publications,
            reason=_failure_reason(error), limit_bytes=result_limit_bytes,
        )


__all__ = ["CoreCallCodecError", "decode_invocation", "decode_outcome", "encode_invocation", "invoke_invocation"]
