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
import sys
import types
from collections.abc import Mapping
from typing import Any

import dill

from .cdef_codec import decode_cdef_graph, encode_cdef_graph
from .cdef_graph import has_stateful_materialization
from .links import DefLink
from .object import Object
from .reference_values import ObjectRef, StateRef
from .repo import Repo
from .store.store import Store
from .signatures import ReferenceSelection, SignatureError, compile_signature
from .symbol import ImportRef


_VERSION = 1
_DEFAULT_LIMIT = 67_108_864
_MAX_NODES = 65_536
_MAX_DEPTH = 64


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

    def __init__(self, *, limit_bytes: int) -> None:
        self.nodes: list[dict[str, Any]] = []
        self.memo: dict[int, int] = {}
        self.active: set[int] = set()
        self.limit_bytes = limit_bytes

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
        if _is_resource(value):
            _fail("live core resource", path)
        if isinstance(value, Object):
            if has_stateful_materialization(value.definition):
                state = value.last_state_ref
                if state is None or state.object != value.object_ref:
                    _fail("unsaved live Object", path)
                return {"tag": "state", "value": state.to_data()}
            return {"tag": "cdef", "value": encode_cdef_graph(value.definition)}
        if isinstance(value, StateRef):
            return {"tag": "state", "value": value.to_data()}
        if isinstance(value, ObjectRef):
            return {"tag": "object_ref", "value": value.to_data()}
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
        if inspect.ismodule(value):
            return {"tag": "import", "module": value.__name__, "qualname": None}
        if inspect.ismethod(value):
            return {"tag": "bound_method", "function": self.value(value.__func__, f"{path}.function", depth + 1), "receiver": self.value(value.__self__, f"{path}.receiver", depth + 1)}
        if inspect.isfunction(value):
            closure = inspect.getclosurevars(value)
            missing = _missing_global_names(value, closure.unbound)
            if missing:
                _fail("missing captured name", path)
            capture_values = {**closure.globals, **closure.nonlocals}
            for name in _annotation_names(value.__annotations__):
                if name in value.__globals__:
                    capture_values.setdefault(name, value.__globals__[name])
            captures = {name: self.value(item, f"{path}.capture[{index}]", depth + 1) for index, (name, item) in enumerate(capture_values.items())}
            return {
                "tag": "function", "code": _dill_code(value.__code__, path), "name": value.__name__,
                "defaults": self.value(value.__defaults__, f"{path}.defaults", depth + 1),
                "kwdefaults": self.value(value.__kwdefaults__, f"{path}.kwdefaults", depth + 1),
                "annotations": self.value(value.__annotations__, f"{path}.annotations", depth + 1),
                "captures": captures, "freevars": list(value.__code__.co_freevars),
            }
        if getattr(value, "__dryml_execute_owner__", None) == "managed":
            receiver = getattr(value, "_instance", None)
            descriptor = getattr(value, "_descriptor", None)
            member = getattr(descriptor, "member", None)
            if receiver is None or not isinstance(member, str):
                _fail("malformed managed target", path)
            return {"tag": "managed_target", "receiver": self.value(receiver, f"{path}.receiver", depth + 1), "member": member}
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
        if not isinstance(graph, Mapping) or graph.get("version") != _VERSION or not isinstance(graph.get("nodes"), list):
            raise CoreCallCodecError("core execution transport rejected malformed call graph")
        self.graph = graph
        self.nodes = graph["nodes"]
        if len(self.nodes) > _MAX_NODES:
            raise CoreCallCodecError("core execution transport rejected oversized call graph")
        self._validate_graph()
        self.repo = repo
        self.memo: dict[int, Any] = {}
        self.active: set[int] = set()

    def _validate_graph(self) -> None:
        """Reject malformed closed nodes before imports or Repo materialization."""
        if isinstance(self.graph.get("root"), bool) or not isinstance(self.graph.get("root"), int):
            _fail("malformed graph root", "$")
        allowed = {
            "atom": {"tag", "value"}, "cdef": {"tag", "value"},
            "state": {"tag", "value"}, "object_ref": {"tag", "value"},
            "import": {"tag", "module", "qualname"}, "link": {"tag", "kind", "target"},
            "tuple": {"tag", "items"}, "list": {"tag", "items"}, "set": {"tag", "items"},
            "frozenset": {"tag", "items"}, "dict": {"tag", "items"},
            "bound_method": {"tag", "function", "receiver"},
            "function": {"tag", "code", "name", "defaults", "kwdefaults", "annotations", "captures", "freevars"},
            "class": {"tag", "name", "bases", "namespace"},
            "instance": {"tag", "type", "fields"},
            "managed_target": {"tag", "receiver", "member"},
        }
        for index, node in enumerate(self.nodes):
            if not isinstance(node, Mapping) or node.get("tag") not in allowed or set(node) != allowed[node["tag"]]:
                _fail("malformed graph node", f"$.node[{index}]")
            tag = node["tag"]
            if tag in {"tuple", "list", "set", "frozenset"} and not isinstance(node["items"], list):
                _fail("malformed container", f"$.node[{index}]")
            if tag == "dict" and (not isinstance(node["items"], list) or any(not isinstance(item, list) or len(item) != 2 for item in node["items"])):
                _fail("malformed mapping", f"$.node[{index}]")
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
            elif tag == "managed_target":
                if not isinstance(node["member"], str):
                    _fail("malformed managed target", f"$.node[{index}]")
                references.append(node["receiver"])
            if any(isinstance(reference, bool) or not isinstance(reference, int) or not 0 <= reference < len(self.nodes) for reference in references):
                _fail("malformed graph reference", f"$.node[{index}]")

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
        if tag == "cdef":
            return decode_cdef_graph(node.get("value"))
        if tag == "state":
            return StateRef.from_data(node.get("value"))
        if tag == "object_ref":
            return ObjectRef.from_data(node.get("value"))
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
        if tag == "managed_target":
            receiver = self.value(node["receiver"], f"{path}.receiver")
            # Managed owns lifecycle admission and currently requires its bound
            # receiver before it can expose the owner seam.
            if isinstance(receiver, (StateRef, ObjectRef)):
                receiver = self.repo.materialize_boundary((receiver,), reuse_live="never")[0]
            target = getattr(receiver, node["member"], None)
            if not callable(target) or getattr(target, "__dryml_execute_owner__", None) != "managed":
                _fail("malformed managed target", path)
            return target
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
) -> bytes:
    """Encode one detached call graph without activating annotations or saving input state."""
    if not callable(fn) or not isinstance(args, tuple) or not isinstance(kwargs, Mapping) or not all(isinstance(key, str) for key in kwargs):
        raise CoreCallCodecError("core execution transport requires a callable, tuple arguments, and string keyword mapping")
    target = getattr(fn, "__dryml_execute_raw_target__", None)
    owner = "function" if target is not None else getattr(fn, "__dryml_execute_owner__", "ordinary")
    if target is None:
        target = fn
    annotation_target = target.__func__ if inspect.ismethod(target) else target
    if not inspect.isfunction(annotation_target) and not inspect.ismethod(annotation_target):
        annotation_target = getattr(annotation_target, "__call__", annotation_target)
    if inspect.iscoroutinefunction(annotation_target) or inspect.isgeneratorfunction(annotation_target) or inspect.isasyncgenfunction(annotation_target):
        raise CoreCallCodecError("core execution transport rejected async or generator target")
    encoder = _Encoder(limit_bytes=limit_bytes)
    root = encoder.value({"target": target, "args": args, "kwargs": dict(kwargs), "owner": owner, "selections": _selection_data(selections or {}, repo, store_table)}, "$")
    return encoder.finish(root)


def decode_invocation(data: bytes, *, repo: Repo, limit_bytes: int = _DEFAULT_LIMIT) -> tuple[Any, tuple[Any, ...], dict[str, Any], str, Mapping[Any, Any]]:
    """Reconstruct a call graph after worker setup and before one local boundary."""
    decoder = _Decoder(data, repo=repo, limit_bytes=limit_bytes)
    decoded = decoder.value(decoder.graph["root"])
    if not isinstance(decoded, Mapping) or set(decoded) != {"target", "args", "kwargs", "owner", "selections"}:
        raise CoreCallCodecError("core execution transport rejected malformed call descriptor")
    if not callable(decoded["target"]) or not isinstance(decoded["args"], tuple) or not isinstance(decoded["kwargs"], dict) or not all(isinstance(key, str) for key in decoded["kwargs"]):
        raise CoreCallCodecError("core execution transport rejected malformed call descriptor")
    if decoded["owner"] not in {"ordinary", "function", "method", "managed"} or not isinstance(decoded["selections"], Mapping):
        raise CoreCallCodecError("core execution transport rejected malformed call descriptor")
    from .definition import ConcreteDefinition, Definition
    if decoded["owner"] in {"method", "managed"} and isinstance(
            decoded["target"], (ConcreteDefinition, Definition, ObjectRef, StateRef)):
        decoded["target"] = repo.materialize_boundary((decoded["target"],), reuse_live="never")[0]
    return decoded["target"], decoded["args"], decoded["kwargs"], decoded["owner"], _selection_controls(decoded["selections"], repo)


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


def invoke_invocation(data: bytes, *, repo: Repo, invocation_limit_bytes: int = _DEFAULT_LIMIT,
                      result_limit_bytes: int = _DEFAULT_LIMIT, managed_config: Any = None) -> bytes:
    """Activate one worker-local signature boundary, invoke once, and encode ordinary output.

    Live output publication is deliberately left to U6; this seam rejects it rather
    than pickling a live Object.
    """
    target, args, kwargs, owner, selections = decode_invocation(
        data, repo=repo, limit_bytes=invocation_limit_bytes,
    )
    try:
        capture_roots, install_captures = _target_materializing_roots(target)
        if owner == "function":
            from .signatures import _invoke_function_with_raw_result
            plan = compile_signature(target)
            result = _invoke_function_with_raw_result(
                plan, target, args, kwargs, repo=repo, reuse_live="never",
                selections=selections, on_raw_result=_reject_live_result,
                extra_mat_roots=capture_roots, on_extra_materialized=install_captures,
            )
        elif owner in {"method", "managed"}:
            invoke = getattr(target, "__dryml_execute_invoke__", None)
            if not callable(invoke):
                raise CoreCallCodecError("core execution transport rejected missing owner invocation seam")
            if owner == "managed":
                result = invoke(args, kwargs, repo=repo, on_raw_result=_reject_live_result,
                                managed_config=managed_config)
            else:
                result = invoke(args, kwargs, repo=repo, on_raw_result=_reject_live_result)
        else:
            plan = compile_signature(target)
            call_args, call_kwargs = plan.prepare_args(
                args, kwargs, repo=repo, reuse_live="never", selections=selections,
            ).deliver_args(
                extra_mat_roots=capture_roots, on_extra_materialized=install_captures,
            )
            result = _reject_live_result(target(*call_args, **call_kwargs))
            result = plan.prepare_return(result, repo=repo, reuse_live="never").deliver_return()
    except SignatureError:
        raise
    encoded = _dill_result(result, "$.result")
    if len(encoded) > result_limit_bytes:
        raise CoreCallCodecError("core execution transport rejected oversized result")
    return encoded


def _reject_live_result(value: Any) -> Any:
    """Keep U5 from falling back to live-result pickling before U6 publication."""
    active: set[int] = set()

    def visit(current: Any) -> None:
        identity = id(current)
        if identity in active:
            return
        active.add(identity)
        try:
            if isinstance(current, (Object, Repo, Store)):
                raise CoreCallCodecError("core execution result publication is required for live resource output")
            if isinstance(current, Mapping):
                for key, item in current.items():
                    visit(key)
                    visit(item)
            elif isinstance(current, (tuple, list, set, frozenset)):
                for item in current:
                    visit(item)
            else:
                fields = _instance_fields(current)
                if fields is not None:
                    for item in fields.values():
                        visit(item)
        finally:
            active.remove(identity)

    visit(value)
    return value


__all__ = ["CoreCallCodecError", "decode_invocation", "encode_invocation", "invoke_invocation"]
