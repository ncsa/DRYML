"""Managed-owned, versioned Core Execute extension and metadata admission handoff.

Owning values explicitly supply ``ADAPTER`` to Core's transport seam. The adapter
owns closed tags, config resources and detached result metadata. It never encodes
open handles or ordinary durable result bytes. Targets reconstruct metadata only;
the managed runtime retains one operation owner through execution realization.
Malformed/version-incompatible input raises CoreCallCodecError before reconstruction.
"""

from collections.abc import Mapping
from dataclasses import fields
import inspect
import types
from uuid import UUID

from dryml.core.execute_codec import CoreCallCodecError, _fail, _import_ref
from dryml.core.node import require_saved
from dryml.core.repo import Repo
from dryml.core.repo_definition import RepoDefinition, _validate_store_descriptor
from dryml.core.store.dir import DirStore
from dryml.core.store.store import Store
from dryml.core.symbol import ImportRef

from .config import ManagedConfig
from .descriptor import ManagedOperation, _ManagedComposite, _BoundOperation, _BoundComposite, _unique_sources
from .model import InvocationResult, ManagedStatus, ExecutionIdentity, FailedInvocationResult
from .errors import ManagedError
from .results import ResultReference, ResultSource

ADAPTER = "dryml.managed.execute_adapter:ManagedExecuteAdapter"


def _resource(value, *, allow_repo, path):
    if value is None:
        return None
    if allow_repo and isinstance(value, Repo):
        definition = value.to_definition().to_data()
        role, descriptors = "repo", definition["stores"]
    elif type(value) is DirStore:
        definition = value.to_definition()
        role, descriptors = "store", [definition]
    else:
        _fail("unsupported managed config resource", path)
    if any(v.get("kind") != "dir" for v in descriptors):
        _fail("unsupported managed config resource", path)
    return {"role": role, "definition": definition}


def _validate_resource(value, *, allow_repo, path):
    if value is None:
        return
    try:
        if not isinstance(value, Mapping) or set(value) != {"role", "definition"}:
            raise ValueError()
        if value["role"] == "repo" and allow_repo:
            descriptors = RepoDefinition.from_data(value["definition"]).to_data()["stores"]
        elif value["role"] == "store":
            descriptors = [_validate_store_descriptor(value["definition"])]
        else:
            raise ValueError()
    except (RecursionError, TypeError, ValueError, OverflowError, UnicodeError):
        _fail("malformed managed config resource", path)
    if any(v["kind"] != "dir" for v in descriptors):
        _fail("unsupported managed config resource", path)


def _open_resource(value):
    if value is None:
        return None
    if value["role"] == "repo":
        return Repo.from_definition(RepoDefinition.from_data(value["definition"]))
    return Store.from_definition(value["definition"])


class _MetadataTarget:
    """Invocation-local borrowed receiver metadata; never a reconstructed Object."""

    __dryml_execute_owner__ = "managed"
    __dryml_execute_adapter__ = ADAPTER

    def __init__(self, receiver, evidence, repo):
        from .storage import _caller
        self.receiver, self.evidence, self.repo = receiver, evidence, repo
        self._caller = _caller()
        self._descriptor = evidence._descriptor if type(evidence) is _ManagedComposite else evidence

    def __call__(self, *args, **kwargs):
        from .runtime import invoke
        from .storage import _caller
        from dryml.core.execute import _current_context, current_context
        if self._caller != _caller():
            raise RuntimeError("managed transported target belongs to a different process, thread or task")
        if _current_context.get() is not None:
            current_context()
        if self.repo._closed or self.repo._closing:
            raise RuntimeError("managed transported target resource scope has expired")
        from dryml.core.symbol import resolve_symbol
        receiver_type = resolve_symbol(self.receiver.cls)
        if not isinstance(receiver_type, type) or not issubclass(receiver_type, self._descriptor._owner):
            _fail("incompatible managed receiver declaration", "$.target")
        config = kwargs.pop("managed", None)
        return invoke(self._descriptor, self.receiver, args, config, kwargs,
            realize=lambda node: self.repo.materialize_boundary((node,), reuse_live="never")[0],
            outer=self.evidence._outer if type(self.evidence) is _ManagedComposite else None)

    def __dryml_execute_invoke__(self, args, kwargs, *, repo, on_raw_result):
        return self(*args, **kwargs)


class ManagedExecuteAdapter:
    """Closed managed transport v2 supplied by managed values, not Core imports.

    Core calls encode/validate/decode with its bounded graph owner. Validation
    returns graph references for Core range checks; reconstruction opens config
    descriptors only after the entire graph and adapter version validate. Result
    decoding requires result-reference v2, returns detached InvocationResult
    metadata and performs no result IO. Adapter v1 is incompatible.
    All methods raise CoreCallCodecError for malformed owned wire data; supported
    source opening and managed invocation errors retain their owning types.
    """

    version = 2
    schemas = {
        "managed_declaration": {"tag", "authored", "executable", "owner", "member", "resumable", "store_parameter", "invocation_parameters"},
        "managed_composite": {"tag", "declaration", "outer", "owner", "member"},
        "managed_target": {"tag", "receiver", "declaration"},
        "managed_composite_target": {"tag", "receiver", "composite"},
        "managed_config": {"tag", "state_repo", "control_store", "rerun", "start_from", "callbacks"},
        "managed_result": {"tag", "status", "execution", "final", "reference", "source", "cleanup"},
        "managed_metadata": {"tag", "kind", "fields"},
        "managed_recovery": {"tag", "disposition", "run_id", "invocation_id", "current", "cleanup"},
    }

    @staticmethod
    def encode(encoder, value, path, depth):
        """Encode a supported exact managed value as a closed node dictionary.

        encoder is Core's bounded graph owner; path is structural text and depth
        an integer budget position. Child values use its shared tables. Config
        resources detach to descriptors; malformed/unsupported values raise
        CoreCallCodecError. No state capture or receiver reconstruction occurs.
        """
        put = lambda item, name: encoder.value(item, f"{path}.{name}", depth + 1)
        if type(value) in {ManagedStatus, ExecutionIdentity, FailedInvocationResult}:
            return {"tag": "managed_metadata", "kind": type(value).__name__,
                "fields": {f.name: put(getattr(value, f.name), f.name) for f in fields(value)}}
        from .runtime import AdmissionOutcome
        if type(value) is AdmissionOutcome:
            return {"tag": "managed_recovery", "disposition": value.disposition, "run_id": value.run_id,
                "invocation_id": value.invocation_id, "current": None if value.current is None else value.current.to_bytes(),
                "cleanup": _cleanup_data(value.cleanup_report)}
        if type(value) is ManagedConfig:
            if encoder.purpose != "call":
                _fail("managed config result", path)
            if type(value.rerun) is not bool:
                _fail("malformed managed config", path)
            callbacks = value.callbacks
            if callbacks is not None:
                from dryml.core._callable_inspection import describe_callable
                if type(callbacks) is not list or len(callbacks) > 64:
                    _fail("malformed managed config", path)
                for callback in callbacks:
                    if not callable(callback) or describe_callable(callback).native_modality != "sync":
                        _fail("unsupported managed callback", path)
            if value.start_from is not None:
                require_saved(value.start_from)
            return {"tag": "managed_config", "rerun": value.rerun,
                "start_from": None if value.start_from is None else put(value.start_from, "start_from"),
                "callbacks": None if callbacks is None else [put(v, f"callback[{i}]") for i, v in enumerate(callbacks)],
                "state_repo": _resource(value.state_repo, allow_repo=True, path=path),
                "control_store": _resource(value.control_store, allow_repo=False, path=path)}
        if type(value) is ManagedOperation:
            owner, member = value._owner, value.member
            if not isinstance(owner, type) or not isinstance(member, str):
                _fail("unbound managed declaration", path)
            ref = _import_ref(owner)
            if ref is not None and ref.module.startswith("dryml.") and inspect.getattr_static(owner, member, None) is value:
                for name in ("_target", "_executable"):
                    encoder.imported_captures[id(getattr(value, name))] = ImportRef(ref.module, f"{ref.qualname}.{member}.{name}")
            return {"tag": "managed_declaration", "authored": put(value._target, "authored"),
                "executable": put(value._executable, "executable"), "owner": put(owner, "owner"),
                "member": member, "resumable": value.resumable, "store_parameter": value.store_parameter,
                "invocation_parameters": dict(value._invocation_parameters)}
        if type(value) is _ManagedComposite:
            return {"tag": "managed_composite", "declaration": put(value._descriptor, "declaration"),
                "outer": put(value._outer, "outer"), "owner": put(value._descriptor._owner, "owner"),
                "member": value._descriptor.member}
        if type(value) in {_BoundOperation, _BoundComposite}:
            receiver = encoder.repo.describe(value._instance)
            # The receipt is metadata, never an availability or restore proof.
            saved = value._instance.last_state_ref
            from dryml.core.cdef_graph import has_stateful_materialization
            if has_stateful_materialization(value._instance.definition) and saved is None:
                _fail("unsaved live Object", path)
            if saved is not None and receiver.object_projection_equal(saved):
                receiver = saved
            composite = type(value) is _BoundComposite
            field = "composite" if composite else "declaration"
            return {"tag": "managed_composite_target" if composite else "managed_target",
                "receiver": put(receiver, "receiver"), field: put(value._composite if composite else value._descriptor, field)}
        if type(value) is InvocationResult:
            cleanup = _cleanup_data(value.cleanup_report)
            return {"tag": "managed_result",
                "status": put(tuple(getattr(value.status, f.name) for f in fields(ManagedStatus)), "status"),
                "execution": put(tuple(getattr(value.execution, f.name) for f in fields(ExecutionIdentity)), "execution"),
                "final": put(value.final_state_ref, "final"), "reference": value.result_reference.to_data(),
                "source": {"path": value.source.path, "store_uuid": value.source.store_uuid}, "cleanup": cleanup}
        _fail("unsupported managed transport value", path)

    @staticmethod
    def validate(decoder, node, path):
        """Return integer child references after inert schema/policy validation.

        decoder supplies the bounded raw node table, node is a Mapping and path
        structural text. Malformed fields/resources/references raise
        CoreCallCodecError; no source or receiver is opened or reconstructed.
        """
        tag, refs = node["tag"], []
        def require(field, tags):
            index = node[field]
            if type(index) is not int or not 0 <= index < len(decoder.nodes):
                _fail("malformed graph reference", path)
            candidate = decoder.nodes[index]
            if not isinstance(candidate, Mapping) or type(candidate.get("tag")) is not str or candidate.get("tag") not in tags:
                _fail("malformed managed evidence", path)
            if candidate.get("tag") == "import" and type(candidate.get("qualname")) is not str:
                _fail("malformed managed evidence", path)
            refs.append(index)
        if tag == "managed_metadata":
            kinds = {v.__name__: v for v in (ManagedStatus, ExecutionIdentity, FailedInvocationResult)}
            kind = node["kind"]
            if type(kind) is not str or kind not in kinds or type(node["fields"]) is not dict or set(node["fields"]) != {f.name for f in fields(kinds[kind])}:
                _fail("malformed managed metadata", path)
            for name, index in node["fields"].items():
                if type(index) is not int or not 0 <= index < len(decoder.nodes):
                    _fail("malformed graph reference", path)
                candidate = decoder.nodes[index]
                expected = ({"managed_metadata"} if name in {"status", "execution"}
                            else {"atom", "node"} if name in {"checkpoint", "final", "checkpoint_state_ref"} else {"atom"})
                if not isinstance(candidate, Mapping) or type(candidate.get("tag")) is not str or candidate.get("tag") not in expected:
                    _fail("malformed managed metadata", path)
                refs.append(index)
            return refs
        if tag == "managed_recovery":
            from .control import ControlSnapshot
            try:
                if node["disposition"] not in {"preparation_abandoned", "run_admitted", "completion_recorded", "projection_reconciled", "finalization_interrupted", "finalization_incomplete"}:
                    raise ValueError()
                _uuid_hex(node["invocation_id"])
                if node["run_id"] is not None:
                    _uuid_hex(node["run_id"])
                if node["current"] is not None:
                    current = ControlSnapshot.from_bytes(node["current"])
                    if current.run_id != node["run_id"]:
                        raise ValueError()
                _restore_cleanup(node["cleanup"])
            except (TypeError, ValueError, ManagedError):
                _fail("malformed managed recovery", path)
            return refs
        if tag == "managed_config":
            if decoder.purpose != "call":
                _fail("managed config result", path)
            if type(node["rerun"]) is not bool or (node["callbacks"] is not None and
                    (type(node["callbacks"]) is not list or len(node["callbacks"]) > 64)):
                _fail("malformed managed config", path)
            _validate_resource(node["state_repo"], allow_repo=True, path=path)
            _validate_resource(node["control_store"], allow_repo=False, path=path)
            if node["start_from"] is not None:
                require("start_from", {"node"})
            refs.extend(node["callbacks"] or [])
        elif tag == "managed_declaration":
            if (type(node["member"]) is not str or not node["member"].isidentifier()
                    or type(node["resumable"]) is not bool
                    or (node["store_parameter"] is not None and type(node["store_parameter"]) is not str)
                    or not isinstance(node["invocation_parameters"], dict)
                    or any(type(k) is not str or type(v) is not str or v not in {"observer_factories", "observer_strict"}
                           for k, v in node["invocation_parameters"].items())):
                _fail("malformed managed declaration", path)
            require("authored", {"function", "import"})
            require("executable", {"function", "function_owner", "import"})
            require("owner", {"class", "import"})
        elif tag == "managed_composite":
            if type(node["member"]) is not str:
                _fail("malformed managed composite", path)
            require("declaration", {"managed_declaration"})
            require("outer", {"function"})
            require("owner", {"class", "import"})
        elif tag in {"managed_target", "managed_composite_target"}:
            require("receiver", {"node"})
            require("declaration" if tag == "managed_target" else "composite",
                    {"managed_declaration"} if tag == "managed_target" else {"managed_composite"})
        elif tag == "managed_result":
            require("status", {"tuple"})
            require("execution", {"tuple"})
            require("final", {"node"})
            for field, metadata_type in (("status", ManagedStatus), ("execution", ExecutionIdentity)):
                sequence = decoder.nodes[node[field]]["items"]
                metadata_fields = fields(metadata_type)
                if type(sequence) is not list or len(sequence) != len(metadata_fields):
                    _fail("malformed managed result", path)
                for metadata_field, index in zip(metadata_fields, sequence):
                    if type(index) is not int or not 0 <= index < len(decoder.nodes):
                        _fail("malformed graph reference", path)
                    candidate = decoder.nodes[index]
                    tags = {"atom", "node"} if metadata_field.name in {"checkpoint", "final"} else {"atom"}
                    if not isinstance(candidate, Mapping) or type(candidate.get("tag")) is not str or candidate.get("tag") not in tags:
                        _fail("malformed managed result", path)
            try:
                ResultReference.from_data(node["reference"])
                if type(node["source"]) is not dict or set(node["source"]) != {"path", "store_uuid"}:
                    raise ValueError()
                ResultSource(**node["source"])
                _restore_cleanup(node["cleanup"])
            except (TypeError, ValueError, ManagedError):
                _fail("malformed managed result", path)
        return refs

    @staticmethod
    def validate_views(decoder, node, path):
        """Validate Node capabilities after shared decode, before reconstruction.

        decoder owns validated immutable graph_views; node/path identify an owned
        wire node. Returns None or raises CoreCallCodecError for non-saved starts,
        incomplete receiver identities or invalid final metadata. All work is
        metadata-only; it resolves no class, opens no Store and invokes no hooks.
        """
        def graph(field):
            return decoder.graph_views[decoder.nodes[node[field]]["view"]]
        try:
            if node["tag"] == "managed_config" and node["start_from"] is not None:
                require_saved(graph("start_from"))
            elif node["tag"] in {"managed_target", "managed_composite_target"}:
                from .identity import _receiver
                _receiver(graph("receiver"))
            elif node["tag"] == "managed_result":
                require_saved(graph("final"))
        except (TypeError, ValueError, ManagedError) as error:
            raise CoreCallCodecError(f"core execution transport rejected malformed managed Node capability at {path}") from error

    @staticmethod
    def decode(decoder, node, path):
        """Return the validated owned value using Core's decoder and text path.

        Targets retain Node metadata only; configs reopen descriptors through the
        active resource scope; result/recovery metadata retains no open handles.
        Malformed records raise CoreCallCodecError and unavailable resource opens
        preserve their owning errors. This method never loads result bytes.
        """
        get = lambda name: decoder.value(node[name], f"{path}.{name}")
        tag = node["tag"]
        if tag == "managed_metadata":
            kinds = {v.__name__: v for v in (ManagedStatus, ExecutionIdentity, FailedInvocationResult)}
            value = kinds[node["kind"]](**{name: decoder.value(index, path) for name, index in node["fields"].items()})
            try:
                if type(value) is ManagedStatus:
                    _validate_status(value)
                elif type(value) is ExecutionIdentity:
                    _validate_execution(value)
                elif (type(value.status) is not ManagedStatus or type(value.execution) is not ExecutionIdentity
                        or value.status.operation_id != value.execution.operation_id
                        or value.status.run_id != value.execution.run_id):
                    raise ValueError()
            except (TypeError, ValueError):
                _fail("malformed managed metadata", path)
            return value
        if tag == "managed_recovery":
            from .runtime import AdmissionOutcome
            from .control import ControlSnapshot
            return AdmissionOutcome(node["disposition"], node["run_id"], node["invocation_id"],
                None if node["current"] is None else ControlSnapshot.from_bytes(node["current"]), _restore_cleanup(node["cleanup"]))
        if tag == "managed_config":
            return ManagedConfig(state_repo=_open_resource(node["state_repo"]),
                control_store=_open_resource(node["control_store"]), rerun=node["rerun"],
                start_from=None if node["start_from"] is None else get("start_from"),
                callbacks=None if node["callbacks"] is None else [decoder.value(i, path) for i in node["callbacks"]])
        if tag == "managed_declaration":
            authored, executable, owner = get("authored"), get("executable"), get("owner")
            if type(authored) is not types.FunctionType or type(executable) is not types.FunctionType or not isinstance(owner, type):
                _fail("malformed managed declaration", path)
            _restore_wrapper_chain(executable, authored)
            declaration = ManagedOperation(executable, resumable=node["resumable"], store_parameter=node["store_parameter"])
            if node["invocation_parameters"]:
                declaration._declare_invocation_parameters(**node["invocation_parameters"])
            declaration.__set_name__(owner, node["member"])
            if declaration.author_signature != inspect.signature(authored):
                _fail("malformed managed declaration", path)
            declaration._target = authored
            return declaration
        if tag == "managed_composite":
            declaration, outer, owner = get("declaration"), get("outer"), get("owner")
            if (type(declaration) is not ManagedOperation or type(outer) is not types.FunctionType
                    or declaration._owner is not owner or declaration.member != node["member"]):
                _fail("malformed managed composite", path)
            _restore_wrapper_chain(outer, declaration)
            return _ManagedComposite(declaration, outer)
        if tag in {"managed_target", "managed_composite_target"}:
            receiver = get("receiver")
            evidence = get("declaration" if tag == "managed_target" else "composite")
            return _MetadataTarget(receiver, evidence, decoder.repo)
        if tag == "managed_result":
            status, execution, final = ManagedStatus(*get("status")), ExecutionIdentity(*get("execution")), get("final")
            require_saved(final)
            if (status.state != "completed" or status.operation_id != execution.operation_id
                    or status.run_id != execution.run_id or status.producer_invocation_id != execution.producer_invocation_id
                    or status.final is None or not status.final.graph_equal(final)
                    or type(status.generation) is not int or status.generation < 1):
                _fail("malformed managed result", path)
            try:
                _validate_status(status)
                _validate_execution(execution)
            except (TypeError, ValueError):
                _fail("malformed managed result identity", path)
            return InvocationResult(status, execution, final, ResultReference.from_data(node["reference"]),
                                    ResultSource(**node["source"]), _restore_cleanup(node["cleanup"]))
        _fail("unknown managed tag", path)

    @staticmethod
    def prepare_call(target, kwargs):
        """Return (detached controls, ordinary kwargs) for a supplied bound target.

        target is a managed bound view and kwargs a caller mapping. The returned
        dictionaries are owned snapshots. Invalid Store or invocation-only
        controls raise CoreCallCodecError before submission; no resource opens.
        """
        descriptor = getattr(target, "_descriptor", None)
        values = dict(kwargs)
        if type(descriptor) is not ManagedOperation:
            return None, values
        _validate_invocation_parameters(descriptor, values)
        name = descriptor.store_parameter
        store = values.pop(name, None) if name is not None else None
        if store is not None and not isinstance(store, Store):
            _fail("unsupported declared managed Store", "$.controls")
        return {"store_parameter": name, "store": None if store is None else store.to_definition()}, values

    @staticmethod
    def restore_call(target, controls, kwargs):
        """Validate/reopen declared controls and update worker kwargs; return None.

        target is a metadata-only bound target, controls a detached mapping and
        kwargs the worker-owned dict. Invalid slots raise CoreCallCodecError;
        unavailable existing Store descriptors preserve their owning open errors.
        Open handles are owned by the active resource cache, never serialized.
        """
        expected = target._descriptor.store_parameter
        if (type(controls) is not dict or set(controls) != {"store_parameter", "store"}
                or controls["store_parameter"] != expected or expected is None and controls["store"] is not None):
            _fail("forbidden managed Store control", "$.controls")
        if expected is not None:
            kwargs[expected] = None if controls["store"] is None else Store.from_definition(_validate_store_descriptor(controls["store"]))


def _validate_invocation_parameters(descriptor, kwargs):
    from dryml.core.factory import FactorySpec
    for name, policy in descriptor._invocation_parameters.items():
        if name not in kwargs:
            continue
        value = kwargs[name]
        path = f"$.kwargs[{name!r}]"
        if policy == "observer_strict":
            if type(value) is not bool:
                _fail("malformed observer strict policy", path)
        elif value is not None:
            if type(value) not in {list, tuple} or len(value) > 64:
                _fail("malformed observer factory collection", path)
            for factory in value:
                if type(factory) is not FactorySpec:
                    _fail("live telemetry observer; use FactorySpec for worker construction", path)
                _validate_factory(factory, path, 0)


def _validate_factory(value, path, depth):
    from pathlib import PurePath
    from dryml.core.factory import FactorySpec
    from dryml.core.symbol import SourceSpec
    if depth > 32:
        _fail("observer factory configuration depth", path)
    if value is None or type(value) in {bool, int, float, str, bytes} or isinstance(value, (ImportRef, SourceSpec, PurePath)):
        return
    if isinstance(value, FactorySpec):
        children = (value.target, *value.args, *value.kwargs.values())
    elif isinstance(value, Mapping):
        children = (*value.keys(), *value.values())
    elif isinstance(value, (tuple, list, set, frozenset)):
        children = value
    else:
        _fail("live observer factory configuration", path)
    for child in children:
        _validate_factory(child, path, depth + 1)


def _restore_wrapper_chain(executable, target):
    """Restore only retained forwarding evidence leading to the selected target."""
    from dryml.core._callable_inspection import describe_callable
    def reaches(current, active):
        if current is target:
            return True
        if type(current) is not types.FunctionType or id(current) in active:
            return False
        description = describe_callable(current)
        if description.owner == "function":
            return description.raw_target is target
        values = [*(current.__defaults__ or ()), *(current.__kwdefaults__ or {}).values()]
        if current.__closure__:
            values.extend(cell.cell_contents for cell in current.__closure__)
        candidates = [value for value in values if reaches(value, active | {id(current)})]
        candidates = _unique_sources(*candidates)
        if len(candidates) != 1:
            return False
        current.__wrapped__ = candidates[0]
        return True
    if not reaches(executable, set()):
        _fail("malformed managed wrapper relationship", "$.declaration")


def _uuid_hex(value):
    if type(value) is not str or UUID(value).hex != value:
        raise ValueError("invalid execution identity")


def _validate_execution(value):
    import re
    if type(value.operation_id) is not str or re.fullmatch("[0-9a-f]{64}", value.operation_id) is None:
        raise ValueError()
    _uuid_hex(value.run_id)
    _uuid_hex(value.invocation_id)
    if value.producer_invocation_id is not None:
        _uuid_hex(value.producer_invocation_id)


def _validate_status(value):
    if value.state == "not_started":
        import re
        if (type(value.operation_id) is not str or re.fullmatch("[0-9a-f]{64}", value.operation_id) is None
                or value.run_id is not None or value.invocation_id is not None or value.producer_invocation_id is not None
                or type(value.generation) is not int or value.generation != 0
                or value.checkpoint is not None or value.final is not None
                or value.interruption_requested is not False or value.failure_code is not None):
            raise ValueError()
        return
    _validate_execution(ExecutionIdentity(value.operation_id, value.run_id, value.invocation_id, value.producer_invocation_id))
    if (type(value.generation) is not int or not 1 <= value.generation < 2**63
            or value.state not in {"running", "finalizing", "completed", "failed", "interrupted"}
            or type(value.interruption_requested) is not bool
            or value.failure_code is not None and type(value.failure_code) is not str):
        raise ValueError()
    if value.state == "completed" and (value.producer_invocation_id is None or value.final is None
            or value.interruption_requested or value.failure_code is not None):
        raise ValueError()
    for selection in (value.checkpoint, value.final):
        if selection is not None:
            require_saved(selection)


def _cleanup_data(report):
    return [(domain, key.cdef_digest, key.codec_hash, outcome.status, outcome.incarnation,
             None if outcome.cleanup_error is None else outcome.cleanup_error.errno)
            for domain, key, outcome in report]


def _restore_cleanup(rows):
    from dryml.core.state_values import StateKey
    from dryml.core.store.retention import StateRemovalOutcome
    if type(rows) is not list:
        raise ValueError()
    result = []
    for row in rows:
        if type(row) is not tuple or len(row) != 6:
            raise ValueError()
        domain, cdef, codec, kind, incarnation, errno = row
        if (type(domain) is not str or str(UUID(domain)) != domain or kind != "cleanup-pending"
                or type(incarnation) is not str or str(UUID(incarnation)) != incarnation
                or errno is not None and type(errno) is not int):
            raise ValueError()
        result.append((domain, StateKey(cdef, codec), StateRemovalOutcome(kind, incarnation,
            OSError(errno, "physical garbage cleanup failed"))))
    return tuple(result)
