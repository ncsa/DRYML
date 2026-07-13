"""Explicit bounded tracing of trusted orchestration functions.

Tracing runs a supported synchronous Python function exactly once in the current
process.  It is not a sandbox and provides no timeout: target code can perform
arbitrary side effects or block forever.  Definition proxies observe only direct
method calls and never build a receiver or invoke its real method.
"""

from __future__ import annotations

import contextvars
import importlib
import inspect
import json
import math
import threading
import types
import uuid
from dataclasses import dataclass
from typing import Any

from dryml.annotations import (
    fragments_for_definition_method,
    fragments_for_method,
    resolve_fragments,
)
from dryml.code.analysis import (
    CodeAnalysisContext,
    CodeAnalysisError,
    CodeAnalysisResult,
    FunctionAnalyzer,
)
from dryml.code.facts import (
    AnnotationFact,
    CodeFact,
    DiagnosticFact,
    DynamicCallFact,
    MethodContractFact,
    RequirementFact,
    ShapeFact,
    _validate_dynamic_method_fact_wire,
)
from dryml.code.targets import (
    CodeTarget,
    CodeTargetSpec,
    target_from_definition_method,
)
from dryml.core2.definition import ConcreteDefinition, Definition
from dryml.core2.symbol import ImportRef, SourceSpec
from dryml.core2.utils.stable_hash import (
    StableHashLimitError,
    StableHashLimits,
    bounded_stable_hash_function,
)
from dryml.formats.refs import format_cdef_id


MAX_CALLS = 10_000
MAX_VALUE_DEPTH = 32
MAX_CONTAINER_ENTRIES = 10_000
MAX_STRING_CHARS = 4_096
MAX_INTEGER_BITS = 4_096
MAX_METHOD_NAME_CHARS = 512
MAX_REFERENCE_CHARS = 4_096
MAX_METHOD_FACTS = 256
MAX_CALL_FACT_BYTES = 1_048_576
MAX_RESULT_BYTES = 16_777_216
MAX_DIAGNOSTICS = 256
MAX_DIAGNOSTIC_CHARS = 1_024
MAX_HASH_DEPTH = 128
MAX_HASH_OCCURRENCES = 100_000
MAX_HASH_EDGES = 200_000
MAX_HASH_ENCODED_BYTES = 4_194_304

_HASH_LIMITS = StableHashLimits(
    max_depth=MAX_HASH_DEPTH,
    max_occurrences=MAX_HASH_OCCURRENCES,
    max_edges=MAX_HASH_EDGES,
    max_encoded_bytes=MAX_HASH_ENCODED_BYTES,
    max_integer_bits=MAX_INTEGER_BITS,
    max_string_chars=MAX_STRING_CHARS,
)


@dataclass(frozen=True, slots=True)
class DynamicTracePolicy:
    """Validated bounds and collection switches for :func:`dryml.code.trace`.

    Args:
        max_calls: Maximum observed calls, from 1 through the hard 10,000-call
            ceiling.  The N+1 call aborts execution and is not recorded.
        require_proxy_only_args: Require Definition/CDef leaves in invocation
            arguments. Empty containers remain valid. When false, bounded JSON
            scalar leaves are additionally accepted.
        collect_requirements: Collect current annotation and requirement facts
            for observed methods. Method contracts remain controlled separately
            by :class:`CodeAnalysisContext`.
    """

    max_calls: int = MAX_CALLS
    require_proxy_only_args: bool = True
    collect_requirements: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.max_calls, bool) or not isinstance(self.max_calls, int):
            raise TypeError("max_calls must be an integer")
        if self.max_calls < 1 or self.max_calls > MAX_CALLS:
            raise ValueError("max_calls must be between 1 and 10000")
        if type(self.require_proxy_only_args) is not bool:
            raise TypeError("require_proxy_only_args must be bool")
        if type(self.collect_requirements) is not bool:
            raise TypeError("collect_requirements must be bool")


class DynamicTraceProxyError(CodeAnalysisError):
    """Raised when an escaped trace proxy is used outside its owning run."""

    code = "dryml.code.dynamic_trace_stale_proxy"

    def __init__(self):
        super().__init__("Dynamic trace proxy is not active in its owning trace.")


@dataclass(frozen=True, slots=True)
class _InvocationRequest:
    target: CodeTarget
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    context: CodeAnalysisContext
    policy: DynamicTracePolicy


class _TraceAbort(Exception):
    """Private cooperative abort raised after a trace-domain diagnostic."""


class _PrevalidationFailure(Exception):
    def __init__(self, diagnostic: DiagnosticFact):
        super().__init__(diagnostic.code)
        self.diagnostic = diagnostic


class _ReferenceLimitError(ValueError):
    def __init__(self, limit_name: str, limit: int, observed_lower_bound: int):
        super().__init__(f"dynamic trace {limit_name} limit exceeded")
        self.limit_name = limit_name
        self.limit = limit
        self.observed_lower_bound = observed_lower_bound


class _MethodFactLimitError(ValueError):
    def __init__(self, limit_name: str, limit: int, observed_lower_bound: int):
        super().__init__(f"dynamic trace {limit_name} limit exceeded")
        self.limit_name = limit_name
        self.limit = limit
        self.observed_lower_bound = observed_lower_bound


_CURRENT_PLANNER: contextvars.ContextVar[_Planner | None] = contextvars.ContextVar(
    "dryml_code_dynamic_trace_planner", default=None
)


def _diagnostic(code: str, message: str, *, target_kind: str, data: dict[str, Any] | None = None) -> DiagnosticFact:
    if len(code) > MAX_DIAGNOSTIC_CHARS or len(message) > MAX_DIAGNOSTIC_CHARS:
        raise RuntimeError("dynamic trace diagnostic constants exceed their bound")
    return DiagnosticFact(
        severity="error",
        code=code,
        message=message,
        source={"analyzer": "dynamic_trace", "target_kind": target_kind[:MAX_STRING_CHARS]},
        data=data or {},
    )


def _validate_method_fact_value(
    value: Any,
    *,
    depth: int = 0,
    active: set[int] | None = None,
    counter: list[int] | None = None,
) -> None:
    """Validate method-fact JSON before canonical serialization."""

    if depth > MAX_VALUE_DEPTH:
        raise _MethodFactLimitError(
            "method_fact_depth", MAX_VALUE_DEPTH, depth
        )
    if value is None or type(value) is bool:
        return
    if type(value) is int:
        bits = value.bit_length()
        if bits > MAX_INTEGER_BITS:
            raise _MethodFactLimitError(
                "method_fact_integer_bits", MAX_INTEGER_BITS, bits
            )
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("method fact float must be finite")
        return
    if type(value) is str:
        if len(value) > MAX_STRING_CHARS:
            raise _MethodFactLimitError(
                "method_fact_string_chars", MAX_STRING_CHARS, len(value)
            )
        return
    if type(value) not in {list, dict}:
        raise ValueError("method fact contains a non-JSON value")

    active = set() if active is None else active
    counter = [0] if counter is None else counter
    oid = id(value)
    if oid in active:
        raise ValueError("method fact JSON must be acyclic")
    active.add(oid)
    try:
        counter[0] += len(value)
        if counter[0] > MAX_CONTAINER_ENTRIES:
            raise _MethodFactLimitError(
                "method_fact_entries", MAX_CONTAINER_ENTRIES, counter[0]
            )
        children = value
        if type(value) is dict:
            for key in value:
                if type(key) is not str:
                    raise ValueError("method fact JSON keys must be strings")
                if len(key) > MAX_STRING_CHARS:
                    raise _MethodFactLimitError(
                        "method_fact_string_chars", MAX_STRING_CHARS, len(key)
                    )
            children = value.values()
        for child in children:
            _validate_method_fact_value(
                child,
                depth=depth + 1,
                active=active,
                counter=counter,
            )
    finally:
        active.remove(oid)


def _bounded_method_fact_json(value: Any) -> tuple[str, int]:
    """Return bounded canonical JSON for already validated method-fact data."""

    _validate_method_fact_value(value)
    chunks: list[str] = []
    encoded_size = 0
    encoder = json.JSONEncoder(
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    for chunk in encoder.iterencode(value):
        encoded_size += len(chunk.encode("utf-8"))
        if encoded_size > MAX_CALL_FACT_BYTES:
            raise _MethodFactLimitError(
                "method_fact_bytes", MAX_CALL_FACT_BYTES, encoded_size
            )
        chunks.append(chunk)
    return "".join(chunks), encoded_size


def _validated_method_fact_wire(fact: CodeFact) -> tuple[dict[str, Any], int]:
    """Validate and normalize one supported method fact before call creation."""

    allowed = (AnnotationFact, RequirementFact, MethodContractFact, ShapeFact)
    if not isinstance(fact, allowed):
        raise ValueError("unsupported method fact")

    _validate_method_fact_value(fact.source)
    _validate_method_fact_value(fact.data)
    if isinstance(fact, RequirementFact):
        for value in (
            fact.namespace,
            fact.requirement_kind,
            fact.fragment,
            fact.priority,
            fact.merge_policy,
        ):
            _validate_method_fact_value(value)

    wire = fact.to_data()
    if type(wire) is not dict:
        raise ValueError("method fact wire data must be an exact dict")
    _validate_dynamic_method_fact_wire(wire)

    _validate_method_fact_value(wire)
    restored = CodeFact.from_data(wire)
    if not isinstance(restored, allowed):
        raise ValueError("restored method fact is unsupported")
    normalized = restored.to_data()
    _, encoded_size = _bounded_method_fact_json(normalized)
    return normalized, encoded_size


class _Planner:
    def __init__(self, request: _InvocationRequest):
        self.request = request
        self.run_id = uuid.uuid4().hex
        self.lock = threading.RLock()
        self.state = "active"
        self.outcome = "complete"
        self.facts: list[DynamicCallFact] = []
        self.diagnostics: list[DiagnosticFact] = []
        self.proxy_memo: dict[int, tuple[Definition | ConcreteDefinition, _DefinitionProxy]] = {}
        self.result_bytes = 0
        self.unexpected: Exception | None = None

    @property
    def target_kind(self) -> str:
        value = self.request.target.spec.kind
        return value if isinstance(value, str) and value and len(value) <= MAX_STRING_CHARS else "unknown"

    def add_diagnostic(self, diagnostic: DiagnosticFact) -> None:
        with self.lock:
            if len(self.diagnostics) < MAX_DIAGNOSTICS:
                self.diagnostics.append(diagnostic)
                return
            truncation = _diagnostic(
                "dryml.code.dynamic_trace_diagnostics_limit_exceeded",
                "Dynamic trace diagnostic limit exceeded.",
                target_kind=self.target_kind,
                data={
                    "limit_name": "diagnostics",
                    "limit": MAX_DIAGNOSTICS,
                    "observed_lower_bound": MAX_DIAGNOSTICS + 1,
                },
            )
            self.diagnostics[-1] = truncation
            self.state = "aborted"
            self.outcome = "diagnostics_limit_exceeded"

    def abort(self, outcome: str, code: str, message: str, *, data: dict[str, Any] | None = None) -> None:
        with self.lock:
            if self.state == "active":
                self.add_diagnostic(_diagnostic(code, message, target_kind=self.target_kind, data=data))
                if self.outcome != "diagnostics_limit_exceeded":
                    self.state = "aborted"
                    self.outcome = outcome
        raise _TraceAbort()

    def algorithm_failure(self, exc: Exception) -> None:
        self.unexpected = exc
        self.abort(
            "algorithm_failed",
            "dryml.code.algorithm_failed",
            "Dynamic trace implementation failed.",
        )

    def proxy_for(self, value: Definition | ConcreteDefinition) -> _DefinitionProxy:
        key = id(value)
        existing = self.proxy_memo.get(key)
        if existing is not None and existing[0] is value:
            return existing[1]
        try:
            receiver_class = _resolve_receiver_class(value, self.request.context)
            receiver_class_path = _verified_class_import_path(receiver_class)
            reference = _definition_reference(value)
        except StableHashLimitError as exc:
            raise _PrevalidationFailure(_diagnostic(
                "dryml.code.dynamic_trace_argument_limit_exceeded",
                "Dynamic trace receiver identity limit exceeded.",
                target_kind=self.target_kind,
                data={
                    "limit_name": f"hash_{exc.limit_name}",
                    "limit": exc.limit,
                    "observed_lower_bound": exc.observed_lower_bound,
                },
            )) from None
        except _ReferenceLimitError as exc:
            raise _PrevalidationFailure(_diagnostic(
                "dryml.code.dynamic_trace_receiver_resolution_failed",
                "Dynamic trace receiver class reference limit exceeded.",
                target_kind=self.target_kind,
                data={
                    "limit_name": exc.limit_name,
                    "limit": exc.limit,
                    "observed_lower_bound": exc.observed_lower_bound,
                },
            )) from None
        except Exception:
            raise _PrevalidationFailure(_diagnostic(
                "dryml.code.dynamic_trace_receiver_resolution_failed",
                "Dynamic trace receiver class could not be resolved.",
                target_kind=self.target_kind,
            )) from None
        proxy = _DefinitionProxy(
            self,
            value,
            receiver_class,
            receiver_class_path,
            reference,
        )
        self.proxy_memo[key] = (value, proxy)
        return proxy

    def record_call(self, proxy: _DefinitionProxy, method_name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> _UnsupportedTraceValue:
        try:
            # A foreign current planner must be inspected and aborted under its
            # own lock, before this owner lock is acquired.  Otherwise close on
            # the foreign planner can turn its public stale-proxy outcome into
            # an internal _TraceAbort.
            if _CURRENT_PLANNER.get() is not self:
                _ensure_proxy_owner(self)
            with self.lock:
                # Owner admission and every lifecycle-sensitive recording step
                # share this lock. A copied context that reaches a proxy while
                # close owns the lock observes the closed owner and receives the
                # public escaped-proxy error rather than an internal abort.
                _ensure_proxy_owner(self)
                return self._record_call_locked(proxy, method_name, args, kwargs)
        except (_TraceAbort, DynamicTraceProxyError):
            raise
        except Exception as exc:  # pragma: no cover - defensive framework path
            self.algorithm_failure(exc)
            raise AssertionError("unreachable")
        except BaseException:
            raise

    def _record_call_locked(self, proxy: _DefinitionProxy, method_name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> _UnsupportedTraceValue:
        with self.lock:
            if self.state != "active":
                raise _TraceAbort()
            attempted = len(self.facts) + 1
            if attempted > self.request.policy.max_calls:
                self.abort(
                    "call_limit_exceeded",
                    "dryml.code.dynamic_trace_call_limit_exceeded",
                    "Dynamic trace call limit exceeded.",
                    data={
                        "limit_name": "calls",
                        "limit": self.request.policy.max_calls,
                        "observed_lower_bound": attempted,
                    },
                )
            sequence = len(self.facts)

        encoded_args, encoded_kwargs = _encode_observed_call(self, args, kwargs)
        method_facts = self._collect_method_facts(proxy, method_name)
        data = {
            "sequence": sequence,
            "receiver_kind": proxy._reference["definition_kind"],
            "receiver_ref": proxy._reference["definition_ref"],
            "receiver_class": proxy._receiver_class_path,
            "method_name": method_name,
            "args": encoded_args,
            "kwargs": encoded_kwargs,
            "method_facts": method_facts,
        }
        raw = {"kind": "dynamic_call", "source": {"analyzer": "dynamic_trace", "target_kind": self.target_kind}, "data": data}
        encoded_size = len(json.dumps(raw, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"))
        if encoded_size > MAX_CALL_FACT_BYTES:
            self.abort(
                "result_limit_exceeded",
                "dryml.code.dynamic_trace_result_limit_exceeded",
                "Dynamic trace call fact byte limit exceeded.",
                data={"limit_name": "call_fact_bytes", "limit": MAX_CALL_FACT_BYTES, "observed_lower_bound": encoded_size},
            )
        fact = DynamicCallFact(source=raw["source"], data=data)
        with self.lock:
            total = self.result_bytes + encoded_size
            summary_size = _encoded_summary_size(
                target_kind=self.target_kind,
                outcome=self.outcome,
                calls_recorded=len(self.facts) + 1,
                max_calls=self.request.policy.max_calls,
            )
            if total + summary_size > MAX_RESULT_BYTES:
                self.abort(
                    "result_limit_exceeded",
                    "dryml.code.dynamic_trace_result_limit_exceeded",
                    "Dynamic trace result byte limit exceeded.",
                    data={"limit_name": "result_bytes", "limit": MAX_RESULT_BYTES, "observed_lower_bound": total + summary_size},
                )
            if self.state != "active":
                raise _TraceAbort()
            self.facts.append(fact)
            self.result_bytes = total
        return _UnsupportedTraceValue(self)

    def _collect_method_facts(self, proxy: _DefinitionProxy, method_name: str) -> list[dict[str, Any]]:
        facts: list[CodeFact] = []
        try:
            method_target = target_from_definition_method(
                proxy._reference["definition_ref"],
                proxy._receiver_class,
                method_name,
            )
            if self.request.policy.collect_requirements and self.request.context.include_annotations:
                # The proxy has already resolved ImportRef/SourceSpec receiver
                # classes without building the Definition.  This is the same
                # class-plus-concrete-method collection owned by
                # fragments_for_definition_method for live-class definitions.
                if isinstance(proxy._receiver.cls, type):
                    fragments = fragments_for_definition_method(
                        proxy._receiver,
                        method_name,
                    )
                else:
                    # ImportRef and SourceSpec receivers have already been
                    # permission-checked and resolved without building the
                    # Definition. The accepted class/method collector is the
                    # equivalent current API for that resolved live class.
                    fragments = fragments_for_method(
                        proxy._receiver_class,
                        method_name,
                    )
                unique_fragments = []
                seen: set[str] = set()
                fragment_bytes = 0
                expected_fact_count = 0
                for index, fragment in enumerate(fragments):
                    fragment_data = fragment.to_data()
                    key, key_size = _bounded_method_fact_json(fragment_data)
                    if key in seen:
                        continue
                    fragment_bytes += key_size
                    if fragment_bytes > MAX_CALL_FACT_BYTES:
                        raise _MethodFactLimitError(
                            "method_fact_bytes",
                            MAX_CALL_FACT_BYTES,
                            fragment_bytes,
                        )
                    seen.add(key)
                    expected_fact_count += 1 + int(fragment.kind in {"requirement", "default"})
                    if expected_fact_count > MAX_METHOD_FACTS:
                        self.abort(
                            "method_fact_collection_failed",
                            "dryml.code.dynamic_trace_method_fact_collection_failed",
                            "Dynamic trace method fact limit exceeded.",
                            data={
                                "limit_name": "method_facts",
                                "limit": MAX_METHOD_FACTS,
                                "observed_lower_bound": expected_fact_count,
                            },
                        )
                    unique_fragments.append((index, fragment, fragment_data))
                resolution = resolve_fragments(fragments)
                resolution_data = resolution.to_data()
                _bounded_method_fact_json(resolution_data)
                traces = {}
                for trace in resolution.source_traces:
                    trace_data = trace.to_data()
                    _bounded_method_fact_json(trace_data)
                    traces[trace.fragment_index] = trace_data
                for index, fragment, fragment_data in unique_fragments:
                    source = {
                        "analyzer": "direct_annotations",
                        "target_kind": method_target.spec.kind,
                        "annotation_source": fragment_data.get("source"),
                    }
                    facts.append(AnnotationFact(source=source, data=fragment_data))
                    if fragment.kind in {"requirement", "default"}:
                        facts.append(RequirementFact(
                            namespace=fragment.namespace,
                            requirement_kind=fragment.kind,
                            fragment=fragment.fragment,
                            priority=fragment.priority,
                            merge_policy=fragment.merge_policy,
                            source=source,
                            data={"annotation": fragment_data, "source_trace": traces.get(index), "resolution": resolution_data},
                        ))
                if resolution.diagnostics:
                    raise ValueError("annotation resolution emitted diagnostics")

            if self.request.context.include_method_contracts:
                from dryml.code.algorithms.method_contracts import analyze_target

                class_target = CodeTarget(
                    CodeTargetSpec("class"),
                    obj=proxy._receiver_class,
                )
                result = analyze_target(class_target, self.request.context)
                if not result.ok:
                    raise ValueError("method contract analysis failed")
                if len(facts) + len(result.facts) > MAX_METHOD_FACTS:
                    self.abort(
                        "method_fact_collection_failed",
                        "dryml.code.dynamic_trace_method_fact_collection_failed",
                        "Dynamic trace method fact limit exceeded.",
                        data={
                            "limit_name": "method_facts",
                            "limit": MAX_METHOD_FACTS,
                            "observed_lower_bound": len(facts) + len(result.facts),
                        },
                    )
                facts.extend(result.facts)

            if len(facts) > MAX_METHOD_FACTS:
                self.abort(
                    "method_fact_collection_failed",
                    "dryml.code.dynamic_trace_method_fact_collection_failed",
                    "Dynamic trace method fact limit exceeded.",
                    data={"limit_name": "method_facts", "limit": MAX_METHOD_FACTS, "observed_lower_bound": len(facts)},
                )
            serialized_facts = []
            serialized_bytes = 0
            for fact in facts:
                fact_data, fact_size = _validated_method_fact_wire(fact)
                serialized_bytes += fact_size
                if serialized_bytes > MAX_CALL_FACT_BYTES:
                    raise _MethodFactLimitError(
                        "method_fact_bytes",
                        MAX_CALL_FACT_BYTES,
                        serialized_bytes,
                    )
                serialized_facts.append(fact_data)
            return serialized_facts
        except _TraceAbort:
            raise
        except _MethodFactLimitError as exc:
            self.abort(
                "method_fact_collection_failed",
                "dryml.code.dynamic_trace_method_fact_collection_failed",
                "Dynamic trace method fact limit exceeded.",
                data={
                    "limit_name": exc.limit_name,
                    "limit": exc.limit,
                    "observed_lower_bound": exc.observed_lower_bound,
                },
            )
        except Exception:
            self.abort(
                "method_fact_collection_failed",
                "dryml.code.dynamic_trace_method_fact_collection_failed",
                "Dynamic trace method fact collection failed.",
            )
        raise AssertionError("unreachable")

    def close(self) -> None:
        with self.lock:
            self.state = "closed"


class _DefinitionProxy:
    __slots__ = (
        "_planner",
        "_receiver",
        "_receiver_class",
        "_receiver_class_path",
        "_reference",
    )

    def __init__(
        self,
        planner: _Planner,
        receiver: Definition | ConcreteDefinition,
        receiver_class: type,
        receiver_class_path: str | None,
        reference: dict[str, str],
    ):
        object.__setattr__(self, "_planner", planner)
        object.__setattr__(self, "_receiver", receiver)
        object.__setattr__(self, "_receiver_class", receiver_class)
        object.__setattr__(self, "_receiver_class_path", receiver_class_path)
        object.__setattr__(self, "_reference", reference)

    def __getattribute__(self, name: str):
        if name in {
            "_planner",
            "_receiver",
            "_receiver_class",
            "_receiver_class_path",
            "_reference",
            "_method_attribute",
        }:
            return object.__getattribute__(self, name)
        return object.__getattribute__(self, "_method_attribute")(name)

    def _method_attribute(self, name: str):
        planner = object.__getattribute__(self, "_planner")
        _ensure_proxy_owner(planner)
        # Keep the post-admission lookup and every failure conversion under the
        # owner lock. A copied context can otherwise pass admission, lose the
        # owner to close, then leak the private _TraceAbort from planner.abort.
        with planner.lock:
            _ensure_proxy_owner(planner)
            if isinstance(name, str) and len(name) > MAX_METHOD_NAME_CHARS:
                planner.abort(
                    "unsupported_receiver_attribute",
                    "dryml.code.dynamic_trace_unsupported_receiver_attribute",
                    "Dynamic trace receiver method name limit exceeded.",
                    data={
                        "limit_name": "method_name_chars",
                        "limit": MAX_METHOD_NAME_CHARS,
                        "observed_lower_bound": len(name),
                    },
                )
            if (
                not isinstance(name, str)
                or not name.isidentifier()
                or name in {"definition", "build", "concretize"}
                or name.startswith("__")
                or name.endswith("__")
            ):
                planner.abort(
                    "unsupported_receiver_attribute",
                    "dryml.code.dynamic_trace_unsupported_receiver_attribute",
                    "Dynamic trace receiver attribute is unsupported.",
                )
            cls = object.__getattribute__(self, "_receiver_class")
            try:
                raw = _static_class_attribute(cls, name)
            except AttributeError:
                planner.abort(
                    "unsupported_receiver_attribute",
                    "dryml.code.dynamic_trace_unsupported_receiver_attribute",
                    "Dynamic trace receiver attribute is unsupported.",
                )
            candidate = object.__getattribute__(raw, "__func__") if type(raw) in {staticmethod, classmethod} else raw
            if type(candidate) is not types.FunctionType:
                planner.abort(
                    "unsupported_receiver_attribute",
                    "dryml.code.dynamic_trace_unsupported_receiver_attribute",
                    "Dynamic trace receiver attribute is unsupported.",
                )
            return _MethodProxy(planner, self, name)

    def __repr__(self) -> str:
        return "<dryml dynamic Definition proxy>"


class _MethodProxy:
    __slots__ = ("_planner", "_proxy", "_name")

    def __init__(self, planner: _Planner, proxy: _DefinitionProxy, name: str):
        self._planner = planner
        self._proxy = proxy
        self._name = name

    def __call__(self, *args, **kwargs):
        if type(kwargs) is not dict or any(type(key) is not str for key in kwargs):
            self._planner.abort(
                "unsupported_argument",
                "dryml.code.dynamic_trace_unsupported_argument",
                "Dynamic trace observed an unsupported method argument.",
            )
        return self._planner.record_call(self._proxy, self._name, args, kwargs)

    def __repr__(self) -> str:
        return "<dryml dynamic method proxy>"


class _UnsupportedTraceValue:
    __slots__ = ("_planner",)

    def __init__(self, planner: _Planner):
        object.__setattr__(self, "_planner", planner)

    def _abort(self, category: str = "operation"):
        planner = object.__getattribute__(self, "_planner")
        planner.abort(
            "unsupported_return_operation",
            "dryml.code.dynamic_trace_unsupported_return_operation",
            "Dynamic trace method return value operation is unsupported.",
            data={"operation": category[:MAX_STRING_CHARS]},
        )

    def __repr__(self):
        return "<dryml unsupported trace value>"

    def __str__(self):
        return "<dryml unsupported trace value>"

    def __getattribute__(self, name):
        if name in {"_planner", "_abort", "__repr__", "__str__"}:
            return object.__getattribute__(self, name)
        return object.__getattribute__(self, "_abort")("attribute_access")

    def __bool__(self): return self._abort("truth_test")
    def __len__(self): return self._abort("length")
    def __iter__(self): return self._abort("iteration")
    def __next__(self): return self._abort("iteration")
    def __aiter__(self): return self._abort("async_iteration")
    def __anext__(self): return self._abort("async_iteration")
    def __getitem__(self, key): return self._abort("indexing")
    def __setitem__(self, key, value): return self._abort("item_assignment")
    def __delitem__(self, key): return self._abort("item_assignment")
    def __call__(self, *args, **kwargs): return self._abort("call")
    def __await__(self): return self._abort("await")
    def __enter__(self): return self._abort("context_manager")
    def __exit__(self, *args): return self._abort("context_manager")
    def __aenter__(self): return self._abort("async_context_manager")
    def __aexit__(self, *args): return self._abort("async_context_manager")
    def __int__(self): return self._abort("conversion")
    def __float__(self): return self._abort("conversion")
    def __complex__(self): return self._abort("conversion")
    def __bytes__(self): return self._abort("conversion")
    def __index__(self): return self._abort("conversion")
    def __hash__(self): return self._abort("hash")
    def __format__(self, spec): return self._abort("format")
    def __contains__(self, item): return self._abort("containment")

    def _arithmetic(self, *args, **kwargs): return self._abort("arithmetic")
    def _comparison(self, *args, **kwargs): return self._abort("comparison")

    __add__ = __radd__ = __sub__ = __rsub__ = __mul__ = __rmul__ = _arithmetic
    __matmul__ = __rmatmul__ = __truediv__ = __rtruediv__ = _arithmetic
    __floordiv__ = __rfloordiv__ = __mod__ = __rmod__ = __divmod__ = __rdivmod__ = _arithmetic
    __pow__ = __rpow__ = __lshift__ = __rlshift__ = __rshift__ = __rrshift__ = _arithmetic
    __and__ = __rand__ = __xor__ = __rxor__ = __or__ = __ror__ = _arithmetic
    __neg__ = __pos__ = __abs__ = __invert__ = _arithmetic
    __lt__ = __le__ = __eq__ = __ne__ = __gt__ = __ge__ = _comparison


def _ensure_proxy_owner(owner: _Planner) -> None:
    current = _CURRENT_PLANNER.get()
    if current is not None and current is not owner:
        # Do not read or abort the foreign planner outside its lifecycle lock.
        # This path deliberately does not acquire owner.lock, so concurrent
        # foreign-owner calls cannot invert the two planner locks.
        with current.lock:
            if current.state == "active":
                current.abort(
                    "stale_proxy",
                    "dryml.code.dynamic_trace_stale_proxy",
                    "Dynamic trace proxy belongs to another active trace.",
                )
    # Inspect the owner's lifecycle while holding its lock.  Do not take this
    # lock on the foreign-owner path above: two foreign proxy calls must not
    # invert owner/current lock order.
    with owner.lock:
        if current is owner and owner.state == "active":
            return
        raise DynamicTraceProxyError()


def _static_class_attribute(cls: type, name: str) -> Any:
    for base in type.__getattribute__(cls, "__mro__"):
        namespace = type.__getattribute__(base, "__dict__")
        if name in namespace:
            return namespace[name]
    raise AttributeError(name)


def _resolve_receiver_class(value: Definition | ConcreteDefinition, context: CodeAnalysisContext) -> type:
    candidate = value.cls
    if isinstance(candidate, ImportRef):
        if context.allow_import is not True:
            raise ValueError("imports disabled")
        candidate = _resolve_import_ref_static(candidate)
    elif isinstance(candidate, SourceSpec):
        if context.allow_import is not True or context.allow_source is not True or context.allow_dynamic_execution is not True:
            raise ValueError("source resolution disabled")
        candidate = candidate.resolve()
    if not isinstance(candidate, type):
        raise TypeError("receiver class is not a class")
    return candidate


def _resolve_import_ref_static(reference: ImportRef) -> Any:
    module = importlib.import_module(reference.module)
    if reference.qualname is None:
        return module
    current: Any = module
    for part in reference.qualname.split("."):
        if part == "<locals>":
            raise ValueError("local import reference")
        current = _static_class_attribute(current, part) if isinstance(current, type) else inspect.getattr_static(current, part)
    return current


def _verified_class_import_path(cls: type) -> str | None:
    if type(cls) is not type:
        return None
    module_name = type.__getattribute__(cls, "__module__")
    qualname = type.__getattribute__(cls, "__qualname__")
    if (
        not isinstance(module_name, str)
        or not isinstance(qualname, str)
        or not module_name
        or not qualname
        or module_name == "__main__"
        or "<locals>" in qualname
    ):
        return None
    path = f"{module_name}:{qualname}"
    if len(path) > MAX_REFERENCE_CHARS:
        raise _ReferenceLimitError(
            "receiver_class_chars",
            MAX_REFERENCE_CHARS,
            len(path),
        )
    try:
        current: Any = importlib.import_module(module_name)
        for part in qualname.split("."):
            current = _static_class_attribute(current, part) if isinstance(current, type) else inspect.getattr_static(current, part)
    except Exception:
        return None
    return path if current is cls else None


def _definition_reference(value: Definition | ConcreteDefinition) -> dict[str, str]:
    digest = bounded_stable_hash_function(value, limits=_HASH_LIMITS)
    if not isinstance(digest, str) or not digest or len(digest) > MAX_REFERENCE_CHARS:
        raise ValueError("invalid definition reference")
    if isinstance(value, ConcreteDefinition):
        return {"definition_kind": "concrete_definition", "definition_ref": format_cdef_id(digest)}
    return {"definition_kind": "definition", "definition_ref": digest}


def _wrap_invocation(planner: _Planner, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[tuple[Any, ...], dict[str, Any]]:
    memo: dict[int, Any] = {}
    active: set[int] = set()
    count = [0]

    def wrap(value: Any, depth: int):
        if depth > MAX_VALUE_DEPTH:
            raise _argument_failure(planner, "argument_depth", MAX_VALUE_DEPTH, depth)
        if isinstance(value, (Definition, ConcreteDefinition)):
            return planner.proxy_for(value)
        if type(value) in {list, tuple, dict}:
            oid = id(value)
            if oid in active:
                raise _unsupported_argument_failure(planner)
            if oid in memo:
                return memo[oid]
            count[0] += len(value)
            if count[0] > MAX_CONTAINER_ENTRIES:
                raise _argument_failure(planner, "invocation_entries", MAX_CONTAINER_ENTRIES, count[0])
            active.add(oid)
            try:
                if type(value) is list:
                    result: Any = []
                    memo[oid] = result
                    result.extend(wrap(child, depth + 1) for child in value)
                elif type(value) is dict:
                    if any(type(key) is not str for key in value):
                        raise _unsupported_argument_failure(planner)
                    oversized = next((len(key) for key in value if len(key) > MAX_STRING_CHARS), None)
                    if oversized is not None:
                        raise _argument_failure(planner, "mapping_key_chars", MAX_STRING_CHARS, oversized)
                    result = {}
                    memo[oid] = result
                    for key, child in value.items():
                        result[key] = wrap(child, depth + 1)
                else:
                    # Tuples cannot directly contain themselves; memoization after
                    # children still preserves aliases for all acyclic inputs.
                    result = tuple(wrap(child, depth + 1) for child in value)
                    memo[oid] = result
                return result
            finally:
                active.remove(oid)
        if not planner.request.policy.require_proxy_only_args:
            return _validated_scalar(value, planner, preexecution=True)
        raise _unsupported_argument_failure(planner)

    return wrap(args, 0), wrap(kwargs, 0)


def _encode_observed_call(planner: _Planner, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[list[Any], dict[str, Any]]:
    active: set[int] = set()
    count = [0]

    def encode(value: Any, depth: int):
        if depth > MAX_VALUE_DEPTH:
            planner.abort(
                "unsupported_argument",
                "dryml.code.dynamic_trace_argument_limit_exceeded",
                "Dynamic trace observed argument depth limit exceeded.",
                data={"limit_name": "observed_depth", "limit": MAX_VALUE_DEPTH, "observed_lower_bound": depth},
            )
        if isinstance(value, _DefinitionProxy):
            _ensure_proxy_owner(value._planner)
            return dict(value._reference)
        if isinstance(value, (Definition, ConcreteDefinition)):
            try:
                return _definition_reference(value)
            except StableHashLimitError as exc:
                planner.abort(
                    "unsupported_argument",
                    "dryml.code.dynamic_trace_argument_limit_exceeded",
                    "Dynamic trace observed definition identity limit exceeded.",
                    data={"limit_name": f"hash_{exc.limit_name}", "limit": exc.limit, "observed_lower_bound": exc.observed_lower_bound},
                )
        if type(value) in {list, tuple, dict}:
            oid = id(value)
            if oid in active:
                planner.abort(
                    "unsupported_argument",
                    "dryml.code.dynamic_trace_unsupported_argument",
                    "Dynamic trace observed an unsupported method argument.",
                )
            count[0] += len(value)
            if count[0] > MAX_CONTAINER_ENTRIES:
                planner.abort(
                    "unsupported_argument",
                    "dryml.code.dynamic_trace_argument_limit_exceeded",
                    "Dynamic trace observed argument entry limit exceeded.",
                    data={"limit_name": "observed_entries", "limit": MAX_CONTAINER_ENTRIES, "observed_lower_bound": count[0]},
                )
            active.add(oid)
            try:
                if type(value) in {list, tuple}:
                    return [encode(child, depth + 1) for child in value]
                if any(type(key) is not str for key in value):
                    planner.abort(
                        "unsupported_argument",
                        "dryml.code.dynamic_trace_unsupported_argument",
                        "Dynamic trace observed an unsupported method argument.",
                    )
                oversized = next((len(key) for key in value if len(key) > MAX_STRING_CHARS), None)
                if oversized is not None:
                    planner.abort(
                        "unsupported_argument",
                        "dryml.code.dynamic_trace_argument_limit_exceeded",
                        "Dynamic trace observed mapping key limit exceeded.",
                        data={"limit_name": "mapping_key_chars", "limit": MAX_STRING_CHARS, "observed_lower_bound": oversized},
                    )
                return {key: encode(child, depth + 1) for key, child in value.items()}
            finally:
                active.remove(oid)
        try:
            return _validated_scalar(value, planner, preexecution=False)
        except _PrevalidationFailure:
            planner.abort(
                "unsupported_argument",
                "dryml.code.dynamic_trace_unsupported_argument",
                "Dynamic trace observed an unsupported method argument.",
            )
        raise AssertionError("unreachable")

    return encode(args, 0), encode(kwargs, 0)


def _validated_scalar(value: Any, planner: _Planner, *, preexecution: bool) -> Any:
    if value is None or type(value) is bool:
        return value
    if type(value) is int:
        bits = value.bit_length()
        if bits > MAX_INTEGER_BITS:
            if preexecution:
                raise _argument_failure(planner, "integer_bits", MAX_INTEGER_BITS, bits)
            planner.abort(
                "unsupported_argument", "dryml.code.dynamic_trace_argument_limit_exceeded",
                "Dynamic trace observed integer limit exceeded.",
                data={"limit_name": "integer_bits", "limit": MAX_INTEGER_BITS, "observed_lower_bound": bits},
            )
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if type(value) is str:
        if len(value) > MAX_STRING_CHARS:
            if preexecution:
                raise _argument_failure(planner, "string_chars", MAX_STRING_CHARS, len(value))
            planner.abort(
                "unsupported_argument", "dryml.code.dynamic_trace_argument_limit_exceeded",
                "Dynamic trace observed string limit exceeded.",
                data={"limit_name": "string_chars", "limit": MAX_STRING_CHARS, "observed_lower_bound": len(value)},
            )
        return value
    if preexecution:
        raise _unsupported_argument_failure(planner)
    raise _PrevalidationFailure(_diagnostic(
        "dryml.code.dynamic_trace_unsupported_argument",
        "Dynamic trace observed an unsupported method argument.",
        target_kind=planner.target_kind,
    ))


def _argument_failure(planner: _Planner, name: str, limit: int, observed: int) -> _PrevalidationFailure:
    return _PrevalidationFailure(_diagnostic(
        "dryml.code.dynamic_trace_argument_limit_exceeded",
        "Dynamic trace invocation argument limit exceeded.",
        target_kind=planner.target_kind,
        data={"limit_name": name, "limit": limit, "observed_lower_bound": observed},
    ))


def _unsupported_argument_failure(planner: _Planner) -> _PrevalidationFailure:
    return _PrevalidationFailure(_diagnostic(
        "dryml.code.dynamic_trace_unsupported_argument",
        "Dynamic trace invocation contains an unsupported argument.",
        target_kind=planner.target_kind,
    ))


def _summary(planner: _Planner) -> CodeFact:
    def make_summary() -> CodeFact:
        return CodeFact(
            "dynamic_trace_summary",
            source={"analyzer": "dynamic_trace", "target_kind": planner.target_kind},
            data={
                "complete": planner.outcome == "complete",
                "outcome": planner.outcome,
                "calls_recorded": len(planner.facts),
                "max_calls": planner.request.policy.max_calls,
            },
        )
    summary = make_summary()
    size = len(json.dumps(summary.to_data(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"))
    if planner.result_bytes + size > MAX_RESULT_BYTES and planner.outcome != "result_limit_exceeded":
        planner.add_diagnostic(_diagnostic(
            "dryml.code.dynamic_trace_result_limit_exceeded",
            "Dynamic trace result byte limit exceeded.",
            target_kind=planner.target_kind,
            data={"limit_name": "result_bytes", "limit": MAX_RESULT_BYTES, "observed_lower_bound": planner.result_bytes + size},
        ))
        planner.outcome = "result_limit_exceeded"
        summary = make_summary()
        size = len(json.dumps(summary.to_data(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"))
        while planner.facts and planner.result_bytes + size > MAX_RESULT_BYTES:
            removed = planner.facts.pop()
            planner.result_bytes -= len(json.dumps(removed.to_data(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"))
            summary = make_summary()
            size = len(json.dumps(summary.to_data(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"))
    return summary


def _encoded_summary_size(*, target_kind: str, outcome: str, calls_recorded: int, max_calls: int) -> int:
    fact = CodeFact(
        "dynamic_trace_summary",
        source={"analyzer": "dynamic_trace", "target_kind": target_kind},
        data={
            "complete": outcome == "complete",
            "outcome": outcome,
            "calls_recorded": calls_recorded,
            "max_calls": max_calls,
        },
    )
    return len(json.dumps(fact.to_data(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"))


def run_trace(request: _InvocationRequest) -> CodeAnalysisResult:
    """Run one already validated current-process dynamic trace request."""

    planner = _Planner(request)
    try:
        wrapped_args, wrapped_kwargs = _wrap_invocation(planner, request.args, request.kwargs)
    except _PrevalidationFailure as exc:
        return CodeAnalysisResult(target=request.target.spec, diagnostics=(exc.diagnostic,))

    token = _CURRENT_PLANNER.set(planner)
    interruption: BaseException | None = None
    try:
        try:
            request.target.obj(*wrapped_args, **wrapped_kwargs)
        except _TraceAbort:
            pass
        except Exception as exc:
            if planner.state == "active":
                exc_type = type(exc)
                module = type.__getattribute__(exc_type, "__module__")
                qualname = type.__getattribute__(exc_type, "__qualname__")
                identity = f"{module}:{qualname}" if isinstance(module, str) and isinstance(qualname, str) else None
                planner.add_diagnostic(_diagnostic(
                    "dryml.code.dynamic_trace_target_failed",
                    "Dynamic trace target raised an exception.",
                    target_kind=planner.target_kind,
                    data={"exception_type": identity[:MAX_STRING_CHARS] if identity else None},
                ))
                planner.state = "aborted"
                planner.outcome = "target_failed"
        except BaseException as exc:
            interruption = exc
    finally:
        planner.close()
        _CURRENT_PLANNER.reset(token)

    if interruption is not None:
        raise interruption
    if planner.unexpected is not None and request.context.diagnostics_policy == "raise":
        raise CodeAnalysisError("Dynamic trace implementation failed.") from planner.unexpected
    return CodeAnalysisResult(
        target=request.target.spec,
        facts=tuple(planner.facts) + (_summary(planner),),
        diagnostics=tuple(planner.diagnostics),
    )


def analyzer_result(target: CodeTarget, context: CodeAnalysisContext) -> CodeAnalysisResult:
    """Reject invocation through the ordinary non-invoking analyzer protocol."""

    return CodeAnalysisResult(
        target=target.spec,
        diagnostics=(_diagnostic(
            "dryml.code.dynamic_trace_requires_trace_facade",
            "Dynamic tracing requires the explicit trace facade.",
            target_kind=target.spec.kind,
        ),),
    )


ANALYZER = FunctionAnalyzer("dynamic_trace", analyzer_result)


__all__ = [
    "ANALYZER",
    "DynamicTracePolicy",
    "DynamicTraceProxyError",
    "MAX_CALLS",
    "run_trace",
]
