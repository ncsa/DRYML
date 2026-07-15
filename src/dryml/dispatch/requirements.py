"""Requirement-aware dispatch planning over normalized operation targets.

This module orchestrates existing code analysis, annotation, environment, world,
and runtime APIs. It does not normalize user targets, merge annotations, or
solve environments; it delegates local world synthesis to :mod:`dryml.worlds`.
"""

from __future__ import annotations

import json
import hashlib
import math
import os
import re
import sys
import types
import uuid
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any

from dryml import annotations, environments, runtime, worlds
from dryml.code import DynamicCallFact, DynamicTracePolicy, analyze, probe_target, target_from_callable, trace
from dryml.code.analysis import CodeAnalysisContext, CodeAnalysisResult
from dryml.code.facts import AnnotationFact, CodeFact, DiagnosticFact
from dryml.code.probe import CodeProbeResult
from dryml.code.targets import CodeTargetSpec
from dryml.core2.definition import ConcreteDefinition
from dryml.core2.utils.general import pickle_load
from dryml.environments.records import EnvironmentRecord
from dryml.environments.specs import EnvironmentSpec, PythonExecutableSpec, spec_from_data
from dryml.runtime import RuntimeEnforcement, RuntimeMode
from dryml.runtime.specs import RuntimeContextSpec
from dryml.worlds.specs import WorldSpec
from dryml.operations import resolve_call_arguments

from .errors import DispatchPlanningError
from .normalize import NormalizedDispatchTarget


PLANNING_METADATA_VERSION = 2
DEFAULT_PROBE_TIMEOUT_S = 30.0
_TRACE_SCHEMA = "dryml.dispatch.dynamic_trace.v1"
_TRACE_MAX_CALLS = 256
_TRACE_MAX_FRAGMENTS = 1024
_TRACE_MAX_DUPLICATES = 1024
_TRACE_MAX_DIAGNOSTICS = 256
_TRACE_MAX_STRING = 4096
_TRACE_MAX_DEPTH = 32
_TRACE_MAX_BYTES = 1_048_576
_TRACE_CORRELATION_INPUT_KEY = "_dryml_dispatch_trace_input_id"
_TRACE_CORRELATION_RUN_KEY = "_dryml_dispatch_trace_run_id"
_TRACE_PREEXEC_CODES = frozenset({
    "dryml.code.dynamic_trace_disabled",
    "dryml.code.dynamic_trace_invalid_context",
    "dryml.code.dynamic_trace_unsupported_target",
    "dryml.code.dynamic_trace_unsupported_argument",
    "dryml.code.dynamic_trace_argument_limit_exceeded",
    "dryml.code.dynamic_trace_receiver_resolution_failed",
})
_TRACE_SUMMARY_OUTCOMES = frozenset({
    "complete",
    "call_limit_exceeded",
    "unsupported_return_operation",
    "unsupported_argument",
    "unsupported_receiver_attribute",
    "stale_proxy",
    "method_fact_collection_failed",
    "target_failed",
    "result_limit_exceeded",
    "diagnostics_limit_exceeded",
    "algorithm_failed",
})
_MAX_METADATA_DEPTH = 8
_MAX_METADATA_ITEMS = 64
_MAX_METADATA_STRING = 4096
_MAX_METADATA_NODES = 8192
_RESERVED_PLANNING_KEYS = frozenset(
    {
        "dryml.dispatch.planning_version",
        "dryml.code_analysis",
        "dryml.code_probe",
        "dryml.requirements",
        "dryml.requirement_sources",
        "dryml.environment_selection",
        "dryml.environment_probe",
        "dryml.environment_check",
        "dryml.environment_resolution",
        "dryml.world_selection",
        "dryml.world_check",
        "dryml.world_synthesis",
        "dryml.local_inventory",
        "dryml.world_allocation",
        "dryml.runtime_selection",
        "dryml.runtime_check",
        "dryml.requirement_policy",
        "dryml.runtime_enforcement",
        "dryml.dispatch.launchable",
        "dryml.dispatch.diagnostics",
        "dryml.dispatch.dynamic_trace",
    }
)


class RequirementPolicy(str, Enum):
    """How dispatch handles discovered hard requirement incompatibilities."""

    STRICT = "strict"
    WARN = "warn"
    IGNORE = "ignore"


class _TraceProvenanceLimitError(ValueError):
    """A provenance construction failure caused specifically by a hard bound."""


@dataclass(frozen=True, slots=True)
class _DynamicTraceRequest:
    """Validated opt-in analysis-policy extension for one dispatch request.

    ``analysis_policy`` remains backwards compatible with a direct
    :class:`CodeAnalysisContext`.  Only this mapping-derived request can enable
    invocation; permission on the caller context alone is never an opt-in.
    """

    context: CodeAnalysisContext
    probe_timeout_s: float
    policy: DynamicTracePolicy | None = None

    @property
    def requested(self) -> bool:
        """Return whether this request explicitly selected dynamic tracing."""

        return self.policy is not None


@dataclass(frozen=True, slots=True)
class DynamicTraceProvenance:
    """Bounded, serializable per-request dynamic-trace planning evidence.

    The carrier admits only the current normalized transport tokens and exact
    9B policy/summary schemas.  Rejected evidence is diagnostic-only: a valid
    summary/call prefix may prove start but never enters requirement resolution;
    complete carriers bind full ``DynamicCallFact`` wires to the carrier target
    and observations to canonical serialized annotation facts; overflow retains
    a valid summary with empty calls. Evidence that cannot be retained unchanged
    without persisting call arguments, unrecognized annotation metadata, local
    source paths, or environment overrides is rejected before resolution. The
    carrier stores fixed code/severity diagnostics rather than target exceptions,
    source text, environment values, streams, live objects, or arbitrary
    representations.
    """

    data: Mapping[str, Any]

    def __post_init__(self) -> None:
        if type(self.data) is not dict:
            raise ValueError("dynamic trace provenance must be an exact dictionary")
        value = dict(self.data)
        required = {
            "schema", "schema_version", "requested", "trace_input_id",
            "trace_run_id", "execution_location", "execution_started",
            "target", "policy", "status", "summary", "calls",
            "accepted_fragments", "duplicate_observations", "diagnostics",
        }
        if set(value) != required or value.get("schema") != _TRACE_SCHEMA or value.get("schema_version") != 1:
            raise ValueError("invalid dynamic trace provenance schema")
        if value.get("requested") is not True or value.get("execution_location") != "current_process":
            raise ValueError("invalid dynamic trace provenance request/location")
        status = value.get("status")
        if status not in {"pre_execution_failed", "complete", "failed", "incomplete", "provenance_limit_exceeded", "evidence_rejected"}:
            raise ValueError("invalid dynamic trace provenance status")
        if value.get("execution_started") not in {True, False, None}:
            raise ValueError("invalid dynamic trace execution state")
        input_id, run_id = value.get("trace_input_id"), value.get("trace_run_id")
        if input_id is not None and (type(input_id) is not str or not input_id):
            raise ValueError("invalid trace input ID")
        if run_id is not None and (type(run_id) is not str or not run_id):
            raise ValueError("invalid trace run ID")
        if status == "pre_execution_failed":
            if run_id is not None or value["execution_started"] is not False:
                raise ValueError("pre-execution trace provenance cannot have a run")
        elif not isinstance(input_id, str) or not input_id or not isinstance(run_id, str) or not run_id:
            raise ValueError("post-start/rejected trace provenance requires input and run IDs")
        if not isinstance(value.get("calls"), list) or len(value["calls"]) > _TRACE_MAX_CALLS:
            if isinstance(value.get("calls"), list):
                raise _TraceProvenanceLimitError("dynamic trace provenance call limit exceeded")
            raise ValueError("invalid dynamic trace call evidence")
        for name, limit in (("accepted_fragments", _TRACE_MAX_FRAGMENTS), ("duplicate_observations", _TRACE_MAX_DUPLICATES), ("diagnostics", _TRACE_MAX_DIAGNOSTICS)):
            if not isinstance(value.get(name), list) or len(value[name]) > limit:
                if isinstance(value.get(name), list):
                    raise _TraceProvenanceLimitError("dynamic trace provenance limit exceeded")
                raise ValueError("invalid dynamic trace provenance collection")
        target = value.get("target")
        if target is not None and (
            type(target) is not dict
            or set(target) != {"target_kind", "transport"}
            or type(target["target_kind"]) is not str
            or not target["target_kind"]
            or target["transport"] not in {"import_path", "pickle_small", "operation_spec", "method_call"}
        ):
            raise ValueError("invalid dynamic trace target")
        if status != "pre_execution_failed" and target is None:
            raise ValueError("post-start/rejected trace provenance requires a target")
        _validate_trace_provenance_value(value)
        if len(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")) > _TRACE_MAX_BYTES:
            raise _TraceProvenanceLimitError("dynamic trace provenance exceeds byte limit")
        _validate_trace_policy(value["policy"])
        summary = value["summary"]
        calls = _validated_trace_calls(value["calls"], target=target)
        if status == "complete":
            _validate_trace_observations(calls, value["accepted_fragments"], value["duplicate_observations"])
        _validate_trace_diagnostics(value["diagnostics"])
        if status == "pre_execution_failed":
            if summary is not None or calls or value["accepted_fragments"] or value["duplicate_observations"] or not value["diagnostics"]:
                raise ValueError("invalid pre-execution trace provenance evidence")
        elif status == "evidence_rejected":
            if value["execution_started"] not in {True, None} or value["accepted_fragments"] or value["duplicate_observations"] or not value["diagnostics"]:
                raise ValueError("invalid rejected trace provenance evidence")
            if value["execution_started"] is None and (calls or summary is not None):
                raise ValueError("rejected trace provenance evidence proves execution start")
            if [call.data["sequence"] for call in calls] != list(range(len(calls))):
                raise ValueError("rejected trace provenance calls must be contiguous")
            if summary is not None:
                _validate_trace_summary(summary, policy=value["policy"], target=target)
                if summary["data"]["calls_recorded"] != len(calls):
                    raise ValueError("rejected trace provenance summary call count differs")
        elif status == "provenance_limit_exceeded":
            if value["execution_started"] is not True:
                raise ValueError("trace provenance status requires execution start")
            _validate_trace_summary(summary, policy=value["policy"], target=target)
            if calls or value["accepted_fragments"] or value["duplicate_observations"] or not value["diagnostics"]:
                raise ValueError("provenance-limit trace cannot retain a truncated prefix")
        else:
            if value["execution_started"] is not True:
                raise ValueError("trace provenance status requires execution start")
            _validate_trace_summary(summary, policy=value["policy"], target=target)
            if [call.data["sequence"] for call in calls] != list(range(len(calls))):
                raise ValueError("trace provenance calls must be contiguous")
            if summary["data"]["calls_recorded"] != len(calls):
                raise ValueError("trace provenance summary call count differs")
            if status == "complete":
                if summary["data"]["complete"] is not True or value["diagnostics"]:
                    raise ValueError("complete trace provenance requires complete summary and no diagnostics")
            elif status in {"failed", "incomplete"}:
                if summary["data"]["complete"] is not False or value["accepted_fragments"] or value["duplicate_observations"] or not value["diagnostics"]:
                    raise ValueError("incomplete trace provenance has invalid evidence")
                if (status == "failed") != (summary["data"]["outcome"] == "target_failed"):
                    raise ValueError("invalid failed/incomplete trace provenance outcome")
        object.__setattr__(self, "data", value)

    def to_data(self) -> dict[str, Any]:
        """Return an isolated JSON-ready provenance copy."""

        return json.loads(json.dumps(self.data, sort_keys=True, separators=(",", ":"), allow_nan=False))

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "DynamicTraceProvenance":
        """Restore strict versioned provenance without accepting extra fields."""

        return cls(data)


def _validate_trace_provenance_value(value: Any, *, depth: int = 0) -> None:
    """Reject, rather than coerce or truncate, the provenance JSON grammar."""

    if depth > _TRACE_MAX_DEPTH:
        raise _TraceProvenanceLimitError("dynamic trace provenance exceeds nesting limit")
    if value is None or type(value) is bool:
        return
    if type(value) is int:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("dynamic trace provenance floats must be finite")
        return
    if type(value) is str:
        if len(value) > _TRACE_MAX_STRING:
            raise _TraceProvenanceLimitError("dynamic trace provenance string exceeds limit")
        return
    if type(value) is list:
        for item in value:
            _validate_trace_provenance_value(item, depth=depth + 1)
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str or len(key) > _TRACE_MAX_STRING:
                raise _TraceProvenanceLimitError("dynamic trace provenance keys must be bounded strings")
            _validate_trace_provenance_value(item, depth=depth + 1)
        return
    raise ValueError("dynamic trace provenance must use JSON values")


def _validate_trace_policy(value: Any) -> None:
    if type(value) is not dict or set(value) != {"max_calls", "require_proxy_only_args", "collect_requirements"}:
        raise ValueError("invalid dynamic trace provenance policy")
    if type(value["max_calls"]) is not int or not 1 <= value["max_calls"] <= 10_000:
        raise ValueError("invalid dynamic trace provenance max_calls")
    if type(value["require_proxy_only_args"]) is not bool or type(value["collect_requirements"]) is not bool:
        raise ValueError("invalid dynamic trace provenance policy booleans")


def _validate_trace_summary(value: Any, *, policy: Mapping[str, Any], target: Mapping[str, Any] | None) -> CodeFact:
    if type(value) is not dict or set(value) != {"kind", "source", "data"}:
        raise ValueError("invalid dynamic trace summary wire")
    fact = CodeFact.from_data(value)
    if fact.kind != "dynamic_trace_summary" or type(fact.source) is not dict or set(fact.source) != {"analyzer", "target_kind"}:
        raise ValueError("invalid dynamic trace summary")
    data = fact.data
    if type(data) is not dict or set(data) != {"complete", "outcome", "calls_recorded", "max_calls"}:
        raise ValueError("invalid dynamic trace summary data")
    if type(data["complete"]) is not bool or type(data["outcome"]) is not str:
        raise ValueError("invalid dynamic trace summary outcome")
    if type(data["calls_recorded"]) is not int or not 0 <= data["calls_recorded"] <= policy["max_calls"]:
        raise ValueError("invalid dynamic trace summary call count")
    if data["max_calls"] != policy["max_calls"]:
        raise ValueError("dynamic trace summary policy mismatch")
    if fact.source["analyzer"] != "dynamic_trace" or (target is not None and fact.source["target_kind"] != target["target_kind"]):
        raise ValueError("dynamic trace summary target mismatch")
    if data["outcome"] not in _TRACE_SUMMARY_OUTCOMES or (data["outcome"] == "complete") != data["complete"]:
        raise ValueError("dynamic trace summary completion mismatch")
    return fact


def _validate_trace_observations(
    calls: list[DynamicCallFact],
    accepted: list[Any],
    duplicates: list[Any],
) -> None:
    """Require one canonical observation for every serialized annotation fact."""

    expected: dict[tuple[int, str, str], int] = {}
    call_fragments: dict[int, set[str]] = {}
    for call in calls:
        sequence = call.data["sequence"]
        method = call.data["method_name"]
        keys = call_fragments.setdefault(sequence, set())
        for fact_data in call.data["method_facts"]:
            fact = CodeFact.from_data(fact_data)
            if not isinstance(fact, AnnotationFact):
                continue
            fragment = annotations.AnnotationFragment.from_data(fact.data)
            fragment_data = json.dumps(fragment.to_data(), sort_keys=True, separators=(",", ":"))
            key = hashlib.sha256(fragment_data.encode()).hexdigest()
            expected[(sequence, method, key)] = expected.get((sequence, method, key), 0) + 1
            keys.add(key)

    observed: dict[tuple[int, str, str], int] = {}
    for item in accepted:
        if type(item) is not dict or set(item) != {"sequence", "method", "fragment_key"}:
            raise ValueError("invalid accepted trace fragment observation")
        if (
            type(item["sequence"]) is not int
            or item["sequence"] < 0
            or type(item["method"]) is not str
            or re.fullmatch(r"[0-9a-f]{64}", item["fragment_key"]) is None
        ):
            raise ValueError("invalid accepted trace fragment observation value")
        key = (item["sequence"], item["method"], item["fragment_key"])
        observed[key] = observed.get(key, 0) + 1
    for item in duplicates:
        if type(item) is not dict or set(item) != {"sequence", "method", "fragment_key", "first"}:
            raise ValueError("invalid duplicate trace fragment observation")
        if (
            type(item["sequence"]) is not int
            or item["sequence"] < 0
            or type(item["method"]) is not str
            or re.fullmatch(r"[0-9a-f]{64}", item["fragment_key"]) is None
            or type(item["first"]) is not str
        ):
            raise ValueError("invalid duplicate trace fragment observation value")
        key = (item["sequence"], item["method"], item["fragment_key"])
        if item["first"] != "direct":
            match = re.fullmatch(r"trace:(\d+)", item["first"])
            if (
                match is None
                or int(match.group(1)) > item["sequence"]
                or item["fragment_key"] not in call_fragments.get(int(match.group(1)), set())
            ):
                raise ValueError("invalid duplicate trace fragment first observation")
        observed[key] = observed.get(key, 0) + 1
    if observed != expected:
        raise ValueError("trace observations do not match annotation method facts")


def _validate_trace_diagnostics(values: list[Any]) -> None:
    for value in values:
        if type(value) is not dict or set(value) != {"code", "severity", "data"}:
            raise ValueError("invalid dynamic trace diagnostic")
        if type(value["code"]) is not str or not value["code"] or type(value["severity"]) is not str or value["severity"] not in {"info", "warning", "error"}:
            raise ValueError("invalid dynamic trace diagnostic value")
        data = value["data"]
        if type(data) is not dict or set(data) != {"trace_diagnostic_codes"} or type(data["trace_diagnostic_codes"]) is not list:
            raise ValueError("invalid dynamic trace diagnostic data")
        if len(data["trace_diagnostic_codes"]) > _TRACE_MAX_DIAGNOSTICS:
            raise _TraceProvenanceLimitError("dynamic trace diagnostic code limit exceeded")
        if any(type(code) is not str or not code for code in data["trace_diagnostic_codes"]):
            raise ValueError("invalid dynamic trace diagnostic code")


@dataclass(frozen=True, slots=True)
class CandidateConsideration:
    """One deterministic candidate-precedence slot considered by planning.

    Attributes:
        slot: Precedence slot name, such as ``explicit`` or ``current``.
        status: Whether the slot was absent, considered, or selected.
        candidate: Bounded canonical candidate data when the slot had a value.
    """

    slot: str
    status: str
    candidate: Mapping[str, Any] | None = None

    def to_data(self) -> dict[str, Any]:
        """Return bounded JSON-ready consideration data."""

        return {"slot": self.slot, "status": self.status, "candidate": None if self.candidate is None else _safe_candidate_data(self.candidate)}


@dataclass(frozen=True, slots=True)
class CandidateSelection:
    """A selected environment, world, or runtime and its precedence trace.

    Attributes:
        kind: Candidate subsystem: environment, world, or runtime.
        candidate: Canonical selected candidate data.
        source: Winning precedence source, including resolver or synthesized.
        considered: Ordered trace of higher-precedence and selected slots.
        diagnostics: Selection diagnostics retained for explanation.
    """

    kind: str
    candidate: Mapping[str, Any]
    source: str
    considered: tuple[CandidateConsideration, ...]
    diagnostics: tuple[DiagnosticFact, ...] = ()

    def to_data(self) -> dict[str, Any]:
        """Return JSON-ready candidate selection data."""

        return {
            "kind": self.kind,
            "candidate": _safe_candidate_data(self.candidate),
            "source": self.source,
            "considered": [item.to_data() for item in self.considered],
            "diagnostics": [item.to_data() for item in self.diagnostics],
        }


@dataclass(frozen=True, slots=True)
class CandidateCheckReport:
    """Normalized result of checking one selected candidate.

    Attributes:
        kind: Candidate subsystem that was checked.
        status: Normalized check outcome.
        compatible: Compatibility decision, or ``None`` when not evaluated.
        requirement: Canonical hard requirement data.
        candidate: Canonical checked candidate data.
        details: Structured compatibility findings.
        diagnostics: Probe or validation diagnostics supporting the result.
    """

    kind: str
    status: str
    compatible: bool | None
    requirement: Mapping[str, Any] | None
    candidate: Mapping[str, Any] | None
    details: tuple[Mapping[str, Any], ...] = ()
    diagnostics: tuple[DiagnosticFact, ...] = ()

    def to_data(self) -> dict[str, Any]:
        """Return JSON-ready compatibility report data."""

        return {
            "kind": self.kind,
            "status": self.status,
            "compatible": self.compatible,
            "requirement": None if self.requirement is None else dict(self.requirement),
            "candidate": None if self.candidate is None else _safe_candidate_data(self.candidate),
            "details": [dict(item) for item in self.details],
            "diagnostics": [item.to_data() for item in self.diagnostics],
        }


@dataclass(frozen=True, slots=True)
class DispatchPlanningResolution:
    """Complete serializable planning decisions for a normalized dispatch target.

    Attributes:
        normalized_target: Bounded normalized operation identity and transport.
        code_analysis: Static facts collected for the target, when available.
        code_probe: Final code-probe evidence, when required.
        bootstrap_environment: Environment used for initial target discovery.
        bootstrap_code_probe: Initial target-discovery probe evidence.
        final_code_probe: Selected-environment target validation evidence.
        requirements: Merged annotation requirements and defaults.
        environment_selection: Selected environment and precedence trace.
        environment_record: Reusable selected environment inventory record.
        environment_check: Final environment requirement result.
        world_selection: Selected world and precedence trace.
        world_check: Final requested-world requirement result.
        runtime_selection: Selected worker runtime and precedence trace.
        runtime_check: Final runtime requirement result.
        requirement_policy: Effective strict, warn, or ignore policy.
        runtime_enforcement: Effective runtime enforcement mode.
        diagnostics: Bounded planning diagnostics.
        launchable: Whether the equivalent plan can be launched safely.
        environment_resolution: Resolver report when environment search ran.
        world_synthesis: Synthesis report when a local world was synthesized.
        inventory_summary: Bounded inventory used by synthesis or allocation.
        world_allocation_summary: Actual local allocation summary when planned.
        local_inventory: Internal reused inventory object; omitted from public data.
        dynamic_trace: Opt-in current-process trace evidence, never operation metadata.
    """

    normalized_target: Mapping[str, Any]
    code_analysis: CodeAnalysisResult | None
    code_probe: CodeProbeResult | None
    bootstrap_environment: Mapping[str, Any] | None
    bootstrap_code_probe: CodeProbeResult | None
    final_code_probe: CodeProbeResult | None
    requirements: annotations.RequirementResolution
    environment_selection: CandidateSelection
    environment_record: EnvironmentRecord | None
    environment_check: CandidateCheckReport
    world_selection: CandidateSelection
    world_check: CandidateCheckReport
    runtime_selection: CandidateSelection
    runtime_check: CandidateCheckReport
    requirement_policy: RequirementPolicy
    runtime_enforcement: RuntimeEnforcement
    diagnostics: tuple[DiagnosticFact, ...]
    launchable: bool
    environment_resolution: environments.EnvironmentResolution | None = None
    world_synthesis: worlds.WorldSynthesisResult | None = None
    inventory_summary: Mapping[str, Any] | None = None
    world_allocation_summary: Mapping[str, Any] | None = None
    local_inventory: worlds.LocalResourceInventory | None = None
    dynamic_trace: DynamicTraceProvenance | None = None

    def to_data(self) -> dict[str, Any]:
        """Return bounded JSON-ready planning decisions without live targets."""

        data = {
            "normalized_target": dict(self.normalized_target),
            "code_analysis": _analysis_summary(self.code_analysis),
            "code_probe": _probe_summary(self.code_probe),
            "bootstrap_environment": None if self.bootstrap_environment is None else dict(self.bootstrap_environment),
            "bootstrap_code_probe": _probe_summary(self.bootstrap_code_probe),
            "final_code_probe": _probe_summary(self.final_code_probe),
            "requirements": self.requirements.to_data(),
            "environment_selection": self.environment_selection.to_data(),
            "environment_record": _environment_record_summary(self.environment_record),
            "environment_check": self.environment_check.to_data(),
            "environment_resolution": None if self.environment_resolution is None else self.environment_resolution.to_data(),
            "world_selection": self.world_selection.to_data(),
            "world_check": self.world_check.to_data(),
            "world_synthesis": None if self.world_synthesis is None else self.world_synthesis.to_data(),
            "inventory_summary": None if self.inventory_summary is None else dict(self.inventory_summary),
            "world_allocation_summary": None if self.world_allocation_summary is None else dict(self.world_allocation_summary),
            "runtime_selection": self.runtime_selection.to_data(),
            "runtime_check": self.runtime_check.to_data(),
            "requirement_policy": self.requirement_policy.value,
            "runtime_enforcement": self.runtime_enforcement.value,
            "diagnostics": [item.to_data() for item in self.diagnostics],
            "launchable": self.launchable,
            "dynamic_trace": None if self.dynamic_trace is None else self.dynamic_trace.to_data(),
        }
        # DynamicTraceProvenance already rejects every schema, depth, scalar,
        # count, and byte overflow.  Passing it through the generic metadata
        # truncator would silently change valid 65th+ observations.
        bounded = _bounded_data({key: value for key, value in data.items() if key != "dynamic_trace"})
        bounded["dynamic_trace"] = data["dynamic_trace"]
        return bounded

    def metadata(self) -> dict[str, Any]:
        """Return authoritative dispatch metadata for this resolution."""

        data = self.to_data()
        return {
            "dryml.dispatch.planning_version": PLANNING_METADATA_VERSION,
            "dryml.code_analysis": data["code_analysis"],
            "dryml.code_probe": {
                "bootstrap_environment": data["bootstrap_environment"],
                "bootstrap_probe": data["bootstrap_code_probe"],
                "final_probe": data["final_code_probe"],
            },
            "dryml.requirements": data["requirements"],
            "dryml.requirement_sources": data["requirements"].get("source_traces", []),
            "dryml.environment_selection": data["environment_selection"],
            "dryml.environment_probe": _environment_record_summary(self.environment_record),
            "dryml.environment_check": data["environment_check"],
            "dryml.environment_resolution": data["environment_resolution"],
            "dryml.world_selection": data["world_selection"],
            "dryml.world_check": data["world_check"],
            "dryml.world_synthesis": data["world_synthesis"],
            "dryml.local_inventory": data["inventory_summary"],
            "dryml.world_allocation": data["world_allocation_summary"],
            "dryml.runtime_selection": data["runtime_selection"],
            "dryml.runtime_check": data["runtime_check"],
            "dryml.requirement_policy": self.requirement_policy.value,
            "dryml.runtime_enforcement": self.runtime_enforcement.value,
            "dryml.dispatch.launchable": self.launchable,
            "dryml.dispatch.diagnostics": data["diagnostics"],
            **({"dryml.dispatch.dynamic_trace": data["dynamic_trace"]} if data["dynamic_trace"] is not None else {}),
        }


@dataclass(frozen=True, slots=True)
class DispatchExplanation:
    """Non-launching view of normalized dispatch planning decisions.

    Attributes:
        resolution: Candidate selections, compatibility checks, metadata, and
            launchability produced by the same pipeline as planning.
        operation_preview: Bounded normalized operation summary.
        blocking_diagnostics: Ordered diagnostics that prevent launch.
    """

    resolution: DispatchPlanningResolution
    operation_preview: Mapping[str, Any]
    blocking_diagnostics: tuple[DiagnosticFact, ...]

    @property
    def launchable(self) -> bool:
        """Return whether the equivalent dispatch plan may launch."""

        return self.resolution.launchable

    @property
    def requirements(self):
        """Return the resolved annotation requirements/defaults."""

        return self.resolution.requirements

    def to_data(self) -> dict[str, Any]:
        """Return JSON-ready explanation data."""

        resolution = self.resolution.to_data()
        dynamic_trace = resolution.pop("dynamic_trace")
        data = _bounded_data({
            "resolution": resolution,
            "operation_preview": dict(self.operation_preview),
            "blocking_diagnostics": [item.to_data() for item in self.blocking_diagnostics],
            "launchable": self.launchable,
        })
        # DynamicTraceProvenance already validated its independent schema and
        # bounds. Do not silently truncate it through the generic explanation
        # metadata projection.
        data["resolution"]["dynamic_trace"] = dynamic_trace
        return data

    def __str__(self) -> str:
        """Format a concise human-readable planning summary."""

        environment_resolution = self.resolution.environment_resolution
        environment_name = None if environment_resolution is None else environment_resolution.selected_name
        attempts = 0 if environment_resolution is None else environment_resolution.attempt_count
        probes = 0 if environment_resolution is None else environment_resolution.probe_count
        inventory = self.resolution.inventory_summary
        inventory_text = ""
        if inventory is not None:
            inventory_text = f" inventory_cpus={inventory.get('cpu_count', 0)} inventory_accelerators={sorted((inventory.get('accelerator_counts') or {}).keys())}"
        blocking_text = ""
        if self.blocking_diagnostics:
            first = self.blocking_diagnostics[0]
            action = first.data.get("action") if isinstance(first.data, Mapping) else None
            blocking_text = f" blocking_action={action or first.message}"
        return (
            f"dispatch target={self.operation_preview.get('kind')} policy={self.resolution.requirement_policy.value} "
            f"environment={self.resolution.environment_selection.source} environment_name={environment_name} "
            f"environment_attempts={attempts} environment_probes={probes} "
            f"world={self.resolution.world_selection.source} runtime={self.resolution.runtime_selection.source} "
            f"launchable={self.launchable}{inventory_text}{blocking_text}"
        )


def effective_requirement_policy(explicit: RequirementPolicy | str | None, enforcement: RuntimeEnforcement | str | None = None) -> RequirementPolicy:
    """Return an explicit policy or the context-local enforcement-derived default."""

    if explicit is not None:
        try:
            return RequirementPolicy(explicit)
        except ValueError as exc:
            raise DispatchPlanningError("invalid requirement_policy; expected strict, warn, or ignore", context={"requirement_policy": explicit}) from exc
    current = runtime.enforcement() if enforcement is None else RuntimeEnforcement(enforcement)
    return {
        RuntimeEnforcement.STRICT: RequirementPolicy.STRICT,
        RuntimeEnforcement.WARN: RequirementPolicy.WARN,
        RuntimeEnforcement.OFF: RequirementPolicy.IGNORE,
    }[current]


def parse_analysis_policy(policy: Any | None) -> _DynamicTraceRequest:
    """Strictly parse dispatch analysis policy before normalization or tracing.

    A direct context is a compatibility form and deliberately cannot request a
    trace, even if it grants dynamic-execution permission.  The mapping form is
    closed so misspellings cannot silently authorize target invocation. The
    mapping and context metadata are copied once into the immutable private
    request used by a public planning entrypoint; an explicit ``context=None``
    is invalid rather than an omitted context.
    """

    if isinstance(policy, _DynamicTraceRequest):
        raise DispatchPlanningError("analysis_policy must use the public CodeAnalysisContext or mapping form")
    if policy is None:
        return _DynamicTraceRequest(_snapshot_analysis_context(CodeAnalysisContext()), DEFAULT_PROBE_TIMEOUT_S)
    if isinstance(policy, CodeAnalysisContext):
        return _DynamicTraceRequest(_snapshot_analysis_context(policy), DEFAULT_PROBE_TIMEOUT_S)
    if not isinstance(policy, Mapping):
        raise DispatchPlanningError("analysis_policy must be a CodeAnalysisContext or mapping")
    policy = dict(policy)
    if any(type(key) is not str for key in policy):
        raise DispatchPlanningError(
            "analysis_policy contains unsupported fields",
            context={"fields": ["<non-string>"]},
        )
    unknown = set(policy) - {"context", "probe_timeout_s", "dynamic_trace"}
    if unknown:
        raise DispatchPlanningError("analysis_policy contains unsupported fields", context={"fields": sorted(unknown)})
    if "context" not in policy:
        context = CodeAnalysisContext()
    else:
        context = policy["context"]
    if not isinstance(context, CodeAnalysisContext):
        raise DispatchPlanningError("analysis_policy.context must be a CodeAnalysisContext")
    timeout = policy.get("probe_timeout_s", DEFAULT_PROBE_TIMEOUT_S)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise DispatchPlanningError("analysis_policy.probe_timeout_s must be a positive number")
    if "dynamic_trace" not in policy:
        trace_policy = None
    elif type(policy["dynamic_trace"]) is bool and policy["dynamic_trace"] is True:
        trace_policy = DynamicTracePolicy()
    elif isinstance(policy["dynamic_trace"], DynamicTracePolicy):
        trace_policy = policy["dynamic_trace"]
    else:
        raise DispatchPlanningError("analysis_policy.dynamic_trace must be exactly True or a DynamicTracePolicy")
    return _DynamicTraceRequest(_snapshot_analysis_context(context), float(timeout), trace_policy)


def _snapshot_analysis_context(context: CodeAnalysisContext) -> CodeAnalysisContext:
    """Return a private context with an isolated JSON metadata snapshot.

    Dispatch derives discovery, trace identity, and facade metadata from this
    value only.  A frozen context does not make a caller-owned nested metadata
    mapping immutable, so copying it at the public policy boundary prevents a
    later mutation from changing one request's effective analysis identity.
    """

    metadata = json.loads(json.dumps(context.metadata, sort_keys=True, separators=(",", ":")))
    return CodeAnalysisContext(
        algorithms=tuple(context.algorithms),
        allow_import=context.allow_import,
        allow_source=context.allow_source,
        allow_dynamic_execution=context.allow_dynamic_execution,
        include_annotations=context.include_annotations,
        include_method_contracts=context.include_method_contracts,
        diagnostics_policy=context.diagnostics_policy,
        metadata=metadata,
    )


def resolve_dispatch_plan(
    normalized: NormalizedDispatchTarget,
    *,
    environment: Any | None = None,
    world: Any | None = None,
    runtime_spec: Any | None = None,
    requirement_policy: RequirementPolicy | str | None = None,
    analysis_policy: Any | None = None,
    emit_warnings: bool = False,
    single_worker_only: bool = False,
    environment_candidates: Any | None = None,
    environment_registry: Any | None = None,
    inventory: worlds.LocalResourceInventory | None = None,
    inventory_policy: str = "lightweight",
    resolver_policy: str | None = None,
    _analysis_request: _DynamicTraceRequest | None = None,
) -> DispatchPlanningResolution:
    """Resolve requirements and candidate checks for one normalized target.

    Args:
        normalized: Already-normalized operation target to evaluate.
        environment: Optional explicit environment candidate.
        world: Optional explicit requested world candidate.
        runtime_spec: Optional explicit runtime candidate.
        requirement_policy: Strict, warning, or ignore requirement policy.
        analysis_policy: Optional code-analysis/probe policy.
        emit_warnings: Emit warning diagnostics through the reporting path.
        single_worker_only: Require a locally enactable single worker world.
        environment_candidates: Ordered resolver candidates after higher slots.
        environment_registry: Explicit registry used by resolver search.
        inventory: Injected local inventory reused for synthesis and allocation.
        inventory_policy: Local inventory discovery policy when not injected.
        resolver_policy: Optional environment resolver policy override.

    Returns:
        A bounded planning resolution without workload allocation.

    ``normalized`` is intentionally the only target input. The resolver never
    calls normalization and therefore retains live annotation targets supplied
    by the public normalization boundary.
    """

    enforcement = runtime.enforcement()
    policy = effective_requirement_policy(requirement_policy, enforcement)
    _validate_sprint8_policies(inventory_policy, resolver_policy)
    if inventory is not None and not isinstance(inventory, worlds.LocalResourceInventory):
        raise DispatchPlanningError("inventory must be a LocalResourceInventory")
    if _analysis_request is None:
        analysis_request = parse_analysis_policy(analysis_policy)
    elif type(_analysis_request) is _DynamicTraceRequest:
        analysis_request = _analysis_request
    else:
        raise DispatchPlanningError("invalid internal dispatch analysis request")
    analysis_context, probe_timeout_s = analysis_request.context, analysis_request.probe_timeout_s
    fragments, analysis, bootstrap_probe, bootstrap_environment, discovery_diagnostics, complete = _discover(
        normalized,
        environment,
        analysis_context=analysis_context,
        probe_timeout_s=probe_timeout_s,
    )
    trace_provenance = None
    trace_diagnostics: tuple[DiagnosticFact, ...] = ()
    if analysis_request.requested:
        preliminary_environment = None
        if normalized.launch.get("same_environment_only"):
            # pickle_small is the one transport whose direct candidate must be
            # accepted before trusted current-process target execution.  This
            # preliminary selection is intentionally repeated after accepted
            # trace facts, because those facts can change the final candidate.
            preliminary = annotations.resolve_fragments(
                fragments,
                source="dryml.dispatch.dynamic_trace",
            )
            _, preliminary_environment, _ = _select_environment(
                environment,
                preliminary.environment_default,
                requirement=preliminary.environment_requirement,
                candidates=environment_candidates,
                registry=environment_registry,
                resolver_policy=resolver_policy,
                bootstrap_probe=bootstrap_probe,
                bootstrap_environment=bootstrap_environment,
            )
        fragments, trace_provenance, trace_diagnostics = _trace_dispatch_invocation(
            normalized,
            analysis_request,
            fragments,
            preliminary_environment=preliminary_environment,
        )
        complete = complete and trace_provenance.data["status"] == "complete"
    resolution = annotations.resolve_fragments(
        fragments,
        source="dryml.dispatch.dynamic_trace" if analysis_request.requested else "dryml.dispatch",
    )
    diagnostics = list(discovery_diagnostics)
    diagnostics.extend(trace_diagnostics)
    diagnostics.extend(_annotation_diagnostics(resolution))

    env_selection, env_spec, environment_resolution = _select_environment(
        environment,
        resolution.environment_default,
        requirement=resolution.environment_requirement,
        candidates=environment_candidates,
        registry=environment_registry,
        resolver_policy=resolver_policy,
        bootstrap_probe=bootstrap_probe,
        bootstrap_environment=bootstrap_environment,
    )
    world_selection, world_spec, world_synthesis = _select_world(
        world,
        resolution.world_default,
        requirement=resolution.world_requirement,
        inventory=inventory,
        inventory_policy=inventory_policy,
    )
    runtime_selection, selected_runtime = _select_runtime(runtime_spec, resolution.runtime_default)
    structural_safe = trace_provenance is None or trace_provenance.data["status"] == "complete"
    if environment_resolution is not None and not environment_resolution.ok:
        incomplete = environment_resolution.status == "incomplete"
        diagnostics.append(_diagnostic(
            "dryml.dispatch.environment_resolver_incomplete" if incomplete else "dryml.dispatch.environment_resolver_no_match",
            "Environment candidate input was truncated before compatibility could be determined; pass, register, or set a compatible environment."
            if incomplete
            else "No resolver candidate satisfied the environment requirement; pass, register, or set a compatible environment.",
            severity="error" if incomplete or policy is RequirementPolicy.STRICT else "warning",
            data=environment_resolution.to_data(),
        ))
        # An incomplete resolver input may contain a compatible higher-precedence
        # candidate. Never probe or launch the fallback in that case.
        structural_safe = structural_safe and not incomplete and policy is not RequirementPolicy.STRICT
    if world_synthesis is not None and not world_synthesis.ok:
        synthesis_structural = world_synthesis.status not in {"insufficient_inventory", "unsupported_requirement"}
        diagnostics.append(_diagnostic(
            "dryml.dispatch.world_synthesis_failed",
            "Local world synthesis failed; inject inventory or pass/set a compatible world.",
            severity="error" if synthesis_structural or policy is RequirementPolicy.STRICT else "warning",
            data=world_synthesis.to_data(),
        ))
        # Ignore may run a feasible fallback without claiming it supplies the
        # skipped requirement. Discovery and malformed-input failures cannot.
        structural_safe = structural_safe and (
            policy is RequirementPolicy.IGNORE
            and world_synthesis.status in {"insufficient_inventory", "unsupported_requirement"}
        )
    if normalized.launch.get("same_environment_only") and not _same_python_environment(env_spec):
        structural_safe = False
        diagnostics.append(_diagnostic(
            "dryml.dispatch.pickle_environment_restriction",
            "Pickled callable transport requires the current Python executable.",
            data={"candidate": env_spec, "restriction": "same_environment_only"},
        ))
    if single_worker_only:
        world_diagnostics = _local_subprocess_world_diagnostics(
            world_spec,
            resolution.world_requirement,
            policy,
        )
        if world_diagnostics:
            structural_failures = tuple(item for item in world_diagnostics if item.code != "dryml.dispatch.single_subprocess_requirement_unsupported")
            structural_safe = structural_safe and (policy is not RequirementPolicy.STRICT and not structural_failures or policy is RequirementPolicy.STRICT and not world_diagnostics)
            diagnostics.extend(world_diagnostics)
    else:
        world_diagnostics = _local_world_topology_diagnostics(resolution.world_requirement)
        if world_diagnostics:
            # The same-host coordinator cannot enact collectives or shared
            # filesystem guarantees, regardless of compatibility policy.
            structural_safe = False
            diagnostics.extend(world_diagnostics)
    if single_worker_only and _is_multi_worker_world(world_spec):
        structural_safe = False
        if not world_diagnostics:
            diagnostics.append(_diagnostic(
                "dryml.dispatch.single_subprocess_world_unsupported",
                "The local subprocess planner supports one worker only; use plan_world() or run_world() for this world.",
                data={"world": world_spec},
            ))

    final_probe = None
    resolver_incomplete = environment_resolution is not None and environment_resolution.status == "incomplete"
    if not resolver_incomplete and _needs_final_probe(normalized, env_spec, bootstrap_environment):
        final_probe = probe_target(
            normalized.code_target,
            environment=spec_from_data(env_spec),
            include_environment_record=environment_resolution is None or environment_resolution.selected_record is None,
            timeout=probe_timeout_s,
        )
        diagnostics.extend(final_probe.diagnostics)
        final_fragments = _fragments_from_analysis(final_probe.analysis) if final_probe.analysis is not None else []
        if final_fragments:
            reconciled_fragments = _dedupe_fragments((*fragments, *final_fragments))
            reconciled = annotations.resolve_fragments(reconciled_fragments, source="dryml.dispatch.final_probe")
            if _resolution_decisions(reconciled) != _resolution_decisions(resolution) and (bootstrap_probe is None or bootstrap_probe.ok):
                structural_safe = False
                diagnostics.append(_diagnostic(
                    "dryml.dispatch.final_probe_annotation_mismatch",
                    "Final environment probe discovered annotation facts that change resolved requirements or defaults.",
                    data={"bootstrap_fragments": len(fragments), "final_fragments": len(final_fragments)},
                ))
            resolution = reconciled
            diagnostics.extend(_annotation_diagnostics(resolution))
            if bootstrap_probe is not None and not bootstrap_probe.ok:
                # A final probe can reveal an annotation default unavailable to
                # bootstrap discovery. Do not continue with a lower-precedence
                # resolver/current candidate unless it is the same environment.
                if resolution.environment_default is not None:
                    default_environment = _environment_data(resolution.environment_default)
                    if default_environment != env_spec and env_selection.source != "explicit":
                        structural_safe = False
                        diagnostics.append(_diagnostic(
                            "dryml.dispatch.final_probe_annotation_mismatch",
                            "Final environment probe discovered a higher-precedence annotation-default environment.",
                            data={
                                "selected_source": env_selection.source,
                                "selected_environment": env_spec,
                                "annotation_default": default_environment,
                                "action": "pass the annotation-default environment explicitly and plan again",
                            },
                        ))
                    elif default_environment == env_spec:
                        env_selection, env_spec, _ = _select_environment(
                            environment,
                            resolution.environment_default,
                            requirement=resolution.environment_requirement,
                        )
                # Bootstrap did not provide usable requirements. Final-probe
                # facts are therefore authoritative for world/runtime selection
                # without restarting environment search after target validation.
                world_selection, world_spec, world_synthesis = _select_world(
                    world,
                    resolution.world_default,
                    requirement=resolution.world_requirement,
                    inventory=inventory or (world_synthesis.resource_inventory if world_synthesis is not None else None),
                    inventory_policy=inventory_policy,
                    inventory_discovery_error=(
                        None
                        if world_synthesis is None
                        else world_synthesis.inventory_discovery_error
                    ),
                )
                runtime_selection, selected_runtime = _select_runtime(runtime_spec, resolution.runtime_default)
                if world_synthesis is not None and not world_synthesis.ok:
                    synthesis_structural = world_synthesis.status not in {"insufficient_inventory", "unsupported_requirement"}
                    diagnostics.append(_diagnostic(
                        "dryml.dispatch.world_synthesis_failed",
                        "Local world synthesis failed; inject inventory or pass/set a compatible world.",
                        severity="error" if synthesis_structural or policy is RequirementPolicy.STRICT else "warning",
                        data=world_synthesis.to_data(),
                    ))
                    structural_safe = structural_safe and (
                        policy is RequirementPolicy.IGNORE
                        and world_synthesis.status in {"insufficient_inventory", "unsupported_requirement"}
                    )
                if single_worker_only:
                    world_diagnostics = _local_subprocess_world_diagnostics(
                        world_spec,
                        resolution.world_requirement,
                        policy,
                    )
                else:
                    world_diagnostics = _local_world_topology_diagnostics(resolution.world_requirement)
                if world_diagnostics:
                    if single_worker_only:
                        structural_failures = tuple(
                            item
                            for item in world_diagnostics
                            if item.code != "dryml.dispatch.single_subprocess_requirement_unsupported"
                        )
                        structural_safe = structural_safe and (
                            policy is not RequirementPolicy.STRICT and not structural_failures
                            or policy is RequirementPolicy.STRICT and not world_diagnostics
                        )
                    else:
                        structural_safe = False
                    diagnostics.extend(world_diagnostics)
                if single_worker_only and _is_multi_worker_world(world_spec):
                    structural_safe = False
                    if not world_diagnostics:
                        diagnostics.append(_diagnostic(
                            "dryml.dispatch.single_subprocess_world_unsupported",
                            "The local subprocess planner supports one worker only; use plan_world() or run_world() for this world.",
                            data={"world": world_spec},
                        ))
        # A successful final probe validates the environment that will execute
        # the target and supersedes failed/incomplete bootstrap discovery.
        if final_probe.ok and final_probe.analysis is not None:
            complete = True
            if bootstrap_probe is not None and not bootstrap_probe.ok:
                diagnostics = [item for item in diagnostics if item not in bootstrap_probe.diagnostics]
        if not final_probe.ok:
            diagnostics.append(_diagnostic(
                "dryml.dispatch.final_environment_probe_failed",
                "Final selected environment could not validate the dispatch target.",
                severity="error",
                data={"environment": env_spec, "timeout_s": probe_timeout_s},
            ))

    env_record, env_probe_diagnostics = _environment_record(
        env_spec,
        resolution.environment_requirement,
        final_probe or bootstrap_probe,
        env_spec if final_probe is not None else bootstrap_environment,
        environment,
        policy,
        validate_candidate=_requires_environment_validation(env_spec, normalized),
        resolved_record=_resolution_record_for(environment_resolution, env_spec),
        resolved_probe=_resolution_probe_for(environment_resolution, env_spec),
        resolver_incomplete=resolver_incomplete,
    )
    diagnostics.extend(env_probe_diagnostics)
    if env_probe_diagnostics:
        # Candidate validation is structural even when requirement compatibility
        # itself is relaxed by warn/ignore.
        structural_safe = False
    environment_check = _check_environment(
        resolution.environment_requirement,
        env_spec,
        env_record,
        policy,
        env_probe_diagnostics,
        validate_candidate=_requires_environment_validation(env_spec, normalized),
    )
    world_check = _check_world(resolution.world_requirement, world_spec, policy)
    runtime_check = _check_runtime(resolution.runtime_requirement, selected_runtime, policy)
    diagnostics.extend(_check_diagnostics(
        (environment_check, world_check, runtime_check),
        policy,
        (env_selection, world_selection, runtime_selection),
    ))

    selected_inventory = inventory
    if world_synthesis is not None:
        selected_inventory = world_synthesis.resource_inventory
    if (
        single_worker_only
        and not _is_multi_worker_world(world_spec)
        and (world_synthesis is None or world_synthesis.resource_inventory is not None)
        and (
            world_selection.source == "fallback"
            or _world_needs_inventory(world_spec)
            or world_synthesis is not None
            or resolution.world_requirement is not None
        )
    ):
        try:
            from .local_world import validate_local_world_feasibility

            selected_inventory = selected_inventory or worlds.local_inventory(policy=inventory_policy)
            allocation_world = _subprocess_allocation_world(world_spec)
            validate_local_world_feasibility(
                allocation_world,
                inventory=selected_inventory,
                allocation_backend_kind="local_subprocess",
            )
            allocation_requirement = _effective_local_allocation_requirement_check(
                allocation_world,
                resolution.world_requirement,
            )
            if allocation_requirement is not None and not allocation_requirement.ok:
                if policy is not RequirementPolicy.IGNORE:
                    diagnostics.append(_diagnostic(
                        "dryml.dispatch.local_allocation_requirement_failed",
                        "The executable local subprocess resource assignment does not satisfy the hard world requirement.",
                        severity="error" if policy is RequirementPolicy.STRICT else "warning",
                        data={"issues": [_world_issue_data(issue) for issue in allocation_requirement.issues]},
                    ))
                if policy is RequirementPolicy.STRICT:
                    structural_safe = False
        except Exception as exc:
            structural_safe = False
            diagnostics.append(_diagnostic(
                "dryml.dispatch.local_allocation_failed",
                "The selected one-worker world cannot be allocated from local inventory.",
                data={"error": type(exc).__name__, "message": str(exc), "action": "inject inventory or pass/set a feasible world"},
            ))

    bootstrap_probe_failed = bootstrap_probe is not None and not bootstrap_probe.ok and (final_probe is None or not final_probe.ok)
    if not complete:
        diagnostics.append(_diagnostic("dryml.dispatch.discovery_incomplete", "Requirement discovery is incomplete.", severity="error" if policy is RequirementPolicy.STRICT else "warning", data={"policy": policy.value, "action": "use an importable target or call dispatch.explain(...)"}))
    checks = (environment_check, world_check, runtime_check)
    merge_safe = not _has_annotation_errors(resolution)
    if policy is RequirementPolicy.STRICT:
        launchable = structural_safe and merge_safe and complete and (final_probe is None or final_probe.ok) and all(report.compatible is not False and report.status != "error" for report in checks)
    else:
        # A failed bootstrap probe proves the target cannot be imported in the
        # selected current environment. Policy relaxes compatibility, never
        # target importability.
        launchable = structural_safe and merge_safe and not bootstrap_probe_failed and (final_probe is None or final_probe.ok)
    if emit_warnings and policy is RequirementPolicy.WARN:
        warning_items = [item for item in diagnostics if item.severity in {"warning", "error"}]
        if warning_items:
            warnings.warn("; ".join(item.message for item in warning_items), RuntimeWarning, stacklevel=3)

    return DispatchPlanningResolution(
        normalized_target=_normalized_target_data(normalized),
        code_analysis=analysis,
        code_probe=final_probe or bootstrap_probe,
        bootstrap_environment=bootstrap_environment,
        bootstrap_code_probe=bootstrap_probe,
        final_code_probe=final_probe,
        requirements=resolution,
        environment_selection=env_selection,
        environment_record=env_record,
        environment_check=environment_check,
        world_selection=world_selection,
        world_check=world_check,
        runtime_selection=runtime_selection,
        runtime_check=runtime_check,
        requirement_policy=policy,
        runtime_enforcement=enforcement,
        diagnostics=tuple(diagnostics),
        launchable=launchable,
        environment_resolution=environment_resolution,
        world_synthesis=world_synthesis,
        # The implicit no-requirement fallback still retains inventory for the
        # planner, but its volatile host-capacity observations are not dispatch
        # intent metadata. Synthesis and resource requirements retain the
        # required inventory evidence.
        inventory_summary=(
            None
            if selected_inventory is None
            or (world_synthesis is None and resolution.world_requirement is None)
            else selected_inventory.summary()
        ),
        world_allocation_summary=None,
        local_inventory=selected_inventory,
        dynamic_trace=trace_provenance,
    )


def explanation_for(normalized: NormalizedDispatchTarget, **kwargs: Any) -> DispatchExplanation:
    """Resolve a normalized target without launching or emitting warnings."""

    result = resolve_dispatch_plan(normalized, emit_warnings=False, **kwargs)
    errors = tuple(item for item in result.diagnostics if item.severity == "error")
    # Structural failures stay blocking under warn/ignore even when their
    # compatibility finding is intentionally presented as a warning.
    blocking = () if result.launchable else errors or tuple(result.diagnostics)
    return DispatchExplanation(result, dict(normalized.operation_spec), blocking)


def _trace_dispatch_invocation(
    normalized: NormalizedDispatchTarget,
    request: _DynamicTraceRequest,
    direct_fragments: tuple[annotations.AnnotationFragment, ...],
    *,
    preliminary_environment: Mapping[str, Any] | None = None,
) -> tuple[tuple[annotations.AnnotationFragment, ...], DynamicTraceProvenance, tuple[DiagnosticFact, ...]]:
    """Trace one worker-effective invocation and admit only complete evidence.

    This is intentionally a dispatch carrier, not an analysis implementation:
    it derives the worker argument grammar, calls the public code facade once,
    then returns typed annotation fragments to the sole annotations resolver.
    """

    target = _trace_target_data(normalized)
    live_target = normalized.trace_live_target
    try:
        trace_args, trace_kwargs = _effective_trace_invocation(normalized)
        facade_target = _trace_facade_target_data(normalized, live_target, request.context.metadata)
        input_id = _trace_input_id(normalized, request.policy, trace_args, trace_kwargs, facade_target)
    except Exception:
        provenance = _trace_provenance(
            normalized, request.policy, target=target, status="pre_execution_failed",
            input_id=None, run_id=None, started=False,
            diagnostics=[_trace_diagnostic("dryml.dispatch.dynamic_trace_unsupported_input")],
        )
        return direct_fragments, provenance, (_trace_diagnostic("dryml.dispatch.dynamic_trace_unsupported_input"),)

    if preliminary_environment is not None and not _same_python_environment(preliminary_environment):
        diagnostic = _trace_diagnostic("dryml.dispatch.dynamic_trace_unsupported_input")
        provenance = _trace_provenance(
            normalized, request.policy, target=target, status="pre_execution_failed",
            input_id=input_id, run_id=None, started=False, diagnostics=[diagnostic],
        )
        return direct_fragments, provenance, (diagnostic,)

    if type(live_target) is not types.FunctionType:
        provenance = _trace_provenance(
            normalized, request.policy, target=target, status="pre_execution_failed",
            input_id=input_id, run_id=None, started=False,
            diagnostics=[_trace_diagnostic("dryml.dispatch.dynamic_trace_unsupported_input")],
        )
        return direct_fragments, provenance, (_trace_diagnostic("dryml.dispatch.dynamic_trace_unsupported_input"),)

    run_id = f"trace-run-v1-{uuid.uuid4().hex}"
    trace_context = replace(
        request.context,
        allow_dynamic_execution=True,
        algorithms=("dynamic_trace",),
        diagnostics_policy="collect",
        metadata={
            **request.context.metadata,
            _TRACE_CORRELATION_INPUT_KEY: input_id,
            _TRACE_CORRELATION_RUN_KEY: run_id,
        },
    )
    result = trace(live_target, args=trace_args, kwargs=trace_kwargs, context=trace_context, policy=request.policy)
    expected_target = target_from_callable(live_target, metadata=trace_context.metadata).spec.to_data()
    return _admit_trace_result(normalized, request.policy, direct_fragments, target, input_id, run_id, expected_target, result)


def _effective_trace_invocation(normalized: NormalizedDispatchTarget) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Use the operation resolver with non-building planning-store callbacks."""

    store = normalized.trace_store
    operation = dict(normalized.operation_spec)
    payload = dict(operation.get("payload") or {})
    if normalized.transport == "pickle_small":
        args = payload.get("args")
        count = normalized.launch.get("identity_arg_count")
        if (
            type(args) is not list
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
            or count > len(args)
            or len(args[count:]) != 1
            or type(args[count]) is not dict
            or set(args[count]) != {"$literal"}
            or not isinstance(args[count]["$literal"], str)
            or re.fullmatch(r"dryml\.pickled_callable\.sha256:[0-9a-f]{64}", args[count]["$literal"]) is None
        ):
            raise ValueError("invalid pickle_small trace marker")
        payload["args"] = args[:count]
        operation["payload"] = payload
        operation.pop("id", None)

    def materialize(cdef_id: str) -> ConcreteDefinition:
        if store is None:
            raise ValueError("trace CDef invocation requires a planning store")
        stored = _load_trace_cdef(store, cdef_id)
        supplied = normalized.trace_cdef_side_table.get(cdef_id, ())
        if any(stored != live for live in supplied):
            raise ValueError("stored CDef differs from caller CDef")
        return supplied[0] if supplied else stored

    resolved = resolve_call_arguments(operation, materialize_cdef=materialize, make_cdef_ref=lambda cdef_id: cdef_id)
    return tuple(resolved.args), dict(resolved.kwargs)


def _load_trace_cdef(store: Any, cdef_id: str) -> ConcreteDefinition:
    """Load a stored CDef structurally without building a DRYML object."""

    path = os.path.join(store.object_dir_for_cdef_id(cdef_id), "def.pkl")
    if not os.path.isfile(path):
        raise ValueError("stored CDef is unavailable")
    value = pickle_load(path)
    if not isinstance(value, ConcreteDefinition):
        raise ValueError("stored definition is not concrete")
    return value


def _trace_input_id(
    normalized: NormalizedDispatchTarget,
    policy: DynamicTracePolicy | None,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    facade_target: Mapping[str, Any],
) -> str:
    if policy is None:
        raise ValueError("missing trace policy")
    record = {
        "schema": "dryml.dispatch.trace_input.v1",
        "operation_id": normalized.operation_spec.get("id"),
        "operation_kind": normalized.operation_spec.get("kind"),
        "facade_target": dict(facade_target),
        "canonical_invocation": (normalized.operation_spec.get("payload") or {}),
        "effective_invocation": {"args": _trace_identity_value(args), "kwargs": _trace_identity_value(dict(kwargs))},
        "cdef_positions": list(normalized.trace_cdef_positions),
        "policy": _trace_policy_data(policy),
    }
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return "trace-input-v1-" + hashlib.sha256(encoded).hexdigest()


def _trace_identity_value(value: Any) -> Any:
    if isinstance(value, ConcreteDefinition):
        return {"$trace_cdef": value.stable_hash()}
    if type(value) is tuple:
        return [_trace_identity_value(item) for item in value]
    if type(value) is list:
        return [_trace_identity_value(item) for item in value]
    if type(value) is dict:
        return {key: _trace_identity_value(item) for key, item in value.items()}
    return value


def _trace_target_data(normalized: NormalizedDispatchTarget) -> dict[str, str] | None:
    if normalized.code_target is None:
        return None
    return {"target_kind": normalized.code_target.kind, "transport": normalized.transport}


def _trace_facade_target_data(
    normalized: NormalizedDispatchTarget,
    live_target: Any,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the complete sanitized target description used to bind trace input."""

    if type(live_target) is types.FunctionType:
        return target_from_callable(live_target, metadata=metadata).spec.to_data()
    if normalized.code_target is None:
        raise ValueError("trace input has no normalized target description")
    return normalized.code_target.to_data()


def _trace_policy_data(policy: DynamicTracePolicy) -> dict[str, Any]:
    return {
        "max_calls": policy.max_calls,
        "require_proxy_only_args": policy.require_proxy_only_args,
        "collect_requirements": policy.collect_requirements,
    }


def _trace_diagnostic(code: str, *, trace_diagnostic_codes: tuple[str, ...] = ()) -> DiagnosticFact:
    messages = {
        "dryml.dispatch.dynamic_trace_unsupported_input": "Requested dynamic trace input is unsupported.",
        "dryml.dispatch.dynamic_trace_failed": "Requested dynamic trace failed after execution started.",
        "dryml.dispatch.dynamic_trace_incomplete": "Requested dynamic trace is incomplete.",
        "dryml.dispatch.dynamic_trace_identity_mismatch": "Dynamic trace evidence does not match this request.",
        "dryml.dispatch.dynamic_trace_evidence_rejected": "Dynamic trace evidence was rejected.",
        "dryml.dispatch.dynamic_trace_provenance_limit_exceeded": "Dynamic trace provenance exceeds dispatch limits.",
    }
    return _diagnostic(code, messages[code], data={"trace_diagnostic_codes": list(trace_diagnostic_codes)})


def _trace_provenance(
    normalized: NormalizedDispatchTarget,
    policy: DynamicTracePolicy | None,
    *,
    target: dict[str, str] | None,
    status: str,
    input_id: str | None,
    run_id: str | None,
    started: bool | None,
    summary: Mapping[str, Any] | None = None,
    calls: list[dict[str, Any]] | None = None,
    accepted_fragments: list[dict[str, Any]] | None = None,
    duplicate_observations: list[dict[str, Any]] | None = None,
    diagnostics: list[DiagnosticFact] | None = None,
) -> DynamicTraceProvenance:
    del normalized
    return DynamicTraceProvenance({
        "schema": _TRACE_SCHEMA,
        "schema_version": 1,
        "requested": True,
        "trace_input_id": input_id,
        "trace_run_id": run_id,
        "execution_location": "current_process",
        "execution_started": started,
        "target": target,
        "policy": _trace_policy_data(policy) if policy is not None else {"max_calls": 0, "require_proxy_only_args": True, "collect_requirements": True},
        "status": status,
        "summary": None if summary is None else dict(summary),
        "calls": [] if calls is None else calls,
        "accepted_fragments": [] if accepted_fragments is None else accepted_fragments,
        "duplicate_observations": [] if duplicate_observations is None else duplicate_observations,
        "diagnostics": [_trace_diagnostic_data(item) for item in diagnostics or ()],
    })


def _trace_diagnostic_data(item: DiagnosticFact) -> dict[str, Any]:
    """Project one diagnostic through the fixed redacted trace wire schema."""

    raw_codes = item.data.get("trace_diagnostic_codes", ()) if type(item.data) is dict else ()
    codes = list(raw_codes) if type(raw_codes) in {tuple, list} else []
    return {
        "code": item.code,
        "severity": item.severity,
        "data": {"trace_diagnostic_codes": codes},
    }


def _admit_trace_result(
    normalized: NormalizedDispatchTarget,
    policy: DynamicTracePolicy,
    direct_fragments: tuple[annotations.AnnotationFragment, ...],
    target: dict[str, str] | None,
    input_id: str,
    run_id: str,
    expected_target: Mapping[str, Any],
    result: Any,
) -> tuple[tuple[annotations.AnnotationFragment, ...], DynamicTraceProvenance, tuple[DiagnosticFact, ...]]:
    """Validate 9B result shape before any fact reaches requirement resolution."""

    # Evidence has its own schema.  Validate it before comparing the result
    # envelope so a stale target cannot erase a bounded summary/call prefix that
    # independently proves execution started.  Rejected evidence remains
    # diagnostic-only and never reaches fragment resolution.
    summary = _independently_validated_trace_summary(result, policy=policy, target=target)
    try:
        calls = _independently_validated_trace_calls(result, target=target)
    except Exception:
        calls = None
    try:
        admitted_calls, admitted_summary = _validated_trace_result_evidence(result, policy=policy, target=target)
        sequences = [fact.data["sequence"] for fact in admitted_calls]
        if sequences != list(range(len(admitted_calls))) or admitted_summary["data"]["calls_recorded"] != len(admitted_calls):
            raise ValueError("invalid call sequence")
    except Exception:
        admitted_calls = None
        admitted_summary = None
    expected_envelope = _trace_result_has_expected_envelope(normalized, result, expected_target)
    safe_diagnostic_codes = _safe_trace_result_diagnostic_codes(result)
    persistence_calls = admitted_calls if admitted_calls is not None else calls
    if persistence_calls is not None:
        try:
            _validate_trace_call_persistence(persistence_calls)
        except ValueError:
            diagnostic = _trace_diagnostic("dryml.dispatch.dynamic_trace_evidence_rejected")
            provenance = _rejected_trace_provenance(
                normalized, policy, target, input_id, run_id, True, None, [], diagnostic,
            )
            return direct_fragments, provenance, (diagnostic,)
    if (
        expected_envelope and not result.facts and result.diagnostics
        and all(item.code in _TRACE_PREEXEC_CODES for item in result.diagnostics)
    ):
        if len(safe_diagnostic_codes) > _TRACE_MAX_DIAGNOSTICS:
            diagnostic = _trace_diagnostic("dryml.dispatch.dynamic_trace_provenance_limit_exceeded")
        else:
            diagnostic = _trace_diagnostic("dryml.dispatch.dynamic_trace_unsupported_input", trace_diagnostic_codes=safe_diagnostic_codes)
        provenance = _trace_provenance(normalized, policy, target=target, status="pre_execution_failed", input_id=input_id, run_id=None, started=False, diagnostics=[diagnostic])
        return direct_fragments, provenance, (diagnostic,)
    mixed_preexecution_diagnostic = (
        type(result) is CodeAnalysisResult
        and type(result.diagnostics) is tuple
        and any(type(item) is DiagnosticFact and item.code in _TRACE_PREEXEC_CODES for item in result.diagnostics)
    )
    if not expected_envelope or mixed_preexecution_diagnostic or admitted_calls is None or admitted_summary is None:
        diagnostic = _trace_diagnostic(
            "dryml.dispatch.dynamic_trace_identity_mismatch"
            if not expected_envelope else "dryml.dispatch.dynamic_trace_evidence_rejected"
            , trace_diagnostic_codes=safe_diagnostic_codes
        )
        started = True if summary is not None or calls is not None else None
        provenance = _rejected_trace_provenance(
            normalized, policy, target, input_id, run_id, started, summary,
            [] if calls is None else [fact.to_data() for fact in calls], diagnostic,
        )
        return direct_fragments, provenance, (diagnostic,)

    if summary["data"]["complete"] is True and result.diagnostics:
        # A complete 9B summary cannot carry post-start diagnostics.  This is
        # inconsistent result evidence, not an incomplete trace outcome: the
        # incomplete-carrier invariant intentionally requires an incomplete
        # summary.  Preserve independently validated start evidence only in the
        # rejected carrier so plan and explain can report their bounded failure.
        diagnostic = _trace_diagnostic(
            "dryml.dispatch.dynamic_trace_evidence_rejected",
            trace_diagnostic_codes=safe_diagnostic_codes,
        )
        provenance = _rejected_trace_provenance(
            normalized, policy, target, input_id, run_id, True, summary,
            [fact.to_data() for fact in admitted_calls], diagnostic,
        )
        return direct_fragments, provenance, (diagnostic,)

    if calls is not None and summary is not None and len(calls) > _TRACE_MAX_CALLS:
        diagnostic = _trace_diagnostic("dryml.dispatch.dynamic_trace_provenance_limit_exceeded")
        provenance = _provenance_limit_exceeded(normalized, policy, target, input_id, run_id, summary)
        return direct_fragments, provenance, (diagnostic,)

    if summary["data"]["complete"] is not True:
        status = "failed" if summary["data"].get("outcome") == "target_failed" else "incomplete"
        diagnostic = _trace_diagnostic(
            "dryml.dispatch.dynamic_trace_failed" if status == "failed" else "dryml.dispatch.dynamic_trace_incomplete",
            trace_diagnostic_codes=safe_diagnostic_codes,
        )
        try:
            provenance = _trace_provenance(normalized, policy, target=target, status=status, input_id=input_id, run_id=run_id, started=True, summary=summary, calls=[fact.to_data() for fact in admitted_calls], diagnostics=[diagnostic])
        except _TraceProvenanceLimitError:
            provenance = _provenance_limit_exceeded(normalized, policy, target, input_id, run_id, summary)
        return direct_fragments, provenance, (diagnostic,)

    fragments, accepted, duplicates = _combine_trace_fragments(direct_fragments, admitted_calls)
    if len(accepted) > _TRACE_MAX_FRAGMENTS or len(duplicates) > _TRACE_MAX_DUPLICATES:
        diagnostic = _trace_diagnostic("dryml.dispatch.dynamic_trace_provenance_limit_exceeded")
        provenance = _provenance_limit_exceeded(normalized, policy, target, input_id, run_id, summary)
        return direct_fragments, provenance, (diagnostic,)
    try:
        provenance = _trace_provenance(normalized, policy, target=target, status="complete", input_id=input_id, run_id=run_id, started=True, summary=summary, calls=[fact.to_data() for fact in admitted_calls], accepted_fragments=accepted, duplicate_observations=duplicates)
    except _TraceProvenanceLimitError:
        provenance = _provenance_limit_exceeded(normalized, policy, target, input_id, run_id, summary)
        return direct_fragments, provenance, (_trace_diagnostic("dryml.dispatch.dynamic_trace_provenance_limit_exceeded"),)
    except ValueError:
        diagnostic = _trace_diagnostic("dryml.dispatch.dynamic_trace_evidence_rejected")
        provenance = _rejected_trace_provenance(
            normalized, policy, target, input_id, run_id, True, summary,
            [fact.to_data() for fact in admitted_calls], diagnostic,
        )
        return direct_fragments, provenance, (diagnostic,)
    return fragments, provenance, ()


def _provenance_limit_exceeded(
    normalized: NormalizedDispatchTarget,
    policy: DynamicTracePolicy,
    target: dict[str, str] | None,
    input_id: str,
    run_id: str,
    summary: Mapping[str, Any],
) -> DynamicTraceProvenance:
    """Return the bounded non-truncating carrier for valid oversized evidence."""

    diagnostic = _trace_diagnostic("dryml.dispatch.dynamic_trace_provenance_limit_exceeded")
    return _trace_provenance(
        normalized, policy, target=target, status="provenance_limit_exceeded",
        input_id=input_id, run_id=run_id, started=True, summary=summary,
        diagnostics=[diagnostic],
    )


def _rejected_trace_provenance(
    normalized: NormalizedDispatchTarget,
    policy: DynamicTracePolicy,
    target: dict[str, str] | None,
    input_id: str,
    run_id: str,
    started: bool | None,
    summary: Mapping[str, Any] | None,
    calls: list[dict[str, Any]],
    diagnostic: DiagnosticFact,
) -> DynamicTraceProvenance:
    """Project rejected evidence without letting a projection limit change its status."""

    try:
        return _trace_provenance(
            normalized, policy, target=target, status="evidence_rejected",
            input_id=input_id, run_id=run_id, started=started, summary=summary,
            calls=calls, diagnostics=[diagnostic],
        )
    except _TraceProvenanceLimitError:
        # Rejection takes precedence over projection limits.  Retain a
        # separately validated summary only when the status schema permits it;
        # otherwise omit oversized evidence rather than relabeling
        # stale/malformed input as an ordinary provenance-overflow failure.
        try:
            return _trace_provenance(
                normalized, policy, target=target, status="evidence_rejected",
                input_id=input_id, run_id=run_id, started=started, summary=summary,
                diagnostics=[diagnostic],
            )
        except (_TraceProvenanceLimitError, ValueError):
            pass
    except ValueError:
        pass
    # The valid carrier records only the trusted start-state conclusion when no
    # independently bounded evidence can be retained.
    return _trace_provenance(
        normalized, policy, target=target, status="evidence_rejected",
        input_id=input_id, run_id=run_id, started=True if started or summary is not None or calls else None,
        diagnostics=[diagnostic],
    )


def _trace_result_has_expected_envelope(
    normalized: NormalizedDispatchTarget,
    result: Any,
    expected_target: Mapping[str, Any],
) -> bool:
    """Validate the whole 9B result envelope against this facade invocation."""

    if type(result) is not CodeAnalysisResult or type(result.facts) is not tuple or type(result.diagnostics) is not tuple:
        return False
    live_target = normalized.trace_live_target
    if type(live_target) is not types.FunctionType:
        return False
    try:
        result_data = result.to_data()
    except Exception:
        return False
    if type(result_data) is not dict or set(result_data) != {"target", "facts", "diagnostics", "ok"}:
        return False
    if result_data["target"] != expected_target or result_data["ok"] is not result.ok:
        return False
    return all(type(item) is DiagnosticFact for item in result.diagnostics)


def _safe_trace_result_diagnostic_codes(result: Any) -> tuple[str, ...]:
    """Retain only bounded underlying 9B diagnostic codes for projection data."""

    if type(result) is not CodeAnalysisResult or type(result.diagnostics) is not tuple or not all(type(item) is DiagnosticFact for item in result.diagnostics):
        return ()
    return tuple(item.code for item in result.diagnostics if type(item.code) is str and item.code)


def _validated_trace_result_evidence(
    result: Any,
    *,
    policy: DynamicTracePolicy,
    target: Mapping[str, Any] | None,
) -> tuple[list[DynamicCallFact], dict[str, Any]]:
    """Independently validate every result fact and reject unexpected mixes."""

    if type(result) is not CodeAnalysisResult or type(result.facts) is not tuple:
        raise ValueError("invalid trace result evidence envelope")
    calls: list[DynamicCallFact] = []
    summaries: list[dict[str, Any]] = []
    for original in result.facts:
        if not isinstance(original, CodeFact):
            raise ValueError("invalid trace result fact")
        wire = original.to_data()
        if type(wire) is not dict:
            raise ValueError("invalid trace result wire")
        fact = CodeFact.from_data(wire)
        if isinstance(fact, DynamicCallFact):
            calls.append(fact)
        elif fact.kind == "dynamic_trace_summary":
            summaries.append(wire)
        else:
            raise ValueError("unexpected trace result fact")
    if len(summaries) != 1:
        raise ValueError("trace result requires exactly one summary")
    summary = _independently_validated_trace_summary(result, policy=policy, target=target)
    if summary is None:
        raise ValueError("invalid trace result summary")
    return _validated_trace_calls([fact.to_data() for fact in calls], target=target), summary


def _independently_validated_trace_summary(
    result: Any,
    *,
    policy: DynamicTracePolicy,
    target: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Retain one self-validating summary despite unrelated rejected facts."""

    if type(result) is not CodeAnalysisResult or type(result.facts) is not tuple:
        return None
    summaries = []
    for original in result.facts:
        if not isinstance(original, CodeFact):
            continue
        try:
            wire = original.to_data()
            if type(wire) is dict and wire.get("kind") == "dynamic_trace_summary":
                _validate_trace_summary(wire, policy=_trace_policy_data(policy), target=target)
                summaries.append(wire)
        except Exception:
            continue
    return summaries[0] if len(summaries) == 1 else None


def _independently_validated_trace_calls(
    result: Any,
    *,
    target: Mapping[str, Any] | None,
) -> list[DynamicCallFact]:
    """Return a contiguous bounded call sequence without trusting its summary."""

    if type(result) is not CodeAnalysisResult or type(result.facts) is not tuple:
        raise ValueError("invalid trace result evidence envelope")
    calls: list[DynamicCallFact] = []
    saw_call = False
    for original in result.facts:
        if not isinstance(original, CodeFact):
            continue
        wire = original.to_data()
        if type(wire) is not dict:
            continue
        fact = CodeFact.from_data(wire)
        if isinstance(fact, DynamicCallFact):
            saw_call = True
            calls.append(fact)
    if not saw_call:
        raise ValueError("no independently valid trace calls")
    return _validated_trace_calls([fact.to_data() for fact in calls], target=target)


def _validated_trace_calls(
    values: list[Any],
    *,
    target: Mapping[str, Any] | None = None,
) -> list[DynamicCallFact]:
    """Restore one contiguous call sequence bound to the carrier target."""

    calls: list[DynamicCallFact] = []
    for value in values:
        if type(value) is not dict:
            raise ValueError("invalid dynamic trace call wire")
        fact = CodeFact.from_data(value)
        if not isinstance(fact, DynamicCallFact):
            raise ValueError("non-dynamic call in trace provenance")
        if target is not None and fact.source["target_kind"] != target["target_kind"]:
            raise ValueError("dynamic trace call target mismatch")
        calls.append(fact)
    if [fact.data["sequence"] for fact in calls] != list(range(len(calls))):
        raise ValueError("no independently valid contiguous trace calls")
    return calls


_ANNOTATION_SOURCE_FIELDS = {"kind", "target", "label", "namespace", "path", "metadata"}
_ANNOTATION_TARGET_FIELDS = {"kind", "module", "qualname", "owner_module", "owner_qualname", "metadata"}
_ANNOTATION_FRAGMENT_FIELDS = {"namespace", "kind", "fragment", "source", "priority", "merge_policy", "schema_version"}


def _validate_trace_call_persistence(calls: list[DynamicCallFact]) -> None:
    """Reject full v1 call wires that cannot be persisted unchanged safely.

    Dispatch does not guess whether arbitrary values are secrets. The closed
    safe subset permits no recorded call arguments, no local annotation source
    path, no target metadata, and only the established bounded legacy merge-mode
    source metadata. Environment override mappings must be empty. Evidence
    outside that subset remains transient and cannot enter resolution or a
    persistent carrier.
    """

    for call in calls:
        if call.data["args"] or call.data["kwargs"]:
            raise ValueError("dynamic trace call arguments are not persistence-safe")
        for fact_data in call.data["method_facts"]:
            _validate_trace_method_fact_persistence(fact_data)


def _validate_trace_method_fact_persistence(value: Any) -> None:
    if type(value) is list:
        for item in value:
            _validate_trace_method_fact_persistence(item)
        return
    if type(value) is not dict:
        return
    fields = set(value)
    if fields == _ANNOTATION_SOURCE_FIELDS:
        metadata = value["metadata"]
        if value["path"] is not None or metadata not in (
            {},
            {"legacy_environment_fragment_mode": "base"},
            {"legacy_environment_fragment_mode": "add"},
            {"legacy_environment_fragment_mode": "override"},
        ):
            raise ValueError("dynamic trace annotation source is not persistence-safe")
    elif fields == _ANNOTATION_TARGET_FIELDS and value["metadata"]:
        raise ValueError("dynamic trace annotation target metadata is not persistence-safe")
    elif fields == _ANNOTATION_FRAGMENT_FIELDS and _has_environment_override(value["fragment"]):
        raise ValueError("dynamic trace annotation environment override is not persistence-safe")
    for item in value.values():
        _validate_trace_method_fact_persistence(item)


def _has_environment_override(value: Any) -> bool:
    if type(value) is list:
        return any(_has_environment_override(item) for item in value)
    if type(value) is not dict:
        return False
    if value.get("env") not in (None, {}):
        return True
    return any(_has_environment_override(item) for item in value.values())


def _combine_trace_fragments(
    direct: tuple[annotations.AnnotationFragment, ...],
    calls: list[DynamicCallFact],
) -> tuple[tuple[annotations.AnnotationFragment, ...], list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply only ingress ordering/deduplication before annotation ownership."""

    result = list(direct)
    seen = {json.dumps(fragment.to_data(), sort_keys=True, separators=(",", ":")): "direct" for fragment in direct}
    accepted: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    for call in calls:
        for fact_data in call.data["method_facts"]:
            fact = CodeFact.from_data(fact_data)
            if not isinstance(fact, AnnotationFact):
                continue
            fragment = annotations.AnnotationFragment.from_data(fact.data)
            key = json.dumps(fragment.to_data(), sort_keys=True, separators=(",", ":"))
            observation = {"sequence": call.data["sequence"], "method": call.data["method_name"], "fragment_key": hashlib.sha256(key.encode()).hexdigest()}
            if key in seen:
                duplicates.append({**observation, "first": seen[key]})
                continue
            seen[key] = f"trace:{call.data['sequence']}"
            result.append(fragment)
            accepted.append(observation)
    return tuple(result), accepted, duplicates


def _discover(
    normalized: NormalizedDispatchTarget,
    explicit_environment: Any | None,
    *,
    analysis_context: CodeAnalysisContext,
    probe_timeout_s: float,
):
    diagnostics: list[DiagnosticFact] = []
    analysis: CodeAnalysisResult | None = None
    probe: CodeProbeResult | None = None
    bootstrap_environment: Mapping[str, Any] | None = None
    fragments = []
    complete = False
    if normalized.definition_target is not None and normalized.method_name:
        try:
            fragments.extend(annotations.fragments_for_definition_method(normalized.definition_target, normalized.method_name))
            complete = True
        except Exception as exc:
            diagnostics.append(_diagnostic("dryml.dispatch.definition_method_annotation_collection_failed", "Definition-method annotation collection failed.", data={"error": type(exc).__name__, "method": normalized.method_name}))
    if not complete and normalized.method_name and normalized.subject_class is not None:
        try:
            fragments.extend(annotations.fragments_for_method(normalized.subject_class, normalized.method_name))
            complete = True
        except Exception as exc:
            diagnostics.append(_diagnostic("dryml.dispatch.method_annotation_collection_failed", "Method annotation collection failed.", data={"error": type(exc).__name__, "method": normalized.method_name}))
    elif not complete and normalized.live_annotation_targets:
        for target in normalized.live_annotation_targets:
            try:
                fragments.extend(annotations.fragments_for(target))
                complete = True
            except Exception as exc:
                diagnostics.append(_diagnostic("dryml.dispatch.annotation_collection_failed", "Live annotation collection failed.", data={"error": type(exc).__name__}))
    if normalized.code_target is not None:
        try:
            analysis = analyze(normalized.code_target, context=analysis_context)
            fragments.extend(_fragments_from_analysis(analysis))
            if not complete:
                complete = _analysis_is_complete(analysis)
            diagnostics.extend(analysis.diagnostics)
        except Exception as exc:
            diagnostics.append(_diagnostic("dryml.dispatch.code_analysis_failed", "Local code analysis failed.", data={"error": type(exc).__name__}))
    if not complete and normalized.code_target is not None:
        bootstrap = _bootstrap_environment(explicit_environment)
        bootstrap_environment = bootstrap.to_data()
        probe = probe_target(normalized.code_target, environment=bootstrap, include_environment_record=True, timeout=probe_timeout_s)
        diagnostics.extend(probe.diagnostics)
        if probe.analysis is not None:
            fragments.extend(_fragments_from_analysis(probe.analysis))
        complete = probe.ok and probe.analysis is not None
    return _dedupe_fragments(fragments), analysis, probe, bootstrap_environment, tuple(diagnostics), complete


def _analysis_options(policy: Any | None) -> tuple[CodeAnalysisContext, float]:
    """Validate bounded dispatch analysis/probe options before discovery starts."""

    request = parse_analysis_policy(policy)
    return request.context, request.probe_timeout_s


def _validate_sprint8_policies(inventory_policy: str, resolver_policy: str | None) -> None:
    """Reject advanced-policy typos before analysis, probes, or persistence."""

    if inventory_policy not in {"lightweight", "external"}:
        raise DispatchPlanningError("invalid inventory_policy", context={"inventory_policy": inventory_policy})
    if resolver_policy is not None and resolver_policy != "first_compatible":
        raise DispatchPlanningError("invalid resolver_policy", context={"resolver_policy": resolver_policy})


def _analysis_is_complete(analysis: CodeAnalysisResult) -> bool:
    if any(item.severity == "error" for item in analysis.diagnostics):
        return False
    incomplete_codes = {"dryml.code.algorithm_not_applicable", "dryml.code.annotations_unsupported_target"}
    return not any(item.code in incomplete_codes for item in analysis.diagnostics)


def _fragments_from_analysis(analysis: CodeAnalysisResult) -> list[annotations.AnnotationFragment]:
    result = []
    for fact in analysis.facts:
        if isinstance(fact, AnnotationFact):
            try:
                result.append(annotations.AnnotationFragment.from_data(fact.data))
            except Exception:
                continue
    return result


def _dedupe_fragments(fragments):
    result = []
    seen = set()
    for fragment in fragments:
        key = json.dumps(fragment.to_data(), sort_keys=True, separators=(",", ":"))
        if key not in seen:
            seen.add(key)
            result.append(fragment)
    return tuple(result)


def _select_environment(
    explicit: Any | None,
    annotation_default: Any | None,
    *,
    requirement=None,
    candidates=None,
    registry=None,
    resolver_policy=None,
    bootstrap_probe: CodeProbeResult | None = None,
    bootstrap_environment: Mapping[str, Any] | None = None,
):
    current = environments.current(default=None)
    if explicit is not None or annotation_default is not None or current is not None:
        selection, data = _select("environment", (("explicit", explicit), ("annotation_default", annotation_default), ("current", current), ("fallback", environments.CurrentEnvironmentSpec())), _environment_data)
        return selection, data, None
    needs_resolver = candidates is not None or registry is not None or requirement is not None
    if needs_resolver:
        def probe_runner(spec, *, timeout):
            if (
                bootstrap_probe is not None
                and bootstrap_probe.ok
                and bootstrap_probe.environment_record is not None
                and bootstrap_environment is not None
                and spec.to_data() == bootstrap_environment
            ):
                return environments.EnvironmentProbeResult(spec=spec, ok=True, record=bootstrap_probe.environment_record)
            return environments.probe(spec, timeout=timeout)

        result = environments.resolve(
            requirement,
            candidates=() if candidates is None else candidates,
            registry=registry,
            include_current=True,
            policy="first_compatible" if resolver_policy is None else resolver_policy,
            probe_runner=probe_runner,
        )
        considered = tuple(CandidateConsideration(slot, "absent") for slot in ("explicit", "annotation_default", "current"))
        if result.selected is not None:
            data = result.selected.to_data()
            return CandidateSelection("environment", data, "resolver", considered + (CandidateConsideration("resolver", "selected", data),)), data, result
        fallback = environments.CurrentEnvironmentSpec().to_data()
        return CandidateSelection("environment", fallback, "fallback", considered + (CandidateConsideration("resolver", result.status), CandidateConsideration("fallback", "selected", fallback))), fallback, result
    selection, data = _select("environment", (("fallback", environments.CurrentEnvironmentSpec()),), _environment_data)
    return selection, data, None


def _select_world(
    explicit: Any | None,
    annotation_default: Any | None,
    *,
    requirement=None,
    inventory=None,
    inventory_policy="lightweight",
    inventory_discovery_error: str | None = None,
):
    current = worlds.current(default=None)
    if explicit is not None or annotation_default is not None or current is not None:
        selection, data = _select("world", (("explicit", explicit), ("annotation_default", annotation_default), ("current", current), ("fallback", {"roles": {"main": {"replicas": 1, "process": {}}}, "backend": {"kind": "local", "parameters": {}}})), _world_data)
        return selection, data, None
    if requirement is not None:
        result = worlds.synthesize(
            requirement,
            inventory=inventory,
            policy="local",
            inventory_policy=inventory_policy,
            _inventory_discovery_error=inventory_discovery_error,
        )
        considered = tuple(CandidateConsideration(slot, "absent") for slot in ("explicit", "annotation_default", "current"))
        if result.world is not None:
            data = result.world.to_data()
            return CandidateSelection("world", data, "synthesized", considered + (CandidateConsideration("synthesized", "selected", data),)), data, result
        fallback = {"roles": {"main": {"replicas": 1, "process": {}}}, "backend": {"kind": "local", "parameters": {}}}
        return CandidateSelection("world", fallback, "fallback", considered + (CandidateConsideration("synthesized", "failed"), CandidateConsideration("fallback", "selected", fallback))), fallback, result
    fallback = {"roles": {"main": {"replicas": 1, "process": {}}}, "backend": {"kind": "local", "parameters": {}}}
    selection, data = _select("world", (("fallback", fallback),), _world_data)
    return selection, data, None


def _select_runtime(explicit: Any | None, annotation_default: Any | None):
    fallback = RuntimeContextSpec(mode=RuntimeMode.WORKER, device_visibility={"policy": "assigned"}, metadata={"source": "dryml.dispatch"})
    return _select("runtime", (("explicit", explicit), ("annotation_default", annotation_default), ("fallback", fallback)), _runtime_data)


def _select(kind: str, choices, serializer):
    considered = []
    selected = None
    source = None
    for slot, value in choices:
        if value is None:
            considered.append(CandidateConsideration(slot, "absent"))
            continue
        try:
            data = serializer(value)
        except Exception as exc:
            raise DispatchPlanningError(f"invalid {kind} candidate", context={"source": slot, "error": str(exc)}) from exc
        if selected is None:
            selected, source = data, slot
            considered.append(CandidateConsideration(slot, "selected", data))
        else:
            considered.append(CandidateConsideration(slot, "not_selected", data))
    assert selected is not None and source is not None
    return CandidateSelection(kind, selected, source, tuple(considered)), selected


def _environment_data(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping) and "spec" in value:
        value = value["spec"]
    if isinstance(value, str):
        return PythonExecutableSpec(value).to_data()
    if isinstance(value, Mapping):
        return spec_from_data(value).to_data()
    if isinstance(value, EnvironmentSpec):
        return value.to_data()
    if hasattr(value, "to_data"):
        return value.to_data()
    raise TypeError(type(value).__name__)


def _world_data(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping) and "spec" in value:
        value = value["spec"]
    if isinstance(value, Mapping) and value.get("schema") == "dryml.world.v1":
        worlds.validate_world_spec(value)
        value = value["payload"]
    if isinstance(value, WorldSpec):
        return value.to_data()
    # Sprint 6 accepted opaque local-backend world policy mappings. Preserve
    # them for no-requirement compatibility; a hard requirement will report
    # their non-checkable shape instead of inventing a replacement world.
    if isinstance(value, Mapping) and "roles" not in value:
        return dict(value)
    return WorldSpec.from_data(value).to_data()


def _runtime_data(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping) and "spec" in value:
        value = value["spec"]
    data = value.to_data() if isinstance(value, RuntimeContextSpec) else dict(value)
    # A dispatch worker always receives a worker runtime unless explicitly set.
    data.setdefault("mode", RuntimeMode.WORKER.value)
    if data.get("mode") == RuntimeMode.ORCHESTRATOR.value:
        data["mode"] = RuntimeMode.WORKER.value
    return RuntimeContextSpec.from_data(data).to_data()


def _bootstrap_environment(explicit: Any | None) -> EnvironmentSpec:
    if explicit is not None:
        return spec_from_data(_environment_data(explicit))
    current = environments.current(default=None)
    return spec_from_data(_environment_data(current)) if current is not None else environments.CurrentEnvironmentSpec()


def _same_python_environment(env_data: Mapping[str, Any]) -> bool:
    if env_data.get("kind") == "current":
        return True
    if env_data.get("kind") == "python":
        return os.path.abspath(os.fspath(env_data.get("executable", ""))) == os.path.abspath(sys.executable)
    return False


def _needs_final_probe(normalized: NormalizedDispatchTarget, final_environment: Mapping[str, Any], bootstrap_environment: Mapping[str, Any] | None) -> bool:
    if normalized.code_target is None or not normalized.code_target.import_path or normalized.transport == "pickle_small":
        return False
    # Local discovery proves only the orchestrator environment. A non-current
    # selected candidate must prove it can import/analyze the serialized target.
    return bootstrap_environment != final_environment and final_environment.get("kind") != "current"


def _requires_environment_validation(env_data: Mapping[str, Any], normalized: NormalizedDispatchTarget) -> bool:
    del normalized
    return env_data.get("kind") != "current"


def _is_multi_worker_world(candidate: Mapping[str, Any]) -> bool:
    try:
        world = WorldSpec.from_data(candidate)
    except Exception:
        return False
    return len(world.roles) > 1 or sum(role.replicas for role in world.roles.values()) > 1


def _world_needs_inventory(candidate: Mapping[str, Any]) -> bool:
    try:
        world = WorldSpec.from_data(candidate)
    except Exception:
        return False
    return any(
        role.process.resources.cpus > 1
        or role.process.resources.memory is not None
        or bool(role.process.resources.accelerators)
        for role in world.roles.values()
    )


def _subprocess_allocation_world(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt the supported local_subprocess request backend for local assignment."""

    world = WorldSpec.from_data(candidate)
    if world.backend.get("kind") != "local_subprocess":
        return dict(candidate)
    data = world.to_data()
    data["backend"] = {"kind": "local", "parameters": {}}
    return data


def _local_subprocess_allocation_summary(allocation_plan) -> dict[str, Any]:
    workers = []
    for key in allocation_plan.worker_keys:
        view = allocation_plan.world_allocation.runtime_view(key.role, key.replica, world_allocation_id=allocation_plan.world_allocation_spec["id"])
        workers.append({
            "role": key.role,
            "replica": key.replica,
            "cpus": list(view.cpus),
            "memory": view.memory,
            "accelerators": {name: list(values) for name, values in sorted(view.accelerators.items())},
        })
    return {"backend": "local_subprocess", "workers": workers}


def _effective_local_allocation_requirement_check(candidate: Mapping[str, Any], requirement):
    """Check executor-normalized CPU counts without creating an allocation spec."""

    if requirement is None:
        return None
    world = WorldSpec.from_data(candidate)
    data = world.to_data()
    for role in data["roles"].values():
        resources = role["process"].setdefault("resources", {})
        resources["cpus"] = resources.get("cpus") or 1
    return worlds.check_world_spec_satisfies_requirement(WorldSpec.from_data(data), requirement)


def _world_issue_data(issue) -> dict[str, Any]:
    return {"severity": issue.severity, "path": issue.path, "message": issue.message, "expected": issue.expected, "actual": issue.actual}


def _local_subprocess_world_diagnostics(
    candidate: Mapping[str, Any],
    requirement,
    policy: RequirementPolicy,
) -> tuple[DiagnosticFact, ...]:
    """Reject selected-world details the one-worker local allocator cannot enact."""

    try:
        world = WorldSpec.from_data(candidate)
    except Exception as exc:
        return (_diagnostic("dryml.dispatch.single_subprocess_world_unsupported", "The local subprocess planner requires a concrete local one-worker WorldSpec.", data={"error": type(exc).__name__}),)
    diagnostics = []
    if len(world.roles) != 1 or sum(role.replicas for role in world.roles.values()) != 1:
        diagnostics.append(_diagnostic("dryml.dispatch.single_subprocess_world_unsupported", "The local subprocess planner supports one worker only; use plan_world() or run_world() for this world.", data={"world": candidate}))
    backend_kind = world.backend.get("kind")
    if backend_kind not in {"local", "local_subprocess"}:
        diagnostics.append(_diagnostic("dryml.dispatch.single_subprocess_backend_unsupported", "The local subprocess planner cannot enact the selected world backend.", data={"backend": dict(world.backend)}))
    elif not isinstance(world.backend.get("parameters", {}), Mapping) or world.backend.get("parameters"):
        diagnostics.append(_diagnostic("dryml.dispatch.single_subprocess_backend_parameters_unsupported", "The local subprocess planner cannot enact requested world backend parameters.", data={"backend": dict(world.backend)}))
    for role_name, role in world.roles.items():
        process = role.process
        resources = process.resources
        if _has_positive_unsupported_resource(resources.devices) or _has_positive_unsupported_resource(resources.named):
            diagnostics.append(_diagnostic("dryml.dispatch.single_subprocess_resources_unsupported", "The local subprocess planner cannot allocate selected named devices or resources.", data={"role": role_name, "resources": resources.to_data()}))
        if process.environment is not None or process.runtime is not None or process.metadata:
            diagnostics.append(_diagnostic("dryml.dispatch.single_subprocess_process_settings_unsupported", "The local subprocess planner cannot enact selected role process settings.", data={"role": role_name, "process": process.to_data()}))
    if requirement is not None:
        for role_name, role_requirement in requirement.roles.items():
            for topology_name in ("collectives", "shared_filesystem"):
                if role_requirement.topology.get(topology_name) not in (None, False):
                    diagnostics.append(_diagnostic(
                        "dryml.dispatch.single_subprocess_topology_unsupported",
                        "The local subprocess planner cannot enforce the requested world topology.",
                        data={"role": role_name, "topology": topology_name},
                    ))
        report = worlds.check_world_spec_satisfies_requirement(world, requirement)
        if not report.ok:
            diagnostics.append(_diagnostic(
                "dryml.dispatch.single_subprocess_requirement_unsupported",
                "The selected local subprocess world does not satisfy the hard world requirement.",
                severity="error" if policy is RequirementPolicy.STRICT else "warning",
                data={"issues": [{"path": item.path, "message": item.message, "expected": item.expected, "actual": item.actual} for item in report.issues]},
            ))
    return tuple(diagnostics)


def _local_world_topology_diagnostics(requirement) -> tuple[DiagnosticFact, ...]:
    """Reject topology guarantees the same-host local-world backend cannot enact."""

    if requirement is None:
        return ()
    diagnostics = []
    for role_name, role_requirement in requirement.roles.items():
        for topology_name in ("collectives", "shared_filesystem"):
            if role_requirement.topology.get(topology_name) not in (None, False):
                diagnostics.append(_diagnostic(
                    "dryml.dispatch.local_world_topology_unsupported",
                    "The local-world planner cannot enforce the requested world topology.",
                    data={"role": role_name, "topology": topology_name},
                ))
    return tuple(diagnostics)


def _has_positive_unsupported_resource(values: Mapping[str, Any]) -> bool:
    """Return whether a concrete unsupported resource requests backend work."""

    for value in values.values():
        if isinstance(value, Mapping):
            if _has_positive_unsupported_resource(value):
                return True
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            if value > 0:
                return True
        elif value:
            return True
    return False


def _environment_record(env_data, requirement, code_probe, probe_environment, explicit_environment, policy, *, validate_candidate: bool, resolved_record=None, resolved_probe=None, resolver_incomplete: bool = False):
    if resolver_incomplete:
        return None, ()
    if requirement is None and not validate_candidate:
        return None, ()
    spec = spec_from_data(env_data)
    launch_error = _environment_launch_error(env_data)
    if launch_error is not None:
        return None, (launch_error,)
    attached = _attached_environment_record(explicit_environment, env_data)
    if attached is not None:
        return attached, ()
    if resolved_record is not None:
        return resolved_record, ()
    if resolved_probe is not None:
        return None, (_diagnostic(
            "dryml.dispatch.environment_probe_failed",
            "Environment probe already failed for the selected candidate during resolution.",
            severity="error",
            data={"candidate": env_data, "probe": _bounded_probe_data(resolved_probe.to_data())},
        ),)
    if code_probe is not None and code_probe.environment_record is not None and probe_environment == env_data:
        return code_probe.environment_record, ()
    probe = environments.probe(spec)
    if probe.ok and probe.record is not None:
        return probe.record, ()
    return None, (_diagnostic("dryml.dispatch.environment_probe_failed", "Environment probe failed for the selected candidate.", severity="error", data={"candidate": env_data, "probe": _bounded_probe_data(probe.to_data())}),)


def _resolution_record_for(result, candidate: Mapping[str, Any]) -> EnvironmentRecord | None:
    if result is None:
        return None
    if result.selected_record is not None and result.selected is not None and result.selected.to_data() == candidate:
        return result.selected_record
    if result.fallback_record is not None and result.fallback_spec is not None and result.fallback_spec.to_data() == candidate:
        return result.fallback_record
    for attempt in result.attempts:
        if (
            attempt.status in {"selected", "incompatible"}
            and attempt.probe is not None
            and attempt.probe.ok
            and attempt.probe.record is not None
            and attempt.spec.to_data() == candidate
        ):
            return attempt.probe.record
    return None


def _resolution_probe_for(result, candidate: Mapping[str, Any]):
    """Return matching failed resolver evidence to avoid a fallback reprobe."""

    if result is None:
        return None
    if result.fallback_probe is not None and result.fallback_spec is not None and result.fallback_spec.to_data() == candidate:
        return result.fallback_probe
    for attempt in result.attempts:
        if attempt.probe is not None and attempt.spec.to_data() == candidate:
            return attempt.probe
    return None


def _environment_launch_error(env_data: Mapping[str, Any]) -> DiagnosticFact | None:
    """Validate local worker command construction without launching a worker."""

    try:
        from .backends import build_worker_command

        build_worker_command(env_data)
    except Exception as exc:
        return _diagnostic(
            "dryml.dispatch.environment_launch_unsupported",
            "The selected environment cannot be launched by the local subprocess backend.",
            data={"candidate": env_data, "error": type(exc).__name__},
        )
    return None


def _attached_environment_record(candidate, selected_data):
    if not isinstance(candidate, Mapping) or "record" not in candidate:
        return None
    spec_value = candidate.get("spec", candidate)
    try:
        if _environment_data(spec_value) != selected_data:
            return None
        record = candidate["record"]
        return record if isinstance(record, EnvironmentRecord) else EnvironmentRecord.from_data(record)
    except Exception:
        return None


def _check_environment(requirement, candidate, record, policy, probe_diagnostics, *, validate_candidate: bool):
    requirement_data = None if requirement is None else requirement.to_data()
    if requirement is None and record is None and not validate_candidate:
        return CandidateCheckReport("environment", "not_required", None, None, candidate)
    if requirement is None and record is None:
        return CandidateCheckReport("environment", "error", False, None, candidate, ({"reason": "environment_record_unavailable"},), probe_diagnostics)
    if requirement is None:
        return CandidateCheckReport("environment", "satisfied", True, None, candidate)
    if policy is RequirementPolicy.IGNORE:
        return CandidateCheckReport("environment", "skipped", None, requirement_data, candidate, ({"reason": "requirement_policy_ignore"},))
    if record is None:
        return CandidateCheckReport("environment", "error", False, requirement_data, candidate, ({"reason": "environment_record_unavailable"},), probe_diagnostics)
    report = requirement.check(record, policy="strict" if policy is RequirementPolicy.STRICT else "warn")
    details = tuple(report.to_data().get("issues") or ())
    return CandidateCheckReport("environment", "satisfied" if report.ok else "incompatible", report.ok, requirement_data, candidate, details)


def _check_world(requirement, candidate, policy):
    requirement_data = None if requirement is None else requirement.to_data()
    if requirement is None:
        return CandidateCheckReport("world", "not_required", None, None, candidate)
    if policy is RequirementPolicy.IGNORE:
        return CandidateCheckReport("world", "skipped", None, requirement_data, candidate, ({"reason": "requirement_policy_ignore"},))
    try:
        world = WorldSpec.from_data(candidate)
    except Exception as exc:
        detail = {"path": "world", "message": "selected world cannot be checked against a hard requirement", "expected": requirement_data, "actual": candidate, "error": type(exc).__name__}
        return CandidateCheckReport("world", "error", False, requirement_data, candidate, (detail,))
    report = worlds.check_world_spec_satisfies_requirement(world, requirement)
    details = tuple({"severity": issue.severity, "path": issue.path, "message": issue.message, "expected": issue.expected, "actual": issue.actual, "source": issue.source} for issue in report.issues)
    return CandidateCheckReport("world", "satisfied" if report.ok else "incompatible", report.ok, requirement_data, candidate, details)


def _check_runtime(requirement, candidate, policy):
    if requirement is None:
        return CandidateCheckReport("runtime", "not_required", None, None, candidate)
    if policy is RequirementPolicy.IGNORE:
        return CandidateCheckReport("runtime", "skipped", None, dict(requirement), candidate, ({"reason": "requirement_policy_ignore"},))
    report = runtime.check_runtime_spec_satisfies_requirement(candidate, requirement)
    details = tuple(item.to_data() for item in report.issues)
    return CandidateCheckReport("runtime", "satisfied" if report.ok else "incompatible", report.ok, dict(requirement), candidate, details)


def _annotation_diagnostics(resolution):
    return tuple(_diagnostic(item.code, item.message, severity=item.level, data=item.data) for item in resolution.diagnostics)


def _has_annotation_errors(resolution) -> bool:
    return any(item.level == "error" for item in resolution.diagnostics)


def _resolution_decisions(resolution) -> dict[str, Any]:
    """Return only requirement/default values whose change invalidates selection."""

    return {
        "environment_requirement": None if resolution.environment_requirement is None else resolution.environment_requirement.to_data(),
        "environment_default": None if resolution.environment_default is None else _environment_data(resolution.environment_default),
        "world_requirement": None if resolution.world_requirement is None else resolution.world_requirement.to_data(),
        "world_default": None if resolution.world_default is None else _world_data(resolution.world_default),
        "runtime_requirement": None if resolution.runtime_requirement is None else dict(resolution.runtime_requirement),
        "runtime_default": None if resolution.runtime_default is None else _runtime_data(resolution.runtime_default),
    }


def _check_diagnostics(reports, policy, selections):
    diagnostics = []
    for report, selection in zip(reports, selections, strict=True):
        if report.status in {"not_required", "skipped", "satisfied"}:
            continue
        severity = "error" if policy is RequirementPolicy.STRICT else "warning"
        details = report.details or ({"reason": report.status},)
        action = {
            "explicit": f"replace the explicit {report.kind} override",
            "annotation_default": f"replace or remove the annotation-default {report.kind}",
            "current": f"set or clear the current {report.kind} override",
        }.get(selection.source)
        diagnostics.append(_diagnostic(
            f"dryml.dispatch.{report.kind}_check_{report.status}",
            f"Selected {selection.source} {report.kind} candidate does not satisfy dispatch requirements.",
            severity=severity,
            data={"check": report.to_data(), "details": details, "policy": policy.value, "action": action},
        ))
    return diagnostics


def _diagnostic(code, message, *, severity="error", data=None):
    return DiagnosticFact(severity=severity, code=code, message=message, source={"component": "dryml.dispatch.requirements"}, data=data or {})


def _normalized_target_data(normalized):
    return {
        "operation_id": normalized.operation_spec.get("id"),
        "operation_kind": normalized.operation_spec.get("kind"),
        "code_target": None if normalized.code_target is None else normalized.code_target.to_data(),
        "method_name": normalized.method_name,
        "transport": normalized.transport,
        "restrictions": list(normalized.launch.get("transport_restrictions") or ()),
    }


def _bounded_probe_data(data):
    if isinstance(data, Mapping):
        data = {key: value for key, value in data.items() if key not in {"stdout", "stderr"}}
    return _bounded_data(data)


def _probe_summary(probe):
    if probe is None:
        return None
    # Imported modules can print credentials. Captured output stays transient and
    # is never copied into persisted planning metadata.
    return _bounded_probe_data({
        "kind": "dryml.code_probe_result",
        "schema_version": 1,
        "ok": probe.ok,
        "analysis": _analysis_summary(probe.analysis),
        "environment_record": _environment_record_summary(probe.environment_record),
        "diagnostics": [item.to_data() for item in probe.diagnostics],
    })


def _analysis_summary(analysis):
    return None if analysis is None else _bounded_data(analysis.to_data())


def _environment_record_summary(record):
    if record is None:
        return None
    return {
        "id": record.id,
        "python": {"implementation": record.python.implementation, "version": record.python.version},
        "platform": {"system": record.platform.system, "machine": record.platform.machine},
    }


def _bounded_data(value, *, depth=0, budget=None):
    budget = [_MAX_METADATA_NODES] if budget is None else budget
    if budget[0] <= 0 or depth > _MAX_METADATA_DEPTH:
        return {"__dryml_truncated__": "depth_or_size"}
    budget[0] -= 1
    if isinstance(value, str):
        return value[:_MAX_METADATA_STRING]
    if isinstance(value, float):
        return value if value == value and value not in {float("inf"), float("-inf")} else None
    if isinstance(value, Mapping):
        items = []
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            if str(key) == "env":
                continue
            items.append((str(key)[:_MAX_METADATA_STRING], _bounded_data(item, depth=depth + 1, budget=budget)))
            if len(items) >= _MAX_METADATA_ITEMS:
                break
        return dict(items)
    if isinstance(value, (list, tuple)):
        return [_bounded_data(item, depth=depth + 1, budget=budget) for item in value[:_MAX_METADATA_ITEMS]]
    if value is None or isinstance(value, (bool, int)):
        return value
    return str(value)[:_MAX_METADATA_STRING]


def _safe_candidate_data(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Redact runtime environment overrides from persisted planning metadata."""

    data = _bounded_data(dict(candidate))
    if data.get("kind") in {"python", "conda", "container"}:
        data.pop("env", None)
    return data


__all__ = [
    "PLANNING_METADATA_VERSION",
    "CandidateCheckReport",
    "CandidateConsideration",
    "CandidateSelection",
    "DynamicTraceProvenance",
    "DispatchExplanation",
    "DispatchPlanningResolution",
    "RequirementPolicy",
    "effective_requirement_policy",
    "explanation_for",
    "parse_analysis_policy",
    "resolve_dispatch_plan",
]
