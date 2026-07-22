from __future__ import annotations

import json

from dryml.annotations import collect_fragments, fragments_for_method, resolve_fragments
from dryml.code.analysis import CodeAnalysisContext, CodeAnalysisResult, FunctionAnalyzer
from dryml.code.facts import AnnotationFact, DiagnosticFact, RequirementFact
from dryml.code.targets import CodeTarget


def analyze_target(target: CodeTarget, context: CodeAnalysisContext) -> CodeAnalysisResult:
    """Collect raw annotation fragments from a live target as facts.

    The analyzer delegates collection and merge semantics to
    :mod:`dryml.annotations`. It emits facts and diagnostics only; it does not
    select environments, worlds, runtimes, or dispatch candidates.
    """

    if not context.include_annotations:
        return CodeAnalysisResult(target=target.spec)
    obj = target.obj
    if obj is None:
        return CodeAnalysisResult(target=target.spec, diagnostics=(DiagnosticFact(
            severity="info",
            code="dryml.code.algorithm_not_applicable",
            message="Direct annotation analysis requires a live target.",
            source={"analyzer": "direct_annotations", "target_kind": target.spec.kind},
        ),))

    try:
        fragments = _collect_target_fragments(target)
    except Exception as exc:
        return CodeAnalysisResult(target=target.spec, diagnostics=(DiagnosticFact(
            severity="error",
            code="dryml.code.annotations_collection_failed",
            message="Direct annotation analysis failed while collecting fragments.",
            source={"analyzer": "direct_annotations", "target_kind": target.spec.kind},
            data={"error": repr(exc)},
        ),))

    if not fragments and target.spec.kind in {"unknown", "callable_instance"}:
        return CodeAnalysisResult(target=target.spec, diagnostics=(DiagnosticFact(
            severity="info",
            code="dryml.code.annotations_unsupported_target",
            message="Direct annotation analysis found no annotation fragments on this target kind.",
            source={"analyzer": "direct_annotations", "target_kind": target.spec.kind},
        ),))

    resolution = resolve_fragments(fragments)
    resolution_data = resolution.to_data()
    trace_by_index = {trace.fragment_index: trace.to_data() for trace in resolution.source_traces}

    facts = []
    seen_fragments: set[str] = set()
    for index, fragment in enumerate(resolution.fragments):
        fragment_data = fragment.to_data()
        key = json.dumps(fragment_data, sort_keys=True, separators=(",", ":"))
        if key in seen_fragments:
            continue
        seen_fragments.add(key)
        source = {
            "analyzer": "direct_annotations",
            "target_kind": target.spec.kind,
            "annotation_source": fragment_data.get("source"),
        }
        facts.append(AnnotationFact(source=source, data=fragment_data))
        if fragment.kind in ("requirement", "default"):
            facts.append(RequirementFact(
                namespace=fragment.namespace,
                requirement_kind=fragment.kind,
                fragment=fragment.fragment,
                priority=fragment.priority,
                merge_policy=fragment.merge_policy,
                source=source,
                data={"annotation": fragment_data, "source_trace": trace_by_index.get(index), "resolution": resolution_data},
            ))
    diagnostics = tuple(DiagnosticFact(
        severity=diagnostic.level,
        code=diagnostic.code,
        message=diagnostic.message,
        source={"analyzer": "direct_annotations", "target_kind": target.spec.kind},
        data=diagnostic.to_data(),
    ) for diagnostic in resolution.diagnostics)
    return CodeAnalysisResult(target=target.spec, facts=tuple(facts), diagnostics=diagnostics)


def _collect_target_fragments(target: CodeTarget):
    if target.owner is not None and target.attribute_name is not None:
        return fragments_for_method(target.owner, target.attribute_name)
    return collect_fragments(target.obj)


def can_analyze(target: CodeTarget, context: CodeAnalysisContext) -> bool:
    """Return true when annotation collection is enabled."""

    return context.include_annotations


ANALYZER = FunctionAnalyzer("direct_annotations", analyze_target, can_analyze)


__all__ = ["ANALYZER", "analyze_target"]
