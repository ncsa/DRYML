from __future__ import annotations

import json

from dryml.annotations.collect import fragments_for
from dryml.code.analysis import CodeAnalysisContext, CodeAnalysisResult, FunctionAnalyzer
from dryml.code.facts import AnnotationFact, DiagnosticFact, RequirementFact
from dryml.code.targets import CodeTarget


def analyze_target(target: CodeTarget, context: CodeAnalysisContext) -> CodeAnalysisResult:
    """Collect raw annotation fragments from a live target as facts."""

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

    fragments = list(fragments_for(obj))
    if target.raw_descriptor is not None and target.raw_descriptor is not obj:
        fragments.extend(fragments_for(target.raw_descriptor))

    facts = []
    seen_fragments: set[str] = set()
    for fragment in fragments:
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
                data={"annotation": fragment_data},
            ))
    return CodeAnalysisResult(target=target.spec, facts=tuple(facts))


def can_analyze(target: CodeTarget, context: CodeAnalysisContext) -> bool:
    """Return true when annotation collection is enabled."""

    return context.include_annotations


ANALYZER = FunctionAnalyzer("direct_annotations", analyze_target, can_analyze)


__all__ = ["ANALYZER", "analyze_target"]
