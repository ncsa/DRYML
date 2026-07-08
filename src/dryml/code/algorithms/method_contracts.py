from __future__ import annotations

from dryml.code.analysis import CodeAnalysisContext, CodeAnalysisResult, FunctionAnalyzer
from dryml.code.facts import MethodContractFact
from dryml.code.targets import CodeTarget
from dryml.core2.methods import Method


def analyze_target(target: CodeTarget, context: CodeAnalysisContext) -> CodeAnalysisResult:
    """Emit minimal DRYML Method contract facts without moving Method."""

    if not context.include_method_contracts:
        return CodeAnalysisResult(target=target.spec)
    obj = target.obj
    cls = obj if isinstance(obj, type) else type(obj) if isinstance(obj, Method) else None
    if cls is None or not issubclass(cls, Method):
        return CodeAnalysisResult(target=target.spec)

    trait_impls = []
    for traits, name in getattr(cls, "__trait_impls__", ()):
        trait_impls.append({"name": name, "traits": repr(traits)})
    return CodeAnalysisResult(target=target.spec, facts=(MethodContractFact(
        source={"analyzer": "method_contracts", "target_kind": target.spec.kind},
        data={
            "method_contract_detected": True,
            "class_module": getattr(cls, "__module__", None),
            "class_qualname": getattr(cls, "__qualname__", None),
            "trait_impls": trait_impls,
            "has_user_call": "__call__" in getattr(cls, "__dict__", {}),
        },
    ),))


def can_analyze(target: CodeTarget, context: CodeAnalysisContext) -> bool:
    """Return true when method-contract analysis is enabled."""

    return context.include_method_contracts


ANALYZER = FunctionAnalyzer("method_contracts", analyze_target, can_analyze)


__all__ = ["ANALYZER", "analyze_target"]
