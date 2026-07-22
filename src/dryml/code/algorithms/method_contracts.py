from __future__ import annotations

from dryml.code.analysis import CodeAnalysisContext, CodeAnalysisResult, FunctionAnalyzer
from dryml.code.facts import MethodContractFact
from dryml.code.targets import CodeTarget
from dryml.core2.backend import Backend
from dryml.core2.methods import BatchMode, Method, Traits


def _trait_selector_data(selector: Traits) -> dict[str, str | None]:
    """Return the fixed typed wire form for a core2 trait selector.

    Reading the declared slots avoids invoking a custom representation hook.
    Malformed selector metadata fails analysis rather than entering a fact.
    """

    if not isinstance(selector, Traits):
        raise TypeError("method trait selector must be a Traits instance")
    backend = Traits.__dict__["backend"].__get__(selector, Traits)
    batch_mode = Traits.__dict__["batch_mode"].__get__(selector, Traits)
    if backend is not None and type(backend) is not Backend:
        raise TypeError("method trait backend must be a Backend or None")
    if batch_mode is not None and type(batch_mode) is not BatchMode:
        raise TypeError("method trait batch mode must be a BatchMode or None")
    return {
        "backend": None if backend is None else backend.value,
        "batch_mode": None if batch_mode is None else batch_mode.value,
    }


def analyze_target(target: CodeTarget, context: CodeAnalysisContext) -> CodeAnalysisResult:
    """Emit minimal DRYML Method contract facts from core2 semantics."""

    if not context.include_method_contracts:
        return CodeAnalysisResult(target=target.spec)
    obj = target.obj
    cls = obj if isinstance(obj, type) else type(obj) if isinstance(obj, Method) else None
    if cls is None or not issubclass(cls, Method):
        return CodeAnalysisResult(target=target.spec)

    trait_impls = []
    for traits, name in getattr(cls, "__trait_impls__", ()):
        trait_impls.append({"name": name, "traits": _trait_selector_data(traits)})
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
