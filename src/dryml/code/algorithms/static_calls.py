"""Opt-in conservative static call resolution.

Only safely importable plain Python or builtin functions, ordinary classes in a
target's real globals mapping, and direct methods on direct parameters annotated
with ordinary concrete classes can resolve. Aliases, closures, protocols,
attribute chains, call-result receivers, properties, dynamic lookup, callable
instances, bound builtin methods, lambda identities, and non-standard metaclasses
remain non-resolved.
Every fact is a source-level possibility, never evidence of runtime execution.
"""

from __future__ import annotations

import ast
import inspect
import sys
import types
from collections.abc import Mapping
from typing import Any

from dryml.code.analysis import CodeAnalysisContext, CodeAnalysisResult, FunctionAnalyzer
from dryml.code.facts import CodeFact, DiagnosticFact, StaticCallFact
from dryml.code.targets import CodeTarget

from .source import get_source_info
from .static_analysis import (
    MAX_AST_NODES,
    MAX_CALL_SITES,
    MAX_CHAIN_COMPONENTS,
    MAX_RESOLUTION_DIAGNOSTICS,
    MAX_SOURCE_BYTES,
    MAX_STATIC_SCALAR_CHARS,
    STATIC_ANALYSIS_LIMITS,
    absolute_line,
    bounded_string,
    bounded_target_mapping,
    limit_diagnostic,
    parse_static_source,
)


class _CallCollector(ast.NodeVisitor):
    """Collect call expressions in deterministic source traversal order."""

    def __init__(self) -> None:
        self.calls: list[tuple[ast.Call, bool]] = []
        self.call_sites_seen = 0
        self.exhausted = False
        self._nested_scope_depth = 0

    def collect_root(self, root: ast.AST) -> None:
        """Collect every call in the selected definition in source order."""

        if not isinstance(root, (ast.FunctionDef, ast.AsyncFunctionDef)):
            self.visit(root)
            self.calls.sort(key=lambda item: (item[0].lineno, item[0].col_offset))
            return
        # Decorators, signature expressions, and the body are all source-level
        # call sites of the selected definition. Nested scopes stay unsupported.
        for decorator in root.decorator_list:
            self.visit(decorator)
        for default in (*root.args.defaults, *(item for item in root.args.kw_defaults if item is not None)):
            self.visit(default)
        for argument in (*root.args.posonlyargs, *root.args.args, *root.args.kwonlyargs):
            if argument.annotation is not None:
                self.visit(argument.annotation)
        if root.args.vararg is not None and root.args.vararg.annotation is not None:
            self.visit(root.args.vararg.annotation)
        if root.args.kwarg is not None and root.args.kwarg.annotation is not None:
            self.visit(root.args.kwarg.annotation)
        if root.returns is not None:
            self.visit(root.returns)
        for type_parameter in getattr(root, "type_params", ()):
            self.visit(type_parameter)
        for statement in root.body:
            self.visit(statement)
        self.calls.sort(key=lambda item: (item[0].lineno, item[0].col_offset))

    def visit(self, node: ast.AST):
        if not self.exhausted:
            return super().visit(node)
        return None

    def visit_Call(self, node: ast.Call) -> None:
        self.call_sites_seen += 1
        if self.call_sites_seen > MAX_CALL_SITES:
            self.exhausted = True
            return
        self.calls.append((node, self._nested_scope_depth > 0))
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Record nested calls without applying outer-scope resolution."""

        self._nested_scope_depth += 1
        self.generic_visit(node)
        self._nested_scope_depth -= 1

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Nested class bodies have their own namespace and remain unsupported."""

        self._nested_scope_depth += 1
        self.generic_visit(node)
        self._nested_scope_depth -= 1

    def visit_Lambda(self, node: ast.Lambda) -> None:
        """Record nested calls without applying outer-scope resolution."""

        self._nested_scope_depth += 1
        self.generic_visit(node)
        self._nested_scope_depth -= 1

    def _visit_comprehension_scope(self, node: ast.AST) -> None:
        """Keep comprehension-local calls out of the enclosing function scope."""

        self._nested_scope_depth += 1
        self.generic_visit(node)
        self._nested_scope_depth -= 1

    visit_ListComp = _visit_comprehension_scope
    visit_SetComp = _visit_comprehension_scope
    visit_DictComp = _visit_comprehension_scope
    visit_GeneratorExp = _visit_comprehension_scope


def analyze_target(target: CodeTarget, context: CodeAnalysisContext) -> CodeAnalysisResult:
    """Emit bounded static call possibilities without executing user code.

    Direct global calls and direct methods on concretely annotated parameters may
    resolve. Attribute chains, aliases, call-result receivers, string annotations,
    and dynamic lookup remain non-resolved by design.
    """

    if not context.allow_source:
        return CodeAnalysisResult(target=target.spec, diagnostics=(DiagnosticFact(
            severity="info",
            code="dryml.code.source_disabled",
            message="Static call analysis requires source extraction.",
            source={"analyzer": "static_calls", "target_kind": target.spec.kind},
        ),))
    function = _function_for_target(target)
    source_target = (
        function
        if function is not None
        else target.unwrapped
        if target.unwrapped is not None
        else target.obj
    )
    info = get_source_info(source_target) if source_target is not None else None
    if info is None:
        return CodeAnalysisResult(target=target.spec, diagnostics=(DiagnosticFact(
            severity="warning",
            code="dryml.code.source_unavailable",
            message="No source is available for static call analysis.",
            source={"analyzer": "static_calls", "target_kind": target.spec.kind},
        ),))
    parsed, parse_diagnostic = parse_static_source(
        target,
        analyzer="static_calls",
        source=info.source,
        filename=info.filename,
        start_line=info.start_line,
    )
    if parse_diagnostic is not None:
        if parse_diagnostic.code == "dryml.code.static_ast_nodes_limit_exceeded":
            return CodeAnalysisResult(
                target=target.spec,
                facts=(_summary_fact(target, filename=info.filename, complete=False),),
                diagnostics=(parse_diagnostic,),
            )
        return CodeAnalysisResult(target=target.spec, diagnostics=(parse_diagnostic,))
    assert parsed is not None

    root = _analysis_root(parsed.tree)
    # Binding information must cover the entire selected lexical scope before a
    # site can be resolved, but call collection itself stops at its hard limit.
    bound_names = _bound_names(root)
    collector = _CallCollector()
    collector.collect_root(root)
    globals_mapping = _globals_for_target(function, source_target)
    annotations = _parameter_annotations(function)
    parameter_names = _parameter_names(root)
    free_names = _free_names(function)
    filename = bounded_string(parsed.filename)
    facts = tuple(
        _fact_for_call(
            target,
            call,
            nested_scope=nested_scope,
            globals_mapping=globals_mapping,
            annotations=annotations,
            parameter_names=parameter_names,
            free_names=free_names,
            bound_names=bound_names,
            filename=filename,
            start_line=parsed.start_line,
        )
        for call, nested_scope in collector.calls
    )
    target_reference_exhausted = any(
        fact.data["reason"] == "target_reference_limit_exceeded" for fact in facts
    )
    complete = not collector.exhausted and not target_reference_exhausted
    summary = _summary_fact(
        target,
        filename=filename,
        complete=complete,
        call_sites_seen=collector.call_sites_seen,
        facts_emitted=len(facts),
    )
    diagnostics: list[DiagnosticFact] = []
    if collector.exhausted:
        diagnostics.append(limit_diagnostic(
            target,
            "static_calls",
            limit_name="call_sites",
            limit=MAX_CALL_SITES,
            observed_lower_bound=collector.call_sites_seen,
        ))
    if target_reference_exhausted:
        diagnostics.append(DiagnosticFact(
            severity="error",
            code="dryml.code.static_target_reference_limit_exceeded",
            message="A static call target reference exceeded the serialized scalar limit.",
            source={"analyzer": "static_calls", "target_kind": target.spec.kind},
            data={
                "limit_name": "scalar_chars",
                "limit": MAX_STATIC_SCALAR_CHARS,
                "observed_lower_bound": MAX_STATIC_SCALAR_CHARS + 1,
            },
        ))
    return CodeAnalysisResult(target=target.spec, facts=(*facts, summary), diagnostics=tuple(diagnostics))


def can_analyze(target: CodeTarget, context: CodeAnalysisContext) -> bool:
    """Return true when this target can report source availability."""

    return (
        target.obj is not None
        or target.unwrapped is not None
        or target.spec.source_spec is not None
        or target.spec.import_path is not None
    )


def _analysis_root(tree: ast.AST) -> ast.AST:
    """Return the inspected function definition without descending into children."""

    if isinstance(tree, ast.Module):
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return node
    return tree


def _function_for_target(target: CodeTarget) -> Any | None:
    candidate = target.unwrapped if target.unwrapped is not None else target.obj
    if type(candidate) is types.MethodType:
        return object.__getattribute__(candidate, "__func__")
    return candidate if type(candidate) is types.FunctionType else None


def _globals_for_target(function: Any | None, source_target: Any | None) -> Mapping[str, Any]:
    """Return a safe globals mapping for a function or ordinary class source."""

    if function is not None:
        return object.__getattribute__(function, "__globals__")
    if type(source_target) is type:
        module_name = object.__getattribute__(source_target, "__module__")
        module = sys.modules.get(module_name)
        if module is not None:
            return vars(module)
    return {}


def _parameter_annotations(function: Any | None) -> Mapping[str, Any]:
    if function is None:
        return {}
    annotations = object.__getattribute__(function, "__annotations__")
    return annotations if isinstance(annotations, Mapping) else {}


def _parameter_names(root: ast.AST) -> set[str]:
    if not isinstance(root, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return set()
    arguments = root.args
    names = {argument.arg for argument in (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs)}
    if arguments.vararg is not None:
        names.add(arguments.vararg.arg)
    if arguments.kwarg is not None:
        names.add(arguments.kwarg.arg)
    return names


def _free_names(function: Any | None) -> set[str]:
    """Return closure names that must not be treated as globals."""

    if function is None:
        return set()
    code = object.__getattribute__(function, "__code__")
    return set(code.co_freevars)


def _fact_for_call(
    target: CodeTarget,
    call: ast.Call,
    *,
    nested_scope: bool,
    globals_mapping: Mapping[str, Any],
    annotations: Mapping[str, Any],
    parameter_names: set[str],
    free_names: set[str],
    bound_names: set[str],
    filename: str | None,
    start_line: int | None,
) -> StaticCallFact:
    func = call.func
    status = "unresolved"
    syntax = "other"
    display: str | None = None
    receiver: str | None = None
    method_name: str | None = None
    target_data: dict[str, str | None] | None = None
    reason: str | None = None

    if nested_scope:
        status = "unsupported"
        reason = "nested_scope_unsupported"
        if isinstance(func, ast.Name):
            syntax = "direct_name"
            display = func.id
            method_name = func.id
        elif isinstance(func, ast.Attribute):
            syntax = "attribute_chain"
            display = func.attr
            method_name = func.attr
        else:
            display = type(func).__name__.lower()
    elif isinstance(func, ast.Name):
        syntax = "direct_name"
        display = func.id
        method_name = func.id
        if func.id in parameter_names:
            reason = "parameter_name_unsupported"
        elif func.id in free_names:
            reason = "closure_name_unsupported"
        elif func.id in bound_names:
            reason = "local_name_unsupported"
        elif func.id not in globals_mapping:
            reason = "global_name_unavailable"
        else:
            value = globals_mapping[func.id]
            target_data, target_reason = _safe_target_data(value)
            if target_data is None:
                status = "unsupported"
                reason = target_reason or "global_value_not_safe_function"
            else:
                status = "resolved"
    elif isinstance(func, ast.Attribute):
        flattened = _flatten_name_attributes(func)
        if flattened is None:
            syntax = "other"
            display = "<call-result>." + func.attr if isinstance(func.value, ast.Call) else func.attr
            method_name = func.attr
            status = "unsupported"
            reason = "call_result_receiver" if isinstance(func.value, ast.Call) else "unsupported_receiver_expression"
        else:
            receiver, chain, chain_limit_exceeded = flattened
            method_name = chain[-1] if chain else None
            display = ".".join(((receiver or "<bounded>"), *chain))
            if chain_limit_exceeded:
                syntax = "attribute_chain"
                status = "unsupported"
                reason = "chain_limit_exceeded"
            elif len(chain) != 1:
                syntax = "attribute_chain"
                status = "unsupported"
                reason = "attribute_chain_unsupported"
            elif receiver in parameter_names:
                syntax = "annotated_receiver_method"
                if receiver in bound_names:
                    status = "unresolved"
                    reason = "receiver_reassigned"
                elif receiver not in annotations:
                    status = "unresolved"
                    reason = "missing_annotation"
                else:
                    annotation = annotations[receiver]
                    target_data, status, reason = _resolve_annotated_method(annotation, method_name)
            else:
                syntax = "attribute_chain"
                status = "unsupported"
                reason = "attribute_chain_unsupported"
    else:
        status = "unsupported"
        reason = "unsupported_callable_expression"
        display = type(func).__name__.lower()

    bounded_values = (
        bounded_string(display),
        bounded_string(receiver),
        bounded_string(method_name),
    )
    if any(
        original is not None and bounded is None
        for original, bounded in zip((display, receiver, method_name), bounded_values, strict=True)
    ):
        status = "unsupported"
        target_data = None
        reason = "scalar_limit_exceeded"
        display = "<bounded>"
        receiver = None
        method_name = None
    confidence = "exact_static" if status == "resolved" else "conservative_hint"
    relative_line = getattr(call, "lineno", None)
    return StaticCallFact(
        source={
            "analyzer": "static_calls",
            "target_kind": bounded_string(target.spec.kind) or "<bounded>",
            "filename": bounded_string(filename),
        },
        data={
            "status": status,
            "confidence": confidence,
            "syntax": syntax,
            "display": display,
            "receiver": receiver,
            "method_name": method_name,
            "target": target_data if status == "resolved" else None,
            "reason": None if status == "resolved" else reason,
            "relative_line": relative_line,
            "absolute_line": absolute_line(relative_line, start_line),
            "col_offset": getattr(call, "col_offset", None),
        },
    )


def _flatten_name_attributes(node: ast.Attribute) -> tuple[str | None, tuple[str, ...], bool] | None:
    chain: list[str] = []
    current: ast.AST = node
    while isinstance(current, ast.Attribute):
        if len(chain) == MAX_CHAIN_COMPONENTS:
            return None, tuple(reversed(chain)), True
        chain.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    chain.reverse()
    return current.id, tuple(chain), False


class _BindingCollector(ast.NodeVisitor):
    """Collect lexical bindings while deliberately skipping nested scopes."""

    def __init__(self) -> None:
        self.names: set[str] = set()

    def _add_targets(self, targets: tuple[ast.AST, ...]) -> None:
        for target in targets:
            self.names.update(_target_names(target))
            self.names.update(_attribute_receiver_names(target))

    def visit_Assign(self, node: ast.Assign) -> None:
        self._add_targets(tuple(node.targets))
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._add_targets((node.target,))
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._add_targets((node.target,))
        self.generic_visit(node)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self._add_targets((node.target,))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Treat dynamic attribute mutation as a conservative receiver rebind."""

        if (
            isinstance(node.func, ast.Name)
            and node.func.id in {"setattr", "delattr"}
            and node.args
            and isinstance(node.args[0], ast.Name)
        ):
            self.names.add(node.args[0].id)
        self.generic_visit(node)

    def visit_Delete(self, node: ast.Delete) -> None:
        self._add_targets(tuple(node.targets))
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self._add_targets((node.target,))
        self.generic_visit(node)

    visit_AsyncFor = visit_For

    def visit_With(self, node: ast.With) -> None:
        self._add_targets(tuple(item.optional_vars for item in node.items if item.optional_vars is not None))
        self.generic_visit(node)

    visit_AsyncWith = visit_With

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name is not None:
            self.names.add(node.name)
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        self.names.update(alias.asname or alias.name.split(".", 1)[0] for alias in node.names)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.names.update(alias.asname or alias.name for alias in node.names)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.names.add(node.name)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.names.add(node.name)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return None

    def visit_MatchAs(self, node: ast.MatchAs) -> None:
        if node.name is not None:
            self.names.add(node.name)
        self.generic_visit(node)

    def visit_MatchStar(self, node: ast.MatchStar) -> None:
        if node.name is not None:
            self.names.add(node.name)

    def visit_MatchMapping(self, node: ast.MatchMapping) -> None:
        if node.rest is not None:
            self.names.add(node.rest)
        self.generic_visit(node)

    def _visit_comprehension(self, node: ast.AST) -> None:
        """Treat comprehension bindings conservatively as local names."""

        for generator in node.generators:
            self._add_targets((generator.target,))
        self.generic_visit(node)

    visit_ListComp = _visit_comprehension
    visit_SetComp = _visit_comprehension
    visit_DictComp = _visit_comprehension
    visit_GeneratorExp = _visit_comprehension


def _bound_names(root: ast.AST) -> set[str]:
    """Return bindings in the inspected function's lexical scope."""

    collector = _BindingCollector()
    if isinstance(root, (ast.FunctionDef, ast.AsyncFunctionDef)):
        for statement in root.body:
            collector.visit(statement)
    else:
        collector.visit(root)
    return collector.names


def _target_names(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, (ast.Tuple, ast.List)):
        return set().union(*(_target_names(item) for item in node.elts))
    return set()


def _attribute_receiver_names(node: ast.AST) -> set[str]:
    """Return receiver names mutated through attribute assignment or deletion."""

    if not isinstance(node, ast.Attribute):
        return set()
    current: ast.AST = node
    while isinstance(current, ast.Attribute):
        current = current.value
    return {current.id} if isinstance(current, ast.Name) else set()


def _resolve_annotated_method(annotation: Any, method_name: str | None) -> tuple[dict[str, str | None] | None, str, str | None]:
    if method_name is None:
        return None, "unsupported", "method_name_unavailable"
    if type(annotation) is str:
        return None, "unresolved", "string_annotation"
    if not issubclass(type(annotation), type):
        return None, "ambiguous", "non_concrete_annotation"
    if type(annotation) is not type:
        return None, "ambiguous", "non_standard_annotation_class"
    try:
        descriptor = inspect.getattr_static(annotation, method_name)
    except AttributeError:
        return None, "unresolved", "annotated_method_missing"
    if type(descriptor) is property:
        return None, "unsupported", "property_descriptor"
    if type(descriptor) in {staticmethod, classmethod}:
        candidate = descriptor.__func__
    elif isinstance(descriptor, types.FunctionType):
        candidate = descriptor
    else:
        return None, "unsupported", "annotated_member_not_safe_function"
    target_data, target_reason = _safe_target_data(candidate, method_name=method_name)
    if target_data is None:
        return None, "unsupported", target_reason or "target_reference_limit_exceeded"
    return target_data, "resolved", None


def _summary_fact(
    target: CodeTarget,
    *,
    filename: str | None,
    complete: bool,
    call_sites_seen: int = 0,
    facts_emitted: int = 0,
) -> CodeFact:
    """Build the mandatory bounded summary for one static-call analysis run."""

    return CodeFact(
        kind="static_call_summary",
        source={
            "analyzer": "static_calls",
            "target_kind": bounded_string(target.spec.kind) or "<bounded>",
            "filename": bounded_string(filename),
        },
        data={
            "complete": complete,
            "call_sites_seen": call_sites_seen,
            "facts_emitted": facts_emitted,
            "limits": STATIC_ANALYSIS_LIMITS,
        },
    )


def _safe_target_data(
    value: Any,
    *,
    method_name: str | None = None,
    subject_ref: str | None = None,
) -> tuple[dict[str, str | None] | None, str | None]:
    """Describe a plain function or ordinary class without dynamic access.

    Arbitrary callable instances and descriptors are intentionally unsupported:
    normalizing them can trigger user-defined attribute hooks.
    """

    if type(value) not in {types.FunctionType, types.BuiltinFunctionType, type}:
        return None, "global_value_not_safe_function"
    if type(value) is types.BuiltinFunctionType:
        receiver = object.__getattribute__(value, "__self__")
        if receiver is not None and type(receiver) is not types.ModuleType:
            return None, "global_value_not_safe_function"
    module_name = object.__getattribute__(value, "__module__")
    qualname = object.__getattribute__(value, "__qualname__")
    import_path = None
    if (
        isinstance(module_name, str)
        and isinstance(qualname, str)
        and module_name != "__main__"
        and "<locals>" not in qualname
    ):
        candidate_path = f"{module_name}:{qualname}"
        if len(candidate_path) > MAX_STATIC_SCALAR_CHARS:
            return None, "target_reference_limit_exceeded"
        if _path_resolves_to_target(candidate_path, value):
            import_path = candidate_path
    if import_path is None:
        return None, "target_reference_unavailable"
    target = bounded_target_mapping({
        "kind": "class" if type(value) is type else "function",
        "import_path": import_path,
        "method_name": method_name,
        "subject_ref": subject_ref,
    })
    return target, None if target is not None else "target_reference_limit_exceeded"


def _path_resolves_to_target(path: str, value: Any) -> bool:
    """Verify a prospective import path without dynamic module/class lookups."""

    module_name, qualname = path.split(":", 1)
    module = sys.modules.get(module_name)
    if module is None:
        return False
    current: Any = module
    try:
        for part in qualname.split("."):
            current = inspect.getattr_static(current, part)
    except AttributeError:
        return False
    if type(current) in {staticmethod, classmethod}:
        current = current.__func__
    return current is value


ANALYZER = FunctionAnalyzer("static_calls", analyze_target, can_analyze)


__all__ = [
    "ANALYZER",
    "MAX_AST_NODES",
    "MAX_CALL_SITES",
    "MAX_CHAIN_COMPONENTS",
    "MAX_RESOLUTION_DIAGNOSTICS",
    "MAX_SOURCE_BYTES",
    "MAX_STATIC_SCALAR_CHARS",
    "analyze_target",
]
