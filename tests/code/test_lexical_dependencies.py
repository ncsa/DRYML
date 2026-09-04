"""Tests for dependency-light lexical free-name evidence."""

from __future__ import annotations

import subprocess
import sys

import pytest

from dryml.code import ImportTarget, KernelCall, SourceTarget, analyze
from dryml.code.algorithms import (
    LexicalDependencies,
    LexicalDependency,
    LexicalDependencyKernel,
    collect_lexical_dependencies,
)
from dryml.code.errors import SourceUnavailableError
from dryml.code.facts import SourceLocation


_SCOPED_SOURCE = """def subject(
    value: ValueAnnotation = default_factory(default_arg),
    *, setting: SettingAnnotation = setting_default,
) -> ReturnAnnotation:
    @nested_decorator
    def nested(arg: NestedAnnotation = nested_default):
        return outer_value + value + arg + nested_global

    class Local(BaseClass, metaclass=MetaClass):
        field: FieldType = class_value

        def method(self, item: MethodAnnotation = method_default):
            return method_global + item

    import package.subpackage as package_alias
    from module import imported as local_import
    for entry in iterable:
        result = transform(entry)
    with manager as resource:
        result = resource.consume()
    try:
        hazardous()
    except ErrorType as caught:
        result = caught
    pairs = [(computed := mapping[item]) for item in source_items if predicate(item)]
    match candidate:
        case Pattern(bound, second=other) if guard(other):
            return final(bound)
    return nested
"""


def _names(result: LexicalDependencies) -> tuple[str, ...]:
    """Project ordered dependency names from one public result."""

    return tuple(dependency.name for dependency in result.dependencies)


def test_collects_free_names_across_supported_lexical_scopes() -> None:
    """Headers, bodies, bindings, and nested scopes retain only free names."""

    result = collect_lexical_dependencies(
        SourceTarget(_SCOPED_SOURCE, name="subject", filename="private/path/subject.py")
    )

    assert _names(result) == (
        "ValueAnnotation",
        "default_factory",
        "default_arg",
        "SettingAnnotation",
        "setting_default",
        "ReturnAnnotation",
        "nested_decorator",
        "NestedAnnotation",
        "nested_default",
        "outer_value",
        "nested_global",
        "BaseClass",
        "MetaClass",
        "FieldType",
        "class_value",
        "MethodAnnotation",
        "method_default",
        "method_global",
        "iterable",
        "transform",
        "manager",
        "hazardous",
        "ErrorType",
        "mapping",
        "source_items",
        "predicate",
        "candidate",
        "Pattern",
        "guard",
        "final",
    )
    assert all(type(dependency.source) is SourceLocation for dependency in result.dependencies)
    assert result.dependencies[0].source == SourceLocation("subject.py", 2, 11)


def test_comprehension_targets_named_expressions_and_patterns_do_not_escape() -> None:
    """Comprehension and match bindings remain local while their inputs remain free."""

    result = collect_lexical_dependencies(
        SourceTarget(
            """def subject():
    values = [(captured := transform(item)) for item in source if predicate(item)]
    match candidate:
        case Point(x, y=other) if guard(other):
            return finish(x, captured)
    return values
""",
            name="subject",
        )
    )

    assert _names(result) == ("transform", "source", "predicate", "candidate", "Point", "guard", "finish")


@pytest.mark.skipif(sys.version_info < (3, 12), reason="type parameters require Python 3.12+")
def test_type_parameters_bind_annotations_and_function_body() -> None:
    """Maintained-version type parameters are local while their bounds remain free."""

    result = collect_lexical_dependencies(
        SourceTarget(
            """def subject[T: Bound](value: T) -> T:
    return generic(value)
""",
            name="subject",
        )
    )

    assert _names(result) == ("Bound", "generic")


def test_convenience_and_direct_kernel_calls_return_identical_evidence() -> None:
    """The convenience wrapper is exactly the public unfused kernel request."""

    target = SourceTarget("def subject(value):\n    return missing(value)\n", name="subject")

    convenience = collect_lexical_dependencies(target)
    direct = analyze(target, (KernelCall(LexicalDependencyKernel(), None),)).require(LexicalDependencyKernel)

    assert direct == convenience
    assert _names(convenience) == ("missing",)


def test_collection_is_static_and_applies_no_live_or_import_policy() -> None:
    """Missing names remain evidence without invoking source or resolving values."""

    result = collect_lexical_dependencies(
        SourceTarget("def subject():\n    return caller_local + unimportable_name\n", name="subject")
    )

    assert _names(result) == ("caller_local", "unimportable_name")
    assert collect_lexical_dependencies(ImportTarget("math")) == LexicalDependencies(())


def test_public_values_validate_and_malformed_source_propagates_typed_failure() -> None:
    """Public result carriers reject mutable values and preserve source failures."""

    with pytest.raises(ValueError):
        LexicalDependency("", None)
    with pytest.raises(ValueError):
        LexicalDependencies([])  # type: ignore[arg-type]
    with pytest.raises(SourceUnavailableError) as error:
        collect_lexical_dependencies(SourceTarget("def subject(:\n", name="subject"))

    assert error.value.code == "source.invalid"


def test_algorithm_package_is_dependency_light_and_not_reexported_by_code() -> None:
    """The concrete built-in stays in its package without loading product modules."""

    import dryml.code as code

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import dryml.code.algorithms; "
            "print(','.join(sorted(name for name in sys.modules if name.startswith('dryml.'))))",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    loaded = set(filter(None, result.stdout.strip().split(",")))
    assert "LexicalDependencyKernel" not in code.__all__
    assert not hasattr(code, "LexicalDependencyKernel")
    assert loaded <= {
        "dryml.code",
        "dryml.code.algorithms",
        "dryml.code.algorithms.lexical_dependencies",
        "dryml.code.analysis",
        "dryml.code.ast_tools",
        "dryml.code.callable_info",
        "dryml.code.errors",
        "dryml.code.facts",
        "dryml.code.graph",
        "dryml.code.kernels",
        "dryml.code.probe",
        "dryml.code.source",
        "dryml.code.targets",
        "dryml.code.trace",
        "dryml._framework_imports",
    }
