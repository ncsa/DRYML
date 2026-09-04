"""Tests for safe target normalization and immutable metadata."""

from __future__ import annotations

import gc
import weakref

import pytest

from dryml.code import (
    DescriptorTarget,
    CodeFact,
    CodeFacts,
    ImportTarget,
    InvalidTargetError,
    SourceLocation,
    SourceTarget,
    normalize_target,
)


def target_function(value: int) -> int:
    """Provide a normal function target."""

    return value


class TargetOwner:
    """Provide supported descriptors and callable-instance behavior."""

    def method(self, value: int) -> int:
        """Provide a normal descriptor target."""

        return value

    @staticmethod
    def static(value: int) -> int:
        """Provide a static method descriptor target."""

        return value

    @classmethod
    def class_method(cls, value: int) -> int:
        """Provide a class method descriptor target."""

        return value

    def __call__(self, value: int) -> int:
        """Provide a supported callable instance target."""

        return value


def test_normalize_target_supports_declared_forms() -> None:
    """Supported targets preserve kind, static ownership, and safe provenance."""

    function = normalize_target(target_function)
    method = normalize_target(TargetOwner().method)
    instance = normalize_target(TargetOwner())
    descriptor = normalize_target(DescriptorTarget(TargetOwner, "static"))
    source = normalize_target(SourceTarget("def source_subject():\n    return 1\n", name="source_subject"))
    imported = normalize_target(ImportTarget("tests.code.test_targets:target_function"))
    imported_descriptor = normalize_target(ImportTarget("tests.code.test_targets:TargetOwner.static"))

    assert function.info.kind == "function"
    assert method.info.kind == "bound_method"
    assert instance.info.kind == "callable_instance"
    assert descriptor.info.kind == "descriptor"
    assert descriptor.info.descriptor_kind == "staticmethod"
    assert source.info.kind == "source"
    assert source.callable is None
    assert imported.info.kind == "import"
    assert imported.info.import_path == "tests.code.test_targets:target_function"
    assert imported_descriptor.info.kind == "import"
    assert imported_descriptor.info.descriptor_kind == "staticmethod"


def test_normalize_target_never_invokes_dynamic_protocols() -> None:
    """Dynamic lookup, descriptors, metaclass access, and constructors stay inert."""

    class Descriptor:
        def __get__(self, instance: object, owner: type | None = None) -> object:
            raise AssertionError("descriptor binding invoked")

    class Owner:
        value = Descriptor()

    class Dynamic:
        def __getattribute__(self, name: str) -> object:
            raise AssertionError("instance lookup invoked")

    with pytest.raises(InvalidTargetError) as descriptor_error:
        normalize_target(DescriptorTarget(Owner, "value"))
    with pytest.raises(InvalidTargetError) as dynamic_error:
        normalize_target(Dynamic())

    assert descriptor_error.value.code == "target.invalid"
    assert dynamic_error.value.code == "target.invalid"


def test_normalize_target_rejects_sourceless_functions() -> None:
    """Interactive and dynamically compiled functions are outside the whitelist."""

    namespace: dict[str, object] = {}
    exec(compile("def generated():\n    return 1\n", "<generated>", "exec"), namespace)

    with pytest.raises(InvalidTargetError) as error:
        normalize_target(namespace["generated"])  # type: ignore[arg-type]

    assert error.value.code == "target.invalid"


@pytest.mark.parametrize("path", ["", "math:", ":sqrt", "math:<locals>.sqrt", "math:sin..real"])
def test_import_target_rejects_malformed_paths(path: str) -> None:
    """The import grammar rejects ambiguity before attempting dynamic lookup."""

    with pytest.raises(InvalidTargetError) as error:
        normalize_target(ImportTarget(path))

    assert error.value.code == "target.invalid"


def test_target_paths_are_sanitized_and_request_handles_release() -> None:
    """Returned provenance omits directories and discarded targets do not leak."""

    class WeakCallable:
        def __call__(self) -> None:
            """Provide a supported callable target."""

    live = WeakCallable()
    reference = weakref.ref(live)
    target = normalize_target(live)

    assert target.info.filename is None or not target.info.filename.startswith("/")
    assert SourceLocation("/secret/project/subject.py", 1, 0).filename == "subject.py"
    del live
    del target
    gc.collect()
    assert reference() is None


def test_facts_are_frozen_closed_values() -> None:
    """Framework-created facts reject mutable, unsorted, and subclass payloads."""

    fact = CodeFact("access", (("chain", ("one", "two")), ("root", "obj")))
    facts = CodeFacts((fact,))

    assert facts.values == (fact,)
    with pytest.raises(ValueError):
        CodeFact("access", (("root", "obj"), ("chain", ())))
    with pytest.raises(ValueError):
        CodeFact("access", ["mutable"])  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        CodeFacts((fact, object()))  # type: ignore[arg-type]
