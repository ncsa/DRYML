"""Tests for bounded provider-neutral static dependency resolution."""

from __future__ import annotations

import inspect
import functools
import types

import pytest

from dryml.code import (
    CodeTarget,
    AnalysisKernel,
    InvalidKernelError,
    KernelCall,
    StaticDependenciesKernel,
    capture_inspection,
    probe,
)


_PARITY_TARGETS: list[object] = []


class ParityConsumer(AnalysisKernel[None, tuple[str | None, ...]]):
    """Require static dependencies and retain scheduler target identity."""

    input_type = type(None)
    output_type = tuple
    requires = (StaticDependenciesKernel, )

    def run(self, graph: object, value: None,
            context: object) -> tuple[str | None, ...]:
        """Return dependency order after checking canonical root identity."""

        require = context.require  # type: ignore[union-attr]
        target = context.target  # type: ignore[union-attr]
        dependencies = require(StaticDependenciesKernel)
        assert dependencies.targets[0] is target
        _PARITY_TARGETS.append(target)
        return tuple(target.info.name for target in dependencies.targets)


class ParityFailure(AnalysisKernel[None, int]):
    """Fail after the dependency consumer to exercise propagation."""

    input_type = type(None)
    output_type = int
    requires = (ParityConsumer, )

    def run(self, graph: object, value: None, context: object) -> int:
        """Fail after validating the required output is available."""

        context.require(ParityConsumer)  # type: ignore[union-attr]
        raise RuntimeError("intentional parity failure")


class ParitySkipped(AnalysisKernel[None, int]):
    """Depend on a failed producer so the scheduler skips this kernel."""

    input_type = type(None)
    output_type = int
    requires = (ParityFailure, )

    def run(self, graph: object, value: None, context: object) -> int:
        """Provide a body that must not run after the failed producer."""

        raise AssertionError("skipped consumer ran")


def _leaf() -> None:
    """Provide a passive terminal helper."""


def _middle() -> None:
    """Reach the terminal helper through a direct global binding."""

    _leaf()


def _root() -> None:
    """Reach the intermediate helper through a direct global binding."""

    _middle()


def _cycle_left() -> None:
    """Provide one side of a passive direct-call cycle."""

    _cycle_right()


def _cycle_right() -> None:
    """Provide the other side of a passive direct-call cycle."""

    _cycle_left()


def _dormant_definition() -> None:
    """Keep a nested call dormant rather than making it an invocation edge."""

    def nested() -> None:
        """Contain a helper call that must remain outside root traversal."""

        _leaf()

    return None


def test_static_dependencies_resolve_transitive_capture_closure(
) -> None:
    """Direct global edges produce complete root-to-leaf targets."""

    capture = capture_inspection(_root)
    result = probe(
        capture.target,
        (KernelCall(StaticDependenciesKernel(), None), ),
    )
    dependencies = result.require(StaticDependenciesKernel)

    assert dependencies.complete
    assert tuple(target.info.name for target in dependencies.targets) == (
        "_root",
        "_middle",
        "_leaf",
    )


def test_static_dependencies_keep_live_targets_local_and_bound_cycles(
) -> None:
    """Live resolution keeps handles local and reports cycle caps."""

    live = probe(
        _root,
        (KernelCall(StaticDependenciesKernel(), None), ),
    ).require(StaticDependenciesKernel)
    cycle = probe(
        capture_inspection(_cycle_left).target,
        (KernelCall(StaticDependenciesKernel(max_targets=1), None), ),
    ).require(StaticDependenciesKernel)

    assert type(live.targets[0]) is CodeTarget
    assert cycle.complete is False
    assert cycle.diagnostics == ("static.target_limit", )


def test_static_dependencies_preindex_live_capture_associations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Live resolution uses the capture's local target index once."""

    from dryml.code.inspection import InspectionCapture

    monkeypatch.setattr(
        InspectionCapture,
        "local_target",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError(
                "resolver must use its request-local association index"
            )
        ),
    )

    dependencies = probe(
        _root,
        (KernelCall(StaticDependenciesKernel(), None), ),
    ).require(StaticDependenciesKernel)

    assert tuple(target.info.name for target in dependencies.targets) == (
        "_root",
        "_middle",
        "_leaf",
    )


def test_wrapper_metadata_neither_unwraps_nor_uses_signature_hooks() -> None:
    """The resolver follows wrapper bodies rather than wrapper metadata."""

    def wrapper() -> None:
        """Invoke the real dependency from the wrapper's own body."""

        _leaf()

    wrapper.__wrapped__ = wrapper  # type: ignore[attr-defined]
    wrapper.__signature__ = inspect.Signature()  # type: ignore[attr-defined]
    dependencies = probe(
        wrapper,
        (KernelCall(StaticDependenciesKernel(), None), ),
    ).require(StaticDependenciesKernel)

    assert tuple(target.info.name for target in dependencies.targets) == (
        "wrapper",
        "_leaf",
    )


def test_static_dependencies_ignore_dormant_nested_definition_bodies() -> None:
    """Only the selected invocation body contributes static call edges."""

    dependencies = probe(
        _dormant_definition,
        (KernelCall(StaticDependenciesKernel(), None), ),
    ).require(StaticDependenciesKernel)

    assert dependencies.complete
    assert tuple(
        target.info.name
        for target in dependencies.targets) == ("_dormant_definition", )


def test_static_dependencies_mark_excluded_nested_definition_work_incomplete(
) -> None:
    """Keep eager nested-definition work out of edges but not coverage."""

    def nested_default() -> None:
        """Define a nested function whose default calls a known helper."""

        def nested(value=_leaf()) -> None:
            """Retain an excluded definition-time default expression."""

            del value

        del nested

    def nested_class() -> None:
        """Define a nested class whose body contains a known helper call."""

        class Local:
            """Retain an excluded definition-time class body call."""

            _leaf()

        del Local

    for subject in (nested_default, nested_class):
        dependencies = probe(
            subject,
            (KernelCall(StaticDependenciesKernel(), None), ),
        ).require(StaticDependenciesKernel)

        assert tuple(target.info.name for target in dependencies.targets) == (
            subject.__name__,
        )
        assert not dependencies.complete


def test_static_dependencies_keep_comprehension_bindings_local() -> None:
    """Resolve a later global call after a same-named comprehension target."""

    def helper() -> None:
        """Provide the global call after the comprehension scope ends."""

    def subject(values: object) -> None:
        """
        Bind ``helper`` only inside the immediately evaluated comprehension.
        """

        [helper for helper in values]  # type: ignore[union-attr]
        helper()

    dependencies = probe(
        subject,
        (KernelCall(StaticDependenciesKernel(), None), ),
    ).require(StaticDependenciesKernel)

    assert tuple(target.info.name for target in dependencies.targets) == (
        "subject",
        "helper",
    )
    assert dependencies.complete


def test_static_dependencies_preserve_generator_eager_iterable_and_uncertainty(
) -> None:
    """Keep the first generator iterable while excluding its deferred body."""

    def values() -> tuple[int, ...]:
        """Provide the generator expression's eager outer iterable."""

        return ()

    def subject(callback: object) -> object:
        """Build a generator whose callback body remains deferred."""

        return (callback() for _ in values())  # type: ignore[operator]

    dependencies = probe(
        subject,
        (KernelCall(StaticDependenciesKernel(), None), ),
    ).require(StaticDependenciesKernel)

    assert tuple(target.info.name for target in dependencies.targets) == (
        "subject",
        "values",
    )
    assert not dependencies.complete


def test_static_dependencies_marks_max_depth_boundary_incomplete() -> None:
    """Retain the boundary target and report its omitted call edges."""

    dependencies = probe(
        capture_inspection(_root).target,
        (KernelCall(StaticDependenciesKernel(max_depth=1), None), ),
    ).require(StaticDependenciesKernel)

    assert tuple(target.info.name for target in dependencies.targets) == (
        "_root",
        "_middle",
    )
    assert not dependencies.complete
    assert dependencies.diagnostics == ("static.depth_limit", )


def test_static_dependencies_respect_scope_shadowing_and_global_declarations(
) -> None:
    """Assignments, parameters, and imports cannot resolve a global twin."""

    def global_helper() -> None:
        """Provide a global that shadowed calls must not select."""

    def subject(global_helper: object) -> None:
        """Contain parameter, assignment, import, and deletion shadowing."""

        global_helper()  # type: ignore[operator]
        alias = _leaf
        alias()
        alias = global_helper
        del alias
        import types as module

        module.FunctionType()

    dependencies = probe(
        subject,
        (KernelCall(StaticDependenciesKernel(), None), ),
    ).require(StaticDependenciesKernel)

    assert tuple(target.info.name
                 for target in dependencies.targets) == ("subject", )
    assert not dependencies.complete


def test_static_dependencies_resolve_a_proven_non_shadowable_receiver_member(
) -> None:
    """Only a slots-only receiver can establish a ``self.method`` edge."""

    class Safe:
        __slots__ = ()

        def run(self) -> None:
            """Call the proven non-shadowable helper."""

            self.helper()

        def helper(self) -> None:
            """Provide a selected instance member."""

    class Shadowable:

        def run(self) -> None:
            """Leave an instance-dictionary call unresolved."""

            self.helper()

        def helper(self) -> None:
            """Provide a member that an instance attribute can shadow."""

    safe = probe(
        Safe().run,
        (KernelCall(StaticDependenciesKernel(), None), ),
    ).require(StaticDependenciesKernel)
    shadowable = probe(
        Shadowable().run,
        (KernelCall(StaticDependenciesKernel(), None), ),
    ).require(StaticDependenciesKernel)

    assert tuple(target.info.name for target in safe.targets) == (
        "run",
        "helper",
    )
    assert safe.complete
    assert tuple(target.info.name
                 for target in shadowable.targets) == ("run", )
    assert not shadowable.complete


def test_static_dependencies_traverse_standard_construction(
) -> None:
    """Class calls include constructors and flag custom metacalls."""

    class Constructed:

        def __new__(cls) -> "Constructed":
            """Provide a passive Python allocation hook."""

            return object.__new__(cls)

        def __init__(self) -> None:
            """Provide a passive Python initialization hook."""

    class Meta(type):

        def __call__(cls) -> object:
            """Provide a custom construction path with uncertain forwarding."""

            return object.__new__(cls)

    class Custom(metaclass=Meta):
        pass

    def ordinary() -> None:
        """Call a class using the ordinary construction protocol."""

        Constructed()

    def custom() -> None:
        """Call a class with a custom metaclass."""

        Custom()

    ordinary_dependencies = probe(
        ordinary,
        (KernelCall(StaticDependenciesKernel(), None), ),
    ).require(StaticDependenciesKernel)
    custom_dependencies = probe(
        custom,
        (KernelCall(StaticDependenciesKernel(), None), ),
    ).require(StaticDependenciesKernel)

    assert {target.info.name
            for target in ordinary_dependencies.targets} >= {
                "ordinary",
                "Constructed",
                "__new__",
                "__init__",
            }
    assert ordinary_dependencies.complete
    assert {target.info.name
            for target in custom_dependencies.targets} >= {
                "custom",
                "Custom",
                "__call__",
            }
    assert not custom_dependencies.complete


def test_static_dependencies_leave_conditional_aliases_unresolved(
) -> None:
    """Conditional aliases and arbitrary locals cannot establish an edge."""

    class Safe:
        __slots__ = ()

        def helper(self) -> None:
            """Provide an otherwise proven method."""

        def run(self, other: object, condition: bool) -> None:
            """Use unsupported conditional and non-receiver dispatch."""

            if condition:
                alias = _leaf
            alias()
            other.helper()  # type: ignore[attr-defined]

    dependencies = probe(
        Safe().run,
        (KernelCall(StaticDependenciesKernel(), None), ),
    ).require(StaticDependenciesKernel)

    assert tuple(target.info.name
                 for target in dependencies.targets) == ("run", )
    assert not dependencies.complete


def test_nested_wraps_collects_wrapper_closure_edges() -> None:
    """Wrapped functions retain wrapper-body dependencies and closures."""

    def closure_helper() -> None:
        """Provide a closure-only wrapper dependency."""

    def decorate(target: object) -> object:
        """Create a wrapper whose copied name differs from its code name."""

        @functools.wraps(target)
        def wrapper() -> None:
            """Call the closure dependency from the selected wrapper body."""

            closure_helper()

        return wrapper

    @decorate
    def wrapped() -> None:
        """Supply metadata copied onto the real wrapper."""

    dependencies = probe(
        wrapped,
        (KernelCall(StaticDependenciesKernel(), None), ),
    ).require(StaticDependenciesKernel)

    assert tuple(target.info.name for target in dependencies.targets) == (
        "wrapped",
        "closure_helper",
    )
    assert dependencies.complete


def test_static_dependencies_keep_scope_writes_unresolved() -> None:
    """Scope and comprehension writes shadow global helper names."""

    def helper() -> None:
        """Provide a global that all local captures must hide."""

    def subject(value: object, callbacks: object) -> None:
        """Contain distinct scope writes that must not select ``helper``."""

        helper: object
        helper()  # type: ignore[used-before-assignment]
        [helper() for helper in callbacks]  # type: ignore[union-attr]
        (helper() for helper in callbacks)  # type: ignore[union-attr]
        match value:
            case helper:
                helper()  # type: ignore[operator]

    dependencies = probe(
        subject,
        (KernelCall(StaticDependenciesKernel(), None), ),
    ).require(StaticDependenciesKernel)

    assert tuple(target.info.name
                 for target in dependencies.targets) == ("subject", )
    assert not dependencies.complete


def test_static_dependencies_reject_global_nonlocal_and_receiver_rebinding(
) -> None:
    """Writes through declared outer scopes and ``self`` invalidate proofs."""

    def helper() -> None:
        """Provide a global that a declared write must not select."""

    def global_write() -> None:
        """Assign through a global declaration before calling its name."""

        global helper
        helper = object
        helper()

    def outer() -> object:
        """Create a nonlocal write subject."""

        nonlocal_helper = helper

        def nonlocal_write() -> None:
            """Assign a nonlocal declaration before calling its name."""

            nonlocal nonlocal_helper
            nonlocal_helper = object
            nonlocal_helper()

        return nonlocal_write

    class Safe:
        """Slots-only class whose receiver is deliberately rebound."""

        __slots__ = ()

        def helper(self) -> None:
            """Provide a normally admissible member."""

        def run(self) -> None:
            """Rebind the receiver before the syntactic member call."""

            self = object()  # type: ignore[self-cls-assignment]
            self.helper()  # type: ignore[attr-defined]

    for subject in (global_write, outer(), Safe().run):
        dependencies = probe(
            subject,
            (KernelCall(StaticDependenciesKernel(), None), ),
        ).require(StaticDependenciesKernel)
        assert "helper" not in tuple(target.info.name
                                     for target in dependencies.targets)
        assert not dependencies.complete


def test_static_dependencies_resolve_module_and_descriptor_paths() -> None:
    """Proven module paths and descriptors retain selected owners."""

    module = types.ModuleType("static_dependencies_fixture")
    child = types.ModuleType("static_dependencies_fixture.child")

    def module_helper() -> None:
        """Provide a nested module member."""

    child.module_helper = module_helper
    module.child = child

    class Slots:
        """Expose only non-shadowable descriptor members."""

        __slots__ = ()

        @staticmethod
        def static() -> None:
            """Provide a static selected target."""

        @classmethod
        def class_method(cls) -> None:
            """Provide a class selected target."""

        def run(self) -> None:
            """Reach all supported static paths."""

            module.child.module_helper()
            self.static()
            self.class_method()

    dependencies = probe(
        Slots().run,
        (KernelCall(StaticDependenciesKernel(), None), ),
    ).require(StaticDependenciesKernel)

    assert {target.info.name
            for target in dependencies.targets} >= {
                "run",
                "module_helper",
                "static",
                "class_method",
            }
    assert dependencies.complete


def test_static_dependencies_does_not_summarize_shadowed_object_constructors(
) -> None:
    """Only builtin ``object`` receives a construction terminal summary."""

    class FakeObject:
        """Provide a shadowable object-like constructor helper."""

        @staticmethod
        def __new__(cls: object) -> object:
            """Supply a callable which must not be silently ignored."""

            return cls

    def subject() -> None:
        """Call a global binding named ``object`` rather than the builtin."""

        object.__new__(FakeObject)

    original = subject.__globals__.get("object")
    subject.__globals__["object"] = FakeObject
    try:
        dependencies = probe(
            subject,
            (KernelCall(StaticDependenciesKernel(), None), ),
        ).require(StaticDependenciesKernel)
    finally:
        if original is None:
            del subject.__globals__["object"]
        else:
            subject.__globals__["object"] = original

    assert tuple(target.info.name for target in dependencies.targets) == (
        "subject",
        "__new__",
    )


def test_static_dependency_scheduler_has_live_snapshot_parity() -> None:
    """One DAG preserves canonical roots, order, output, failure, and skips."""

    from dryml.code.targets import normalize_target

    live_target = normalize_target(_root)
    snapshot_target = capture_inspection(_root).target
    calls = (
        KernelCall(ParitySkipped(), None),
        KernelCall(ParityFailure(), None),
        KernelCall(ParityConsumer(), None),
        KernelCall(StaticDependenciesKernel(), None),
    )
    _PARITY_TARGETS.clear()
    live = probe(live_target, calls)
    snapshot = probe(snapshot_target, calls)

    assert _PARITY_TARGETS == [live_target, snapshot_target]
    assert tuple(outcome.status for outcome in live.outcomes) == (
        "skipped",
        "failed",
        "succeeded",
        "succeeded",
    )
    snapshot_statuses = tuple(
        outcome.status for outcome in snapshot.outcomes)
    live_statuses = tuple(outcome.status for outcome in live.outcomes)
    assert snapshot_statuses == live_statuses
    parity_output = (
        live.require(ParityConsumer),
        snapshot.require(ParityConsumer),
    )
    assert parity_output == (
        ("_root", "_middle", "_leaf"),
        ("_root", "_middle", "_leaf"),
    )
    assert live.outcomes[0].skipped_for == (ParityFailure, )
    assert snapshot.outcomes[0].skipped_for == (ParityFailure, )


def test_static_dependencies_reject_mismatched_kernel_input() -> None:
    """The standard scheduler rejects non-``None`` static dependency inputs."""

    with pytest.raises(InvalidKernelError):
        probe(_root, (KernelCall(StaticDependenciesKernel(), object()), ))
