"""Tests for detached passive inspection projections."""

from __future__ import annotations

import hashlib

import pytest

from dryml.code import (
    AnalysisKernel,
    InspectionTarget,
    InvalidTargetError,
    KernelCall,
    SourceUnavailableError,
    capture_inspection,
    extract_source,
    probe,
    trace,
)
from dryml.code.algorithms import collect_lexical_dependencies
from dryml.code.inspection import (
    InspectionCall,
    InspectionRecord,
    InspectionSnapshot,
)
from dryml.code.targets import TargetInfo

mutable_helper: object = None


def _helper() -> None:
    """Provide a reachable file-backed helper without executing it."""


def _root() -> None:
    """Provide a file-backed root for passive projection."""

    _helper()


class TargetKernel(AnalysisKernel[type, tuple]):
    """Return the context target identity and semantic kind."""

    input_type = type(None)
    output_type = tuple

    def run(self, graph: object, value: None, context: object) -> tuple:
        """Return canonical target evidence supplied by the scheduler."""

        target = context.target  # type: ignore[union-attr]
        return (type(target), target.info.kind)


def test_capture_projects_detached_target_through_normal_scheduler() -> None:
    """Snapshots retain semantic kind and schedule without source or trace."""

    capture = capture_inspection(_root)
    target = capture.target

    assert type(target) is InspectionTarget
    assert target.info.kind == "function"
    assert probe(
        target,
        (KernelCall(TargetKernel(), None), )).require(TargetKernel) == (
            InspectionTarget,
            "function",
        )
    with pytest.raises(SourceUnavailableError):
        extract_source(target)
    with pytest.raises(SourceUnavailableError):
        collect_lexical_dependencies(target)
    with pytest.raises(InvalidTargetError):
        trace(target, ())
    with pytest.raises(InvalidTargetError):
        capture_inspection(target)


def test_snapshot_rejects_invalid_versions_duplicate_ids_and_unknown_edges(
) -> None:
    """Malformed records fail before reaching the scheduler."""

    info = TargetInfo(
        "function",
        "subject",
        "tests.code.test_inspection_projection",
        "subject",
        None,
        None,
        None,
        None,
        None,
        None,
    )
    record = InspectionRecord("t0000", info, ())

    with pytest.raises(ValueError):
        InspectionSnapshot(2, "t0000", (record, ))
    with pytest.raises(ValueError):
        InspectionSnapshot(1, "t0000", (record, record))
    with pytest.raises(ValueError):
        InspectionSnapshot(
            1,
            "t0000",
            (InspectionRecord("t0000", info, (InspectionCall("missing"), )), ),
        )


def test_capture_does_not_mutate_globals_or_leak_source_metadata() -> None:
    """Capture reads referenced bindings and projects no local provenance."""

    global _helper
    original_helper = _helper

    def closure_factory() -> object:
        """Create a closure binding that shadows a module global."""

        def _helper() -> None:
            """Provide a closure-only helper."""

        def subject() -> None:
            """Reference the closure binding from its invocation body."""

            _helper()

        return subject

    subject = closure_factory()
    globals_before = dict(_root.__globals__)
    capture = capture_inspection(subject)

    assert _root.__globals__ == globals_before
    assert _helper is original_helper
    assert all(record.info.filename is None
               for record in capture.target.snapshot.records)
    assert all(record.info.start_line is None
               for record in capture.target.snapshot.records)


def test_snapshot_identity_uses_unambiguous_field_encoding() -> None:
    """Delimiter-join collisions retain distinct record identities."""

    info = TargetInfo(
        "function",
        "subject",
        "tests.code.test_inspection_projection",
        "subject",
        None,
        None,
        None,
        None,
        None,
        None,
    )
    first = InspectionSnapshot(1, "a:b", (InspectionRecord("a:b", info, ()), ))
    second = InspectionSnapshot(1, "a",
                                (InspectionRecord("a", info,
                                                  (InspectionCall("a"), )), ))

    assert first.identity != second.identity


def test_snapshot_identity_is_cached_with_the_existing_wire_encoding() -> None:
    """Validated snapshots hash once without changing their wire identity."""

    info = TargetInfo(
        "function", "subject", "example.module", "subject", "owner.module",
        "Owner", "staticmethod", None, None, None,
    )
    snapshot = InspectionSnapshot(
        1,
        "t0000",
        (InspectionRecord("t0000", info, (InspectionCall(None), )), ),
    )

    def wire(value: str | None) -> bytes:
        """Encode one reference field independently from production code."""

        if value is None:
            return b"n"
        raw = value.encode("utf-8")
        return b"s" + len(raw).to_bytes(8, "big") + raw

    payload = bytearray(b"dryml.code.inspection.v1")
    payload.extend((1).to_bytes(8, "big"))
    payload.extend(wire("t0000"))
    payload.extend(wire("t0000"))
    for value in (
        info.kind, info.name, info.module, info.qualname, info.owner_module,
        info.owner_qualname, info.descriptor_kind,
    ):
        payload.extend(wire(value))
    payload.extend(b"0")
    payload.extend((1).to_bytes(8, "big"))
    payload.extend(wire(None))
    expected = hashlib.sha256(payload).hexdigest()
    cached = InspectionSnapshot(
        1,
        "t0000",
        (InspectionRecord("t0000", info, (InspectionCall(None), )), ),
    )

    assert snapshot.identity == expected
    assert cached.identity == expected
    assert cached.identity == expected
    assert cached.identity == cached._identity


def test_capture_marks_oversize_source_incomplete_without_parsing() -> None:
    """Oversized source remains bounded, incomplete evidence."""

    from dryml.code import SourceTarget

    source = SourceTarget(
        "def subject():\n    pass\n#" + "x" * 1_048_576,
        name="subject",
    )
    capture = capture_inspection(source)

    assert capture.target.snapshot.record(capture.target.target_id).incomplete


def test_probe_bounds_source_before_normal_graph_construction() -> None:
    """Ordinary probe keeps oversized and deep roots incomplete."""

    from dryml.code import SourceTarget, StaticDependenciesKernel

    oversized = SourceTarget(
        "def subject():\n    pass\n#" + "x" * 1_048_576,
        name="subject",
    )
    deep = SourceTarget(
        "def subject():\n    return " + "nested(" * 129 + "value" + ")" * 129,
        name="subject",
    )

    for target in (oversized, deep):
        result = probe(
            target,
            (KernelCall(StaticDependenciesKernel(), None), ),
        )
        dependencies = result.require(StaticDependenciesKernel)
        assert not result.complete
        assert not dependencies.complete
        assert dependencies.targets[0].info.kind == "source"
        assert any(diagnostic.code == "source.unavailable"
                   for diagnostic in result.diagnostics)


def test_capture_drift_guard_rejects_relevant_global_rebinding() -> None:
    """The local guard covers captured globals as well as code identity."""

    def first() -> None:
        """Provide the initially captured helper."""

    def second() -> None:
        """Provide a replacement helper."""

    def subject() -> None:
        """Call the mutable module binding."""

        mutable_helper()

    original = globals().get("mutable_helper")
    globals()["mutable_helper"] = first
    try:
        capture = capture_inspection(subject)
        globals()["mutable_helper"] = second
        with pytest.raises(InvalidTargetError, match="changed"):
            capture.validate()
    finally:
        if original is None:
            del globals()["mutable_helper"]
        else:
            globals()["mutable_helper"] = original


def test_capture_drift_guard_rejects_proven_class_member_rebinding() -> None:
    """Class-member resolution records member and MRO identities."""

    class Owner:

        @staticmethod
        def helper() -> None:
            """Provide the captured static helper."""

    def subject() -> None:
        """Call the class member through a closure binding."""

        Owner.helper()

    capture = capture_inspection(subject)
    original = Owner.__dict__["helper"]
    Owner.helper = staticmethod(lambda: None)
    try:
        with pytest.raises(InvalidTargetError, match="changed"):
            capture.validate()
    finally:
        Owner.helper = original


def test_capture_drift_guard_rejects_callable_instance_call_rebinding(
) -> None:
    """Reject a new raw ``__call__`` binding without invoking either body."""

    calls: list[str] = []

    class Owner:
        """Provide a callable instance with a mutable raw call descriptor."""

        def __call__(self) -> None:
            """Record invocation only if validation incorrectly executes it."""

            calls.append("original")

    def replacement(self) -> None:
        """
        Record invocation only if validation follows the rebound descriptor.
        """

        del self
        calls.append("replacement")

    capture = capture_inspection(Owner())
    original = Owner.__dict__["__call__"]
    Owner.__call__ = replacement
    try:
        with pytest.raises(InvalidTargetError, match="changed"):
            capture.validate()
    finally:
        Owner.__call__ = original

    assert calls == []


def test_capture_callable_instance_guard_uses_no_instance_hooks() -> None:
    """
    Read native type slots without ``__class__``, equality, or hash hooks.
    """

    class Hooked:
        """Reject every instance hook unnecessary for passive validation."""

        def __getattribute__(self, name: str) -> object:
            """
            Reject instance reflection while allowing no validation access.
            """

            raise AssertionError(f"instance hook accessed {name}")

        def __hash__(self) -> int:
            """Reject hash-based callable-instance indexing."""

            raise AssertionError("instance hash hook accessed")

        def __eq__(self, other: object) -> bool:
            """Reject equality-based callable-instance indexing."""

            del other
            raise AssertionError("instance equality hook accessed")

        def __call__(self) -> None:
            """Provide the admitted raw Python call implementation."""

    capture_inspection(Hooked()).validate()


def test_capture_marks_ast_fact_and_mro_limits_incomplete() -> None:
    """Parser, call-fact, and class exhaustion retain a root record."""

    from dryml.code import SourceTarget

    deep = SourceTarget(
        "def subject():\n    return " + "nested(" * 129 + "value" + ")" * 129,
        name="subject",
    )
    flood = SourceTarget(
        "def subject():\n" + "    helper()\n" * 16_385,
        name="subject",
    )
    bases: list[type] = [type("Base", (), {})]
    for index in range(256):
        bases.append(type(f"Layer{index}", (bases[-1], ), {}))

    deep_capture = capture_inspection(deep)
    flood_capture = capture_inspection(flood)
    mro_capture = capture_inspection(bases[-1])

    assert deep_capture.target.snapshot.record("t0000").incomplete
    assert flood_capture.target.snapshot.record("t0000").incomplete
    assert len(flood_capture.target.snapshot.record("t0000").calls) == 16_384
    assert mro_capture.target.snapshot.record("t0000").incomplete


def test_snapshot_metadata_identity_includes_every_field() -> None:
    """Detached records reject malformed metadata and hash every field."""

    info = TargetInfo("function", "subject", None, None, None, None, None,
                      None, None, None)
    first = InspectionSnapshot(1, "t0000",
                               (InspectionRecord("t0000", info, ()), ))
    changed = TargetInfo(
        "function",
        "subject",
        None,
        "other",
        None,
        None,
        None,
        None,
        None,
        None,
    )
    second = InspectionSnapshot(1, "t0000",
                                (InspectionRecord("t0000", changed, ()), ))

    assert first.identity != second.identity
    with pytest.raises(ValueError):
        InspectionRecord("t0000", object(), ())  # type: ignore[arg-type]


def test_capture_enforces_aggregate_call_budget_before_snapshot_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Candidate closures enforce one aggregate fact budget."""

    import dryml.code.inspection as inspection

    monkeypatch.setattr(inspection, "_MAX_CALLS", 2)

    def first() -> None:
        """Consume the first call fact."""

        _helper()

    def second() -> None:
        """Consume the second and overflowing call facts."""

        _helper()
        _helper()

    def subject() -> None:
        """Reach both helpers into one captured closure."""

        first()
        second()

    capture = capture_inspection(subject)

    assert (sum(
        len(record.calls) for record in capture.target.snapshot.records) <= 2)
    assert any(record.incomplete for record in capture.target.snapshot.records)
