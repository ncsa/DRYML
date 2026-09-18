"""Keep Dispatch guides, public shapes, and ownership claims synchronized."""

from __future__ import annotations

from dataclasses import fields
import inspect
from pathlib import Path

import dryml.dispatch as dispatch
from dryml.dispatch import DispatchReport, DispatchView, ProbeOptions


ROOT = Path(__file__).resolve().parents[2]


def _guide(name: str) -> str:
    """Return one repository guide as UTF-8 text."""

    return (ROOT / "docs" / name).read_text(encoding="utf-8")


def test_dispatch_guide_documents_actual_defaults_and_boundaries() -> None:
    """Require explicit-placement and bounded-probe guide policy."""

    guide = " ".join(_guide("dispatch.md").split())
    defaults = ProbeOptions()
    for name, rendered in (
        ("placement", 'placement="auto"'),
        (
            "execution_timeout",
            f"execution_timeout={defaults.execution_timeout}",
        ),
        ("max_targets", f"max_targets={defaults.max_targets}"),
        ("max_depth", f"max_depth={defaults.max_depth}"),
    ):
        assert getattr(defaults, name) is not None
        assert rendered in guide
    for phrase in (
        "`BackendChoice` accepts",
        "no backend fallback",
        "no resource growth",
        "Stage 8A retry/resume",
        "4 MiB",
        "64 diagnostic entries",
        "512 characters",
        "1 MiB",
        "8 MiB",
        "16,384",
        "4,096",
        "__wrapped__",
        "__signature__",
        "class-only non-shadowable proof",
        "point-in-time evidence",
        "source may already differ from loaded code",
        "external environment can change after admission",
        "coroutine, generator, or async-generator",
        "never advances or awaits",
        "same Ray configuration",
        "lost-response reconciliation",
    ):
        assert phrase in guide, phrase


def test_dispatch_public_surface_and_dataclass_fields_are_closed() -> None:
    """Keep the source guide aligned with the supported Dispatch surface."""

    assert set(dispatch.__all__) == {
        "BackendChoice", "DispatchCoverageWarning", "DispatchError",
        "DispatchReport", "DispatchView", "InProcess", "ProbeOptions",
        "backends", "explain", "register_backend", "run",
        "set_execute_backend_default", "set_probe_default",
        "set_worker_environment_default", "set_worker_python_default",
        "set_worker_world_default", "submit", "unregister_backend",
        "with_options",
    }
    assert [field.name for field in fields(ProbeOptions)] == [
        "placement", "backend", "environment", "world", "environment_spec",
        "execution_timeout", "max_targets", "max_depth",
    ]
    assert [field.name for field in fields(DispatchReport)] == [
        "workload_placement", "workload_backend", "supported_methods",
        "probe_placement", "probe_backend", "probe_reason", "coverage",
        "environment", "world", "eligible", "diagnostics", "warnings",
    ]
    for operation in (
        dispatch.explain, dispatch.run, dispatch.submit, dispatch.with_options,
        DispatchView.explain, DispatchView.run, DispatchView.submit,
        DispatchView.with_options,
    ):
        assert inspect.getdoc(operation)


def test_dispatch_docs_link_execution_and_current_generation_owners() -> None:
    """Reject guides that lose exact-pin, lease, or owner boundaries."""

    toc = _guide("table_of_content.md")
    execute = _guide("execute.md")
    environments = _guide("environments.md")
    signatures = _guide("signatures.md")
    session = _guide("session.md")
    world = _guide("world_runtime.md")
    release_notes = _guide("release_notes.md")
    testing = _guide("testing.md")
    assert "[Dispatch](dispatch.md)" in toc
    for text, phrase in (
        (execute, "separate probe and workload"),
        (environments, "existing-environment pin"),
        (signatures, "passive static capture"),
        (session, "publication lease"),
        (world, "does not reserve, widen, or grow"),
        (release_notes, "dryml.dispatch"),
        (testing, "Ray-probe/in-process"),
    ):
        assert phrase in text, phrase
