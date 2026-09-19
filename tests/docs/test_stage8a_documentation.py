"""Stage 8A documentation contracts for managed execution integration."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _guide(name: str) -> str:
    """Return one public Markdown guide as normalized UTF-8 text."""

    return " ".join((ROOT / "docs" / name).read_text(encoding="utf-8").split())


def test_stage8a_guides_cross_link_resource_and_execution_boundaries() -> None:
    """Keep Session, Store, managed, Execute, format, annotation, and signature guides aligned."""

    expected_links = {
        "session.md": ("[Repos and Stores](repos.md)", "[Generic Execute](execute.md)", "[Managed Operations](managed_operations.md)"),
        "repos.md": ("[Session](session.md)", "[Generic Execute](execute.md)"),
        "managed_operations.md": ("[Generic Execute](execute.md)", "[Repos and Stores](repos.md)"),
        "execute.md": ("[Managed Operations](managed_operations.md)", "[Session](session.md)"),
        "formats.md": ("[Repos and Stores](repos.md)", "[Session](session.md)"),
        "annotations.md": ("[Managed Operations](managed_operations.md)", "[Signatures](signatures.md)"),
        "signatures.md": ("[Generic Execute](execute.md)", "[Managed Operations](managed_operations.md)"),
    }

    for name, links in expected_links.items():
        guide = _guide(name)
        for link in links:
            assert link in guide, (name, link)


def test_stage8a_guides_preserve_implemented_transport_and_code_shift_limits() -> None:
    """Document detached trusted transport without claiming provisioning or code migration."""

    managed = _guide("managed_operations.md")
    execute = _guide("execute.md")
    annotations = _guide("annotations.md")
    limitation = _guide("solutions/architecture-patterns/2026-09-18-code-shift-in-long-lived-sessions.md")

    for phrase in (
        "detached invocation-graph", "does not inspect status", "rerun a managed method automatically",
    ):
        assert phrase in managed
    for phrase in (
        "does not inject a root `ManagedConfig`", "does not inspect managed status",
        "or make generic Execute responsible", "does not change generic Execute RPC/setup fields",
    ):
        assert phrase in execute
    assert "does not import or depend on any of them" in annotations
    for phrase in ("accepted limitation", "not automatic migration", "not an operation-long code-version guarantee"):
        assert phrase in limitation
