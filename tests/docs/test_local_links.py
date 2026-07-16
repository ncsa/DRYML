"""Bounded offline Markdown-link checks for Sprint 10 documentation."""

from __future__ import annotations

import re
from pathlib import Path


CHANGED_DOCUMENTS = (
    "README.md",
    "docs/table_of_content.md",
    "docs/annotations.md",
    "docs/dispatch.md",
    "docs/environments.md",
    "docs/world_runtime.md",
    "docs/context.md",
    "docs/operations.md",
    "docs/migration/legacy_context_execute_removal.md",
    "docs/release_notes.md",
    "docs/architecture/code_analysis.md",
    "docs/architecture/runtime_dispatch_requirements.md",
    "docs/architecture/environment_world_runtime_boundaries.md",
    "docs/architecture/annotation_requirement_collection.md",
    "docs/records.md",
    "docs/reporting.md",
)
LINK_RE = re.compile(r"(?<!!)\[[^]]*\]\(([^)\s]+)(?:\s+[^)]*)?\)")
HEADING_RE = re.compile(r"^#{1,6}\s+(.+?)\s*#*\s*$", re.MULTILINE)


def _anchor(value: str) -> str:
    """Approximate GitHub's stable simple Markdown heading anchor."""

    normalized = value.strip().lower()
    normalized = re.sub(r"[`*_]", "", normalized)
    normalized = re.sub(r"[^a-z0-9 -]", "", normalized)
    return re.sub(r"\s+", "-", normalized)


def _anchors(document: Path) -> set[str]:
    """Return the local anchors defined by a Markdown document."""

    return {_anchor(match.group(1)) for match in HEADING_RE.finditer(document.read_text(encoding="utf-8"))}


def test_changed_markdown_links_are_local_and_resolve():
    """Check changed docs only; never crawl external URLs or untracked paths."""

    repository = Path(__file__).resolve().parents[2]
    for relative in CHANGED_DOCUMENTS:
        document = repository / relative
        text = document.read_text(encoding="utf-8")
        for match in LINK_RE.finditer(text):
            target = match.group(1)
            if "://" in target or target.startswith(("mailto:", "#")):
                continue
            path_text, separator, fragment = target.partition("#")
            resolved = (document.parent / path_text).resolve() if path_text else document
            assert resolved.is_file(), f"{relative}: missing local link target {target!r}"
            assert resolved.is_relative_to(repository), f"{relative}: target escapes repository: {target!r}"
            if separator:
                assert fragment in _anchors(resolved), f"{relative}: missing anchor {fragment!r} in {target!r}"
