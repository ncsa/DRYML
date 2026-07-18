"""Bounded offline Markdown-link checks for release-facing documentation."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


CHANGED_DOCUMENTS = (
    "README.md",
    "docs/table_of_content.md",
    "docs/models.md",
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
    "docs/representations_adapters.md",
)
TUTORIAL_NOTEBOOKS = (
    "examples/notebooks/objects_definitions_and_repos.ipynb",
    "examples/notebooks/datasets_and_transforms.ipynb",
    "examples/notebooks/local_defaults_and_plain_mode.ipynb",
    "examples/notebooks/models_experiments_and_metrics.ipynb",
    "examples/notebooks/definition_driven_experiments.ipynb",
    "examples/notebooks/local_hyperparameter_search.ipynb",
)
LINK_RE = re.compile(r"!?\[[^]]*\]\(([^)\s]+)(?:\s+[^)]*)?\)")
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
    """Check bounded docs and images without crawling external URLs."""

    repository = Path(__file__).resolve().parents[2]
    tracked = {
        line
        for line in subprocess.check_output(
            ["git", "ls-files"], cwd=repository, text=True,
        ).splitlines()
    }
    for relative in CHANGED_DOCUMENTS:
        assert relative in tracked, f"documentation input is not tracked: {relative}"
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
            assert resolved.relative_to(repository).as_posix() in tracked, (
                f"{relative}: local link target is not tracked: {target!r}"
            )
            if separator:
                assert fragment in _anchors(resolved), f"{relative}: missing anchor {fragment!r} in {target!r}"


def test_tutorial_notebooks_are_linked_in_canonical_order_and_tracked():
    """Keep the documentation index and release-facing paths in lockstep."""

    repository = Path(__file__).resolve().parents[2]
    document = repository / "docs/table_of_content.md"
    tutorial_links = tuple(
        (document.parent / match.group(1)).resolve().relative_to(repository).as_posix()
        for match in LINK_RE.finditer(document.read_text(encoding="utf-8"))
        if match.group(1).startswith("../examples/notebooks/")
    )
    tracked = set(subprocess.check_output(
        ["git", "ls-files"], cwd=repository, text=True,
    ).splitlines())

    assert tutorial_links == TUTORIAL_NOTEBOOKS
    assert set(TUTORIAL_NOTEBOOKS) <= tracked
