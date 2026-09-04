"""Keep the Stage 3 code-analysis guide and architecture guidance aligned."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_code_analysis_guide_covers_the_public_contract() -> None:
    """Require documentation of supported forms, boundaries, and lifecycle limits."""

    guide = " ".join(
        (ROOT / "docs" / "code_analysis.md").read_text(encoding="utf-8").split()
    )
    required = (
        "ImportTarget", "module:qualname", "SourceTarget", "static-only",
        "never reconstructed", "trusted", "ProgramGraph", "kernel DAG",
        "artifacts", "branch-aware control-flow", "data-flow", "alias",
        "whole-program", "redacted", "raise", "AnalysisResult", "require()",
        "facts", "fusion", "in-process", "1 through 100,000", "exactly once",
        "hook", "does not compose", "interruption", "cleanup", "same-process",
        "separate processes", "ephemeral", "nonserializable", "persistence",
        "core.symbol", "lexical", "transformation", "execute", "dispatch",
        "process isolation",
    )
    for phrase in required:
        assert phrase in guide, phrase


def test_code_analysis_docs_are_linked_and_methods_keep_ownership_boundary() -> None:
    """Require discoverability and consistent Method/deferred-work wording."""

    table_of_content = (ROOT / "docs" / "table_of_content.md").read_text(encoding="utf-8")
    methods = (ROOT / "docs" / "methods.md").read_text(encoding="utf-8")
    release_notes = (ROOT / "docs" / "release_notes.md").read_text(encoding="utf-8")
    assert "[Code Analysis](code_analysis.md)" in table_of_content
    assert "local analysis foundation" in methods
    assert "cross-process Method probing" in methods
    assert "transformation" in methods
    assert "Stage 3" in release_notes
    assert "dryml.code" in release_notes
