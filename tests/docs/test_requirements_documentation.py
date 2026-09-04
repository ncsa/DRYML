"""Keep Stage 4 hard-requirement documentation aligned with public behavior."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _guide(name: str) -> str:
    """Return one UTF-8 documentation guide from the repository root."""

    return (ROOT / "docs" / name).read_text(encoding="utf-8")


def test_requirements_guide_describes_shared_contract() -> None:
    """Require the shared declaration, result, admission, and boundary contract."""

    guide = _guide("requirements.md")
    for phrase in (
        "RequirementSource",
        "RequirementDeclaration",
        "RequirementIssue",
        "RequirementReport",
        "RequirementResult",
        "RequirementCombiner",
        "combine_requirements",
        "AdmissionReport",
        "require_admission",
        "EnvironmentRequirement.check",
        "CompatibilityReport",
        "check_world_spec_satisfies_requirement",
        "check_allocation_satisfies_requirement",
        "WorldCompatibilityReport",
        "empty success",
        "valued success",
        "conflict failure",
        "admission_ok",
        "policy-dependent `ok`",
        "256 declarations",
        "1,024",
        "4,096",
        "source",
        "redact",
        "process-local",
        "defaults",
        "selection",
        "runtime",
        "session",
        "dispatch",
        "automatic enforcement",
        "code inference",
    ):
        assert phrase in guide, phrase


def test_domain_guides_describe_passive_independent_declarations() -> None:
    """Require environment and world guides to preserve their domain boundaries."""

    environments = _guide("environments.md")
    worlds = _guide("world_runtime.md")
    for phrase in (
        "req(...)",
        "requirements_for",
        "requirements_for_method",
        "hard requirement",
        "passive",
        "RequirementResult",
        "admission_ok",
        "independent",
        "default",
        "dispatch",
        "automatic enforcement",
        "fragment",
        "record IDs",
    ):
        assert phrase in environments, phrase
    for phrase in (
        "dryml.worlds.req",
        "requirements_for",
        "requirements_for_method",
        "RequirementResult",
        "unconstrained",
        "omitted",
        "roles",
        "admission_ok",
        "independent",
        "automatic enforcement",
    ):
        assert phrase in worlds, phrase
    for retired in ("add_req", "override_req", "RequirementFragment", "requirements_for_class"):
        assert retired not in environments


def test_cross_guide_boundaries_and_release_notes_match_stage_four() -> None:
    """Require the passive, lazy, fragment-drop, and release boundaries to agree."""

    annotations = _guide("annotations.md")
    context = _guide("context.md")
    session = _guide("session.md")
    runtime = _guide("world_runtime.md")
    release_notes = _guide("release_notes.md")
    toc = _guide("table_of_content.md")
    testing = _guide("testing.md")

    for guide in (annotations, context, session, runtime):
        for phrase in ("process-local", "session", "runtime"):
            assert phrase in guide, phrase
    for phrase in ("dryml.env", "dryml.world", "lazy", "not importable"):
        assert phrase in annotations, phrase
    for phrase in (
        "0.3.0.dev2",
        "dryml.requirements",
        "dryml.env",
        "dryml.world",
        "RequirementSource",
        "RequirementDeclaration",
        "RequirementIssue",
        "RequirementReport",
        "RequirementResult",
        "RequirementCombiner",
        "combine_requirements",
        "AdmissionReport",
        "require_admission",
        "EnvironmentRequirement.check",
        "CompatibilityReport.admission_ok",
        "check_world_spec_satisfies_requirement",
        "check_allocation_satisfies_requirement",
        "WorldCompatibilityReport.admission_ok",
        "add_req",
        "override_req",
        "RequirementFragment",
        "environment_fragment",
        "record IDs",
        "no compatibility",
    ):
        assert phrase in release_notes, phrase
    assert "[Hard Requirements](requirements.md)" in toc
    assert "profile --unknown-only" in testing
