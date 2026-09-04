"""Keep the environment hard-requirement guide aligned with U2 behavior."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_environment_guide_describes_passive_hard_declarations() -> None:
    """Require the U2 declaration, conflict, and deferred-behavior boundaries."""

    guide = (ROOT / "docs" / "environments.md").read_text(encoding="utf-8")
    for phrase in ("requirements_for", "requirements_for_method", "hard requirement", "passive", "RequirementResult", "admission_ok", "default", "dispatch"):
        assert phrase in guide, phrase
    for retired in ("add_req", "override_req", "RequirementFragment", "requirements_for_class"):
        assert retired not in guide
