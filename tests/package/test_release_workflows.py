"""Validate release workflow triggers, trust boundaries, and publication order."""

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"


def _load_workflow(name: str) -> dict[str, object]:
    """Load one workflow with YAML scalars preserved as strings."""

    value = yaml.load((WORKFLOWS / name).read_text(encoding="ascii"), Loader=yaml.BaseLoader)
    assert isinstance(value, dict)
    return value


def test_rehearsal_uses_versioned_test_tag_and_testpypi_oidc() -> None:
    """Require a checked wheel and an environment-scoped TestPyPI publisher."""

    workflow = _load_workflow("release-rehearsal.yaml")
    assert workflow["on"]["push"]["tags"] == ["test-v*"]
    build = workflow["jobs"]["build"]
    build_commands = "\n".join(
        step.get("run", "") for step in build["steps"] if isinstance(step, dict)
    )
    assert '"test-v$version"' in build_commands
    assert "python -m build --wheel" in build_commands
    assert "python -m twine check" in build_commands

    publish = workflow["jobs"]["publish-testpypi"]
    assert publish["needs"] == "build"
    assert publish["environment"]["name"] == "testpypi"
    assert publish["permissions"] == {"id-token": "write"}
    action = publish["steps"][-1]
    assert action["uses"] == "pypa/gh-action-pypi-publish@release/v1"
    assert action["with"]["repository-url"] == "https://test.pypi.org/legacy/"


def test_official_release_publishes_before_creating_github_release() -> None:
    """Require production OIDC publication before exposing the GitHub Release."""

    workflow = _load_workflow("release.yaml")
    assert workflow["on"]["push"]["tags"] == ["v*"]
    publish = workflow["jobs"]["publish-pypi"]
    assert publish["needs"] == "build"
    assert publish["environment"]["name"] == "pypi"
    assert publish["permissions"] == {"id-token": "write"}
    assert publish["steps"][-1]["uses"] == "pypa/gh-action-pypi-publish@release/v1"

    release = workflow["jobs"]["github-release"]
    assert release["needs"] == "publish-pypi"
    assert release["permissions"] == {"contents": "write"}
    command = release["steps"][-1]["run"]
    assert 'gh release create "$GITHUB_REF_NAME" dist/*.whl' in command
