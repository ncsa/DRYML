"""Cross-domain root-alias and source import-isolation contracts."""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import dryml

_ROOT = Path(__file__).resolve().parents[2]
_SOURCE = _ROOT / "src"
_FORBIDDEN = (
    "dryml.artifacts",
    "dryml.context",
    "dryml.core",
    "dryml.dispatch",
    "dryml.execute",
    "dryml.runtime",
    "dryml.session",
    "tensorflow",
    "torch",
    "jax",
    "jaxlib",
    "ray",
)

_REQUIREMENTS_EXPORTS = {
    "AdmissionReport",
    "RequirementBarrierError",
    "RequirementCombinationError",
    "RequirementCombiner",
    "RequirementDeclaration",
    "RequirementError",
    "RequirementIssue",
    "RequirementReport",
    "RequirementResult",
    "RequirementSource",
    "combine_requirements",
    "require_admission",
}


def test_root_aliases_have_exact_owner_surfaces_and_no_singular_packages() -> None:
    """Expose lazy plural-owner aliases without defaults or implementation packages."""

    from dryml import env, world

    assert env is dryml.environments
    assert world is dryml.worlds
    assert dryml.requirements.__all__ == sorted(_REQUIREMENTS_EXPORTS)
    assert "ENVIRONMENT_REQUIREMENT_KEY" not in env.__all__
    assert "WORLD_REQUIREMENT_KEY" not in world.__all__
    for owner in (env, world):
        assert {"current", "set_current", "reset_current", "use"} <= set(owner.__all__)
        assert not {"default", "default_for", "set_default", "reset_default", "use_default"} & set(owner.__all__)
    assert "default" not in dryml.runtime.__all__
    for module in ("dryml.env", "dryml.world", "dryml.runtime.default"):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(module)
        assert importlib.util.find_spec(module) is None


def test_root_aliases_resolve_domain_annotations_independently() -> None:
    """One target's environment and world declarations remain domain-local."""

    @dryml.env.req(tags=("environment",))
    @dryml.world.req(cpus=2)
    def target() -> None:
        return None

    environment = dryml.env.requirements_for(target)
    world = dryml.world.requirements_for(target)

    assert environment.value is not None
    assert environment.value.tags == ("environment",)
    assert world.value is not None
    assert world.value.roles["main"].resources.cpus.min == 2


def test_cross_domain_unsupported_collection_and_merge_fail_closed() -> None:
    """Domain boundaries reject unsupported targets and cross-domain values unchanged."""

    target = object()
    with pytest.raises(dryml.env.EnvironmentRequirementError):
        dryml.env.requirements_for(target)
    with pytest.raises(dryml.world.WorldRequirementError):
        dryml.world.requirements_for(target)

    environment = dryml.env.EnvironmentRequirement(tags=("environment",))
    world = dryml.world.WorldRequirement({"main": {"resources": {"cpus": 1}}})
    with pytest.raises(dryml.env.EnvironmentRequirementError):
        environment.merge(world)
    with pytest.raises(dryml.world.WorldSpecValidationError):
        world.merge(environment)
    assert environment.tags == ("environment",)
    assert world.roles["main"].resources.cpus == dryml.world.CountConstraint(1, 1)


@pytest.mark.parametrize(
    ("action", "inverse"),
    [
        pytest.param("import dryml.requirements", ("dryml.environments", "dryml.worlds"), id="requirements"),
        pytest.param("assert dryml.env is dryml.environments", ("dryml.worlds",), id="env"),
        pytest.param("assert dryml.world is dryml.worlds", ("dryml.environments",), id="world"),
        pytest.param("from dryml import env, world\nassert env is dryml.environments\nassert world is dryml.worlds", (), id="both"),
    ],
)
def test_fresh_source_root_entry_points_are_effect_free_and_isolated(
    action: str, inverse: tuple[str, ...]
) -> None:
    """Fresh source processes keep root entry points lazy and domain-local."""

    script = f"""
import json
import os
import platform
import socket
import subprocess
import sys

effects = []
def blocked(name):
    def call(*args, **kwargs):
        effects.append(name)
        raise AssertionError(f"unexpected {{name}}")
    return call

os.cpu_count = blocked("cpu_count")
platform.system = blocked("platform.system")
socket.socket = blocked("socket.socket")
subprocess.Popen = blocked("subprocess.Popen")

import dryml
{action}

forbidden = {_FORBIDDEN!r} + {inverse!r}
print(json.dumps({{
    "effects": effects,
    "loaded": sorted(
        name for name in sys.modules
        if any(name == prefix or name.startswith(prefix + ".") for prefix in forbidden)
    ),
}}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd="/tmp/dryml",
        env={**os.environ, "PYTHONPATH": str(_SOURCE)},
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {"effects": [], "loaded": []}
