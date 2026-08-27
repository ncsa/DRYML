"""Annotation imports and resolution are passive with respect to runtime state."""

import subprocess
import sys

import dryml
from dryml.annotations import resolve_target_requirements


def test_root_sugar_is_lazy_and_runtime_default_is_passive():
    assert dryml.env.req is not None
    assert dryml.world.req is not None
    assert dryml.runtime.default is not None


def test_domain_sugar_builds_declarations_without_runtime_effects():
    @dryml.env.req(requirements=("packaging>=20",))
    @dryml.world.req(cpus=1)
    @dryml.world.default(cpus=1)
    @dryml.runtime.default(limits={"threads": 1})
    class Subject:
        pass

    result = resolve_target_requirements(Subject)
    assert result.usable
    assert result.environment_requirement.requirements == ("packaging>=20",)
    assert result.world_requirement.roles["main"].resources.cpus.min == 1
    assert result.world_default.roles["main"].process.resources.cpus == 1
    assert result.runtime_default["limits"]["threads"] == 1


def test_fresh_annotation_imports_leave_optional_frameworks_and_session_unchanged():
    """Declaration modules perform no optional import, inventory, or publication."""

    script = """
import sys
import dryml
before = dryml.session.current().generation
import dryml.annotations
assert dryml.session.current().generation == before
assert not {'tensorflow', 'torch', 'jax', 'jaxlib'} & set(sys.modules)
"""
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
