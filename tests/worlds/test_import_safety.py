"""Tests for the import-light world declaration and combination surface."""

import json
import subprocess
import sys


def test_world_requirements_are_environment_independent_and_effect_free() -> None:
    """World declaration resolution imports no inverse domain or active backend."""

    script = """
import json
import sys
import dryml.worlds as worlds
@worlds.req(cpus=1)
class Target:
    pass
assert worlds.requirements_for(Target).has_value
forbidden = ('dryml.environments', 'dryml.execute', 'dryml.runtime', 'dryml.session', 'tensorflow', 'torch', 'jax', 'jaxlib', 'ray')
print(json.dumps(sorted(name for name in sys.modules if name in forbidden)))
"""
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == []
