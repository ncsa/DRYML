"""Tests for the shared requirement package's dependency-light import closure."""

import json
import subprocess
import sys


def test_fresh_requirements_import_loads_only_annotations_and_stdlib():
    """Shared contracts do not initialize consumers, authority, or optional backends."""

    script = """
import json
import sys
import dryml.requirements
forbidden = (
    'dryml.core', 'dryml.environments', 'dryml.execute', 'dryml.formats',
    'dryml.runtime', 'dryml.session', 'dryml.worlds', 'tensorflow', 'torch',
    'jax', 'jaxlib', 'ray',
)
print(json.dumps(sorted(name for name in sys.modules if name in forbidden)))
"""
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == []


def test_documented_stage_four_lightweight_surfaces_do_not_load_optional_frameworks():
    """Each documented Stage 4 import surface remains independently framework-free."""

    actions = (
        "import dryml.annotations",
        "import dryml.requirements",
        "import dryml.environments",
        "import dryml.environments.introspection",
        "import dryml.environments.probe",
        "import dryml.worlds",
        "import dryml.worlds.allocation",
        "import dryml.worlds.resources",
        "import dryml.worlds.specs",
    )
    for action in actions:
        script = f"""
import json
import sys
{action}
forbidden = ('tensorflow', 'torch', 'jax', 'jaxlib', 'ray')
print(json.dumps(sorted(
    name for name in sys.modules
    if any(name == prefix or name.startswith(prefix + '.') for prefix in forbidden)
)))
"""
        completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        assert completed.returncode == 0, completed.stderr
        assert json.loads(completed.stdout) == []
