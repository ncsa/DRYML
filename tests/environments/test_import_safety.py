import os
from pathlib import Path
import subprocess
import sys


def run_probe(code: str):
    src_dir = Path(__file__).resolve().parents[2] / "src"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(src_dir), env.get("PYTHONPATH", "")])
    return subprocess.run([sys.executable, "-c", code], check=True, text=True, capture_output=True, env=env)


def test_import_dryml_environments_is_lightweight():
    run_probe(
        """
import sys
assert "dryml.context" not in sys.modules
assert "dryml.core" not in sys.modules
assert "dryml.execute" not in sys.modules
assert "tensorflow" not in sys.modules
assert "torch" not in sys.modules
assert "jax" not in sys.modules
assert "ray" not in sys.modules
import dryml.environments as envs
assert envs.ENVIRONMENT_RECORD_SCHEMA_VERSION == "1.1"
assert "dryml.context" not in sys.modules
assert "dryml.core" not in sys.modules
assert "dryml.execute" not in sys.modules
assert "tensorflow" not in sys.modules
assert "torch" not in sys.modules
assert "jax" not in sys.modules
assert "ray" not in sys.modules
        """
    )


def test_environment_requirements_do_not_import_worlds():
    run_probe(
        """
import sys
import dryml.environments as envs
@envs.req(requirements=("dryml>=0.3",))
class Target:
    pass
result = envs.requirements_for(Target)
assert result.has_value
assert "dryml.worlds" not in sys.modules
        """
    )


def test_environment_kernel_is_lazy_and_loads_no_inverse_domain():
    run_probe("""
import sys
import dryml.environments as envs
assert "dryml.code" not in sys.modules
assert "dryml.worlds" not in sys.modules
assert envs.EnvironmentRequirementsKernel.__name__"""
              ''' == "EnvironmentRequirementsKernel"
assert "dryml.code" in sys.modules
assert "dryml.worlds" not in sys.modules
        ''')


def test_selection_first_preserves_the_public_probe_callable():
    run_probe(
        """
import sys
import dryml.environments as envs
selection = envs.resolve_environment_spec(
    envs.PythonExecutableSpec(sys.executable)
)
assert selection.command
assert callable(envs.probe)
assert envs.probe(envs.CurrentEnvironmentSpec()).ok
        """
    )


def test_sibling_exports_load_before_the_public_probe_callable():
    run_probe(
        """
import dryml.environments as envs
assert callable(envs.resolve_environment_spec)
assert callable(envs.compare_selection)
assert callable(envs.probe)
assert envs.probe(envs.CurrentEnvironmentSpec()).ok
        """
    )


def test_top_level_dryml_submodules_are_lazy_but_accessible():
    run_probe(
        """
import sys
import dryml
assert "dryml.context" not in sys.modules
assert "dryml.core" not in sys.modules
_ = dryml.environments
assert "dryml.environments" in sys.modules
assert "dryml.context" not in sys.modules
_ = dryml.core
assert "dryml.core" in sys.modules
        """
    )


def test_probe_worker_imports_only_lightweight_environment_modules():
    run_probe(
        """
import sys
from dryml.environments.probe_worker import main
assert "dryml.context" not in sys.modules
assert "tensorflow" not in sys.modules
assert "torch" not in sys.modules
assert "jax" not in sys.modules
        """
    )
