import os
from pathlib import Path
import subprocess
import sys


def run_probe(code: str):
    src_dir = Path(__file__).resolve().parents[2] / "src"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(src_dir), env.get("PYTHONPATH", "")])
    return subprocess.run([sys.executable, "-c", code], check=True, text=True, capture_output=True, env=env)


def test_operations_import_is_lightweight():
    run_probe(
        """
import sys
import dryml.operations
for name in ["dryml.core2", "dryml.execute", "dryml.context", "dryml.environments", "tensorflow", "torch", "jax", "ray"]:
    assert name not in sys.modules, name
        """
    )


def test_top_level_operations_export_is_lazy():
    run_probe(
        """
import sys
import dryml
assert "dryml.operations" not in sys.modules
_ = dryml.operations
assert "dryml.operations" in sys.modules
assert "dryml.core2" not in sys.modules
        """
    )
