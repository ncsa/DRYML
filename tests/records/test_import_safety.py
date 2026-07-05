import os
from pathlib import Path
import subprocess
import sys


def run_probe(code: str):
    src_dir = Path(__file__).resolve().parents[2] / "src"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(src_dir), env.get("PYTHONPATH", "")])
    return subprocess.run([sys.executable, "-c", code], check=True, text=True, capture_output=True, env=env)


def test_records_import_is_lightweight():
    run_probe(
        """
import sys
import dryml.records
assert "dryml.core2" not in sys.modules
assert "tensorflow" not in sys.modules
assert "torch" not in sys.modules
assert "jax" not in sys.modules
assert "ray" not in sys.modules
assert "dryml." + "execute" not in sys.modules
assert "dryml." + "context" not in sys.modules
assert "dryml.environments" not in sys.modules
        """
    )


def test_top_level_dryml_records_export_is_lazy():
    run_probe(
        """
import sys
import dryml
assert "dryml.records" not in sys.modules
_ = dryml.records
assert "dryml.records" in sys.modules
assert "torch" not in sys.modules
assert "tensorflow" not in sys.modules
        """
    )


def test_scanner_and_index_imports_are_lightweight():
    run_probe(
        """
import sys
import dryml.records.scanner
import dryml.records.index
for name in ["dryml.core2", "dryml." + "execute", "dryml." + "context", "dryml.environments", "tensorflow", "torch", "jax", "ray"]:
    assert name not in sys.modules, name
        """
    )


def test_policy_and_export_imports_are_lightweight():
    run_probe(
        """
import sys
import dryml.records.policy
import dryml.records.export
for name in ["dryml.core2", "dryml.operations", "dryml." + "execute", "dryml." + "context", "dryml.environments", "tensorflow", "torch", "jax", "ray"]:
    assert name not in sys.modules, name
        """
    )


def test_dispatch_import_is_lightweight():
    run_probe(
        """
import sys
import dryml.dispatch
for name in ["dryml.core2", "dryml." + "execute", "dryml." + "context", "dryml.environments", "tensorflow", "torch", "jax", "ray"]:
    assert name not in sys.modules, name
        """
    )
