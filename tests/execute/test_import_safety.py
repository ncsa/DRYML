"""Fresh-interpreter import boundaries for generic Execute."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


_SCRIPT = r'''
import importlib
import importlib.abc
import sys

blocked = ("dryml.core", "dryml.managed", "dryml.dispatch", "dryml.session", "dryml.runtime", "ray", "tensorflow", "torch", "jax")

class BlockedImport(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "dryml.execute.ray":
            return None
        if any(fullname == prefix or fullname.startswith(prefix + ".") for prefix in blocked):
            raise ImportError("blocked import: " + fullname)
        return None

sys.meta_path.insert(0, BlockedImport())
importlib.import_module(sys.argv[1])
assert not any(name == prefix or name.startswith(prefix + ".") for name in sys.modules for prefix in blocked)
if sys.argv[1] == "dryml.execute":
    assert "dryml.execute.subprocess" not in sys.modules
    assert "dryml.execute.ray" not in sys.modules
'''


@pytest.mark.parametrize("module", ["dryml.execute", "dryml.execute.subprocess", "dryml.execute.ray"])
def test_execute_imports_are_safe_in_a_fresh_blocked_interpreter(module):
    """Common and specialization imports avoid core/runtime and optional SDK imports."""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(path for path in sys.path if path)
    result = subprocess.run(
        [sys.executable, "-c", _SCRIPT, module],
        cwd=Path(__file__).resolve().parents[2],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
