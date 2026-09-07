"""Import and ownership-boundary tests for :mod:`dryml.locking`."""

from __future__ import annotations

import ast
import os
from pathlib import Path
import subprocess
import sys


def test_locking_import_is_dependency_light_and_root_export_is_lazy():
    """Loading the lock owner neither imports consumers nor makes root eager."""

    root = Path(__file__).resolve().parents[2]
    env = os.environ | {"PYTHONPATH": str(root / "src")}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json, sys; import dryml; assert 'dryml.locking' not in sys.modules; "
            "assert dryml.locking.__name__ == 'dryml.locking'; "
            "print(json.dumps(sorted(name for name in sys.modules if name.startswith("
            "('dryml.core', 'dryml.managed', 'dryml.runtime', 'dryml.execute', 'tensorflow', 'torch', 'jax', 'ray')))))",
        ],
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )
    assert result.stdout.strip() == "[]"


def test_native_file_lock_calls_have_one_framework_owner():
    """Only the shared module may import native advisory-lock adapters."""

    source_root = Path(__file__).resolve().parents[2] / "src" / "dryml"
    offenders = []
    for path in source_root.rglob("*.py"):
        if path == source_root / "locking.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) and any(alias.name in {"fcntl", "msvcrt"} for alias in node.names):
                offenders.append(str(path.relative_to(source_root)))
            if isinstance(node, ast.ImportFrom) and node.module in {"fcntl", "msvcrt"}:
                offenders.append(str(path.relative_to(source_root)))
    assert offenders == []
