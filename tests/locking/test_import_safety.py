"""Import and ownership-boundary tests for :mod:`dryml.locking`."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

from tests.tools.native_lock_audit import native_advisory_lock_offenders


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
    """Only the shared module may use native advisory-lock adapters."""

    source_root = Path(__file__).resolve().parents[2] / "src" / "dryml"
    native_lock_sources = set()
    for path in source_root.rglob("*.py"):
        if native_advisory_lock_offenders(path.read_bytes(), filename=str(path)):
            native_lock_sources.add(str(path.relative_to(source_root)))
    assert native_lock_sources == {"locking.py"}


def test_native_lock_guard_permits_only_windows_handle_conversion():
    """Allow the non-locking Windows file-descriptor handle conversion."""

    source = (
        "import msvcrt\n"
        "handle = msvcrt.get_osfhandle(fd)\n"
        "description = 'msvcrt.locking remains reserved for dryml.locking'\n"
        "# import fcntl\n"
    )

    assert native_advisory_lock_offenders(source) == []
    assert native_advisory_lock_offenders(source.encode()) == []


def test_native_lock_guard_rejects_windows_advisory_lock_escapes():
    """Reject direct, imported, and aliased native advisory-lock access."""

    sources = (
        "import msvcrt\nmsvcrt.locking(fd, operation, 1)\n",
        "from msvcrt import locking\nlocking(fd, operation, 1)\n",
        "from msvcrt import locking as native_lock\nnative_lock(fd, operation, 1)\n",
        "import msvcrt as native\nnative.locking(fd, operation, 1)\n",
        "import fcntl\nfcntl.flock(fd, operation)\n",
        "from fcntl import flock as native_lock\nnative_lock(fd, operation)\n",
    )

    for source in sources:
        assert native_advisory_lock_offenders(source)
