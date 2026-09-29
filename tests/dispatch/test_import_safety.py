"""Import-safety tests for the Dispatch facade."""

from __future__ import annotations

import subprocess
import sys


def test_dispatch_import_does_not_load_optional_frameworks() -> None:
    """Keep configuration import free of optional execution implementations."""

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import dryml.dispatch; "
            "assert not set(sys.modules) & "
            "{'ray', 'tensorflow', 'torch', 'jax'}",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
