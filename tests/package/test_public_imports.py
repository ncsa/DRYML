"""Fresh-process public import contracts documented by Sprint 10."""

from __future__ import annotations

import subprocess
import sys


HEAVY_FRAMEWORK_TOP_LEVEL_MODULES = frozenset({"jax", "ray", "tensorflow", "torch"})


def test_explicit_code_import_has_public_all_and_no_dispatch_coupling():
    """The supported explicit import remains lightweight and self-describing."""

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import dryml.code as code, sys; "
            "assert 'analyze' in code.__all__; "
            "assert 'trace' in code.__all__; "
            "assert 'dryml.dispatch' not in sys.modules; "
            f"assert not {{name.split('.', 1)[0] for name in sys.modules}} & {HEAVY_FRAMEWORK_TOP_LEVEL_MODULES!r}",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
