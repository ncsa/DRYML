"""Import-light guards around the framework interception boundary."""

from __future__ import annotations

import subprocess
import sys


def test_base_and_registry_imports_do_not_import_watched_roots():
    """Metadata setup remains dependency-light in a fresh interpreter."""
    completed = subprocess.run(
        [sys.executable, "-c", "import sys, dryml, dryml.worlds, dryml.core, dryml.runtime.frameworks; assert not {'tensorflow', 'torch', 'jax', 'jaxlib'} & set(sys.modules)"],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
