"""Fresh-process smoke coverage for the tracked Sprint 10 Python examples."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


EXAMPLES = (
    "examples.requirements.requirements_and_explain",
    "examples.dispatch.python_shaped_dispatch",
    "examples.code_analysis.static_and_dynamic_analysis",
)


def _copy_example_tree(destination: Path) -> None:
    """Copy only the exact tracked documentation examples into *destination*."""

    repository = Path(__file__).resolve().parents[2]
    for module_name in EXAMPLES:
        relative = Path(*module_name.split(".")).with_suffix(".py")
        source = repository / relative
        assert source.is_file(), f"missing tracked example: {relative}"
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        package = target.parent
        while package != destination:
            (package / "__init__.py").touch()
            package = package.parent


@pytest.mark.parametrize("module_name", EXAMPLES)
def test_documentation_python_examples_run_as_isolated_modules(tmp_path, module_name):
    """Run one copied example without exposing the repository examples tree."""

    example_root = tmp_path / "example-tree"
    _copy_example_tree(example_root)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(example_root)
    completed = subprocess.run(
        [sys.executable, "-m", module_name],
        cwd=tmp_path,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, (
        f"{module_name} failed\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
