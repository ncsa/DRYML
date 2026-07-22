from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


FIXTURE_DIR = Path(__file__).parents[1] / "fixtures"
if str(FIXTURE_DIR) not in sys.path:
    sys.path.insert(0, str(FIXTURE_DIR))


@pytest.fixture(scope="session")
def requirement_targets():
    """Load reusable requirement targets under an importable module name."""

    path = Path(__file__).parents[1] / "fixtures" / "requirements_targets.py"
    spec = importlib.util.spec_from_file_location("dryml_requirement_targets", path)
    if spec.name in sys.modules:
        return sys.modules[spec.name]
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
