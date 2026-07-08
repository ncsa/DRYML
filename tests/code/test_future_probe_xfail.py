from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

import dryml.code as code



pytestmark = pytest.mark.future_behavior



def _load_targets():
    path = Path(__file__).parents[1] / "fixtures" / "requirements_targets.py"
    spec = importlib.util.spec_from_file_location("dryml_requirement_targets", path)
    if spec.name in sys.modules:
        return sys.modules[spec.name]
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


targets = _load_targets()


@pytest.mark.xfail(reason="Sprint 5: lightweight code probe worker not implemented yet", strict=True)
def test_code_probe_target_exists():
    assert callable(code.probe_target)


@pytest.mark.xfail(reason="Sprint 5: lightweight code probe worker not implemented yet", strict=True)
def test_code_probe_returns_facts_without_world_allocation():
    result = code.probe_target(targets.plain_importable_function, include_environment=True)
    assert result.ok
    assert result.runtime.mode == "probe"
    assert result.allocation is None
    assert result.code_facts
    assert result.environment_record


@pytest.mark.xfail(reason="Sprint 5: structured code probe diagnostics are not implemented yet", strict=True)
def test_code_probe_has_structured_diagnostics_for_import_failure():
    result = code.probe_target("missing.module:target")
    assert not result.ok
    assert result.diagnostics[0].code == "import_failed"
