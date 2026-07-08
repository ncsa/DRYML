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


@pytest.mark.xfail(reason="Sprint 1: dryml.code.analyze fact API not implemented yet", strict=True)
def test_dryml_code_analyze_exists():
    assert callable(code.analyze)


@pytest.mark.xfail(reason="Sprint 1: dryml.code.analyze fact API not implemented yet", strict=True)
def test_dryml_code_analyze_returns_callable_and_source_facts():
    result = code.analyze(targets.plain_importable_function)
    assert result.facts["callable"].qualname == "plain_importable_function"
    assert result.facts["source"].text


@pytest.mark.xfail(reason="Sprint 1/Sprint 3: annotation facts are not exposed by dryml.code.analyze yet", strict=True)
def test_dryml_code_analyze_returns_direct_annotation_facts():
    result = code.analyze(targets.run_training)
    assert "pandas>=2" in result.facts["annotations"].requirements.environment.requirements


@pytest.mark.xfail(reason="Sprint 1: serializable analysis result is not implemented yet", strict=True)
def test_analysis_result_can_serialize_to_json_compatible_data():
    result = code.analyze(targets.plain_importable_function)
    assert result.to_data()["facts"]
