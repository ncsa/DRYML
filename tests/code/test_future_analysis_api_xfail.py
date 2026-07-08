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


def test_dryml_code_analyze_exists():
    assert callable(code.analyze)


def test_dryml_code_analyze_returns_callable_and_source_facts():
    result = code.analyze(targets.plain_importable_function)
    assert result.facts_of_kind("callable")[0].data["qualname"] == "plain_importable_function"
    assert result.facts_of_kind("source")[0].data["source"]


def test_dryml_code_analyze_returns_direct_annotation_facts():
    result = code.analyze(targets.run_training)
    requirements = result.facts_of_kind("requirement")
    assert any(
        fact.namespace == "environment" and "pandas>=2" in fact.fragment["requirements"]
        for fact in requirements
    )


def test_analysis_result_can_serialize_to_json_compatible_data():
    result = code.analyze(targets.plain_importable_function)
    assert result.to_data()["facts"]
