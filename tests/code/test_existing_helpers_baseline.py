from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from dryml.code.ast_tools import collect_accesses_from_source
from dryml.code.callable_info import analyze_callable
from dryml.code.source import get_source_info
from dryml.core.symbol import ImportRef, SourceSpec


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


def test_callable_info_for_module_level_function():
    info = analyze_callable(targets.plain_importable_function)

    assert info.func is targets.plain_importable_function
    assert info.module == "dryml_requirement_targets"
    assert info.qualname == "plain_importable_function"
    assert info.is_function is True


def test_callable_info_for_bound_method():
    instance = targets.LightningModel()
    info = analyze_callable(instance.train)

    assert info.bound_self is instance
    assert info.is_bound_method is True
    assert info.qualname.endswith("LightningModel.train")


def test_callable_info_for_lambda_or_local_function_current_behavior():
    local = targets.make_local_training_function()
    lambda_info = analyze_callable(targets.local_lambda_with_annotation)
    local_info = analyze_callable(local)

    assert lambda_info.is_function is True
    assert lambda_info.qualname == "<lambda>"
    assert "<locals>" in local_info.qualname


def test_source_info_for_module_level_function():
    info = get_source_info(targets.plain_importable_function)

    assert info is not None
    assert "def plain_importable_function" in info.source
    assert info.filename.endswith("requirements_targets.py")
    assert isinstance(info.start_line, int)


def test_source_info_handles_unavailable_source_gracefully():
    assert get_source_info(len) is None


def test_ast_access_collector_finds_attribute_access():
    collector = collect_accesses_from_source("def f(obj):\n    return obj.value\n")

    assert any(access.root == "obj" and access.chain == ("value",) for access in collector.attr_accesses)


def test_ast_access_collector_finds_method_call_like_nodes():
    collector = collect_accesses_from_source("def f(obj):\n    return obj.child.train(1)\n")

    assert any(call.root == "obj" and call.chain == ("child", "train") for call in collector.method_calls)


def test_ast_helper_behavior_on_nested_method_call_current_behavior():
    collector = collect_accesses_from_source("def f(obj):\n    return obj.child().train()\n")
    assert all(call.chain != ("child", "train") for call in collector.method_calls)
    assert any(call.chain == ("child",) for call in collector.method_calls)


def test_core_symbol_import_ref_round_trip_if_public():
    ref = ImportRef.from_import_path("operator:add")
    assert ref.import_path() == "operator:add"
    assert ref.resolve()(2, 3) == 5


def test_core_symbol_closure_rejection_current_behavior():
    value = 1

    def closure():
        return value

    with pytest.raises(ValueError, match="closure"):
        SourceSpec.from_function(closure)
