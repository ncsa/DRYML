from __future__ import annotations

import dryml.code as code
from dryml.code.source import func_source_extract, get_source_info


def test_source_algorithm_function_class_method_and_lambda(requirement_targets):
    for target in (
        requirement_targets.plain_importable_function,
        requirement_targets.LightningModel,
        requirement_targets.LightningModel.train,
        requirement_targets.local_lambda_with_annotation,
    ):
        result = code.analyze(target, algorithms=("source",))
        assert result.facts_of_kind("source")
        assert result.facts_of_kind("source")[0].data["filename"].endswith("requirements_targets.py")


def test_source_unavailable_and_disabled_diagnostics():
    unavailable = code.analyze(len, algorithms=("source",))
    disabled = code.analyze(len, algorithms=("source",), context=code.CodeAnalysisContext(allow_source=False))

    assert unavailable.diagnostics_of_code("dryml.code.source_unavailable")
    assert disabled.diagnostics_of_code("dryml.code.source_disabled")


def test_old_source_helper_compatibility(requirement_targets):
    info = get_source_info(requirement_targets.plain_importable_function)

    assert info is not None
    assert "def plain_importable_function" in info.source
    assert "def plain_importable_function" in func_source_extract(requirement_targets.plain_importable_function)
