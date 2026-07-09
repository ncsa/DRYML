from __future__ import annotations

import importlib

import dryml.code as code


TARGET_MODULE = "probe_targets"


def _codes(result: code.CodeProbeResult) -> set[str]:
    return {item.code for item in result.diagnostics}


def _requirements(result: code.CodeProbeResult):
    assert result.analysis is not None
    return result.analysis.facts_of_kind("requirement")


def test_probe_current_environment_importable_function():
    result = code.probe_target(f"{TARGET_MODULE}:plain_function", include_environment_record=False)

    assert result.ok
    assert result.analysis is not None
    assert result.analysis.facts_of_kind("callable")


def test_probe_import_path_target_round_trips_request_result():
    request = code.CodeProbeRequest(
        target=code.CodeTargetSpec.from_import_path(f"{TARGET_MODULE}:plain_function"),
        include_environment_record=False,
    )
    restored = code.CodeProbeRequest.from_data(request.to_data())
    result = code.run_probe_request(restored)

    assert result.ok
    assert code.CodeProbeResult.from_data(result.to_data()).ok


def test_probe_direct_annotation_facts_for_function():
    result = code.probe_target(f"{TARGET_MODULE}:decorated_function", include_environment_record=False)

    assert result.ok
    assert _requirements(result)[0].fragment["requirements"] == ["probepkg>=1"]


def test_probe_method_classmethod_and_staticmethod_annotations():
    method = code.probe_target(f"{TARGET_MODULE}:ProbeMethods.train", include_environment_record=False)
    classmethod = code.probe_target(f"{TARGET_MODULE}:ProbeMethods.build", include_environment_record=False)
    staticmethod = code.probe_target(f"{TARGET_MODULE}:ProbeMethods.make", include_environment_record=False)

    assert _requirements(method)[0].fragment["requirements"] == ["methodpkg>=1"]
    assert _requirements(classmethod)[0].fragment["requirements"] == ["classpkg>=1"]
    assert _requirements(staticmethod)[0].fragment["requirements"] == ["staticpkg>=1"]


def test_probe_source_and_symbol_facts_are_returned():
    result = code.probe_target(f"{TARGET_MODULE}:plain_function", include_environment_record=False)
    assert result.analysis is not None
    assert result.analysis.facts_of_kind("source")
    assert result.analysis.facts_of_kind("symbol")


def test_probe_algorithm_list_override():
    result = code.probe_target(
        f"{TARGET_MODULE}:plain_function",
        algorithms=("symbol_capture",),
        include_environment_record=False,
    )
    assert result.analysis is not None
    assert result.analysis.facts_of_kind("symbol")
    assert not result.analysis.facts_of_kind("callable")


def test_probe_unsupported_target_diagnostic():
    result = code.probe_target(object(), include_environment_record=False)
    assert not result.ok
    assert "code_probe.target_normalization_error" in _codes(result)


def test_probe_import_failure_diagnostic_includes_probe_code():
    result = code.probe_target("probe_import_failure:target", include_environment_record=False)
    assert not result.ok
    assert "code_probe.import_error" in _codes(result)


def test_probe_does_not_execute_target_function_body_or_instantiate_class():
    module = importlib.import_module(TARGET_MODULE)
    module.BODY_EXECUTED = False

    function_result = code.probe_target(f"{TARGET_MODULE}:body_must_not_run", include_environment_record=False)
    class_result = code.probe_target(f"{TARGET_MODULE}:ProbeMethods", include_environment_record=False)

    assert function_result.ok
    assert class_result.ok
    assert module.BODY_EXECUTED is False
