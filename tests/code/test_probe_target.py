from __future__ import annotations

import importlib

import dryml
import dryml.code as code
import pytest


TARGET_MODULE = "probe_targets"


def _codes(result: code.CodeProbeResult) -> set[str]:
    return {item.code for item in result.diagnostics}


def _requirements(result: code.CodeProbeResult):
    assert result.analysis is not None
    return result.analysis.facts_of_kind("requirement")


class LiveBoundTarget:
    def inspectable_method(self):
        return None


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


def test_probe_result_rejects_inconsistent_or_unproven_success_protocol_data():
    with pytest.raises(ValueError, match="ok does not match diagnostics"):
        code.CodeProbeResult.from_data({
            "kind": "dryml.code_probe_result",
            "schema_version": 1,
            "ok": False,
            "analysis": None,
            "environment_record": None,
            "diagnostics": [],
        })
    with pytest.raises(ValueError, match="requires analysis"):
        code.CodeProbeResult.from_data({
            "kind": "dryml.code_probe_result",
            "schema_version": 1,
            "ok": True,
            "analysis": None,
            "environment_record": None,
            "diagnostics": [],
        })


def test_probe_direct_annotation_facts_for_function():
    result = code.probe_target(f"{TARGET_MODULE}:decorated_function", include_environment_record=False)

    assert result.ok
    assert _requirements(result)[0].fragment["requirements"] == ["probepkg>=1"]


def test_probe_current_process_live_local_function_keeps_annotations():
    @dryml.env.req(requirements=("localpkg>=1",))
    def local_target():
        return "not executed"

    result = code.probe_target(local_target, include_environment_record=False)

    assert result.ok
    assert _requirements(result)[0].fragment["requirements"] == ["localpkg>=1"]
    assert result.analysis.facts_of_kind("callable")


def test_probe_current_process_live_bound_method_keeps_callable_metadata():
    result = code.probe_target(LiveBoundTarget().inspectable_method, include_environment_record=False)

    assert result.ok
    assert result.analysis is not None
    assert result.analysis.target.kind == "bound_method"
    assert result.analysis.facts_of_kind("callable")


def test_current_process_timeout_rejects_live_non_serializable_function():
    def local_target():
        return "not executed"

    result = code.probe_target(local_target, include_environment_record=False, timeout=0.1)

    assert not result.ok
    assert result.diagnostics[0].code == "code_probe.timeout"


def test_current_process_timeout_rejects_source_spec_without_reconstruction():
    spec = code.CodeTargetSpec("source_spec", source_spec={"kind": "function", "source": "lambda x: x"})

    result = code.probe_target(spec, include_environment_record=False, timeout=0.1)

    assert not result.ok
    assert result.diagnostics[0].code == "code_probe.source_spec_reconstruction_unavailable"


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


def test_probe_can_explicitly_request_opt_in_static_calls():
    result = code.probe_target(
        f"{TARGET_MODULE}:plain_function",
        algorithms=("static_calls",),
        include_environment_record=False,
    )

    assert result.ok
    assert result.analysis is not None
    assert result.analysis.facts_of_kind("static_call_summary")


def test_inline_and_worker_static_calls_have_same_fact_data():
    target = "dryml.code.algorithms.source:analyze_target"
    inline = code.probe_target(
        target,
        algorithms=("static_calls",),
        include_environment_record=False,
    )
    worker = code.probe_target(
        target,
        algorithms=("static_calls",),
        include_environment_record=False,
        timeout=10,
    )

    assert inline.ok and worker.ok
    assert inline.analysis is not None and worker.analysis is not None
    assert [fact.to_data() for fact in inline.analysis.facts] == [fact.to_data() for fact in worker.analysis.facts]


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
