import pytest

import dryml.annotations as ann
import dryml.operations as ops
import dryml.providers as providers


def operation_spec():
    return ops.attach_operation_id(ops.make_function_call_spec("providers.fake_provider:target_fn"))


def test_request_round_trip_and_malformed_operation_rejection():
    request = providers.OperationInspectionRequest(operation_spec=operation_spec(), provider_names=("fake",))

    round_trip = providers.request_from_data(request.to_data())
    assert round_trip.to_data() == request.to_data()
    assert request.operation_id.startswith("op-v1-")
    with pytest.raises(providers.ProviderValidationError):
        providers.OperationInspectionRequest(operation_spec={"bad": "shape"})


def test_probe_policy_rejects_materialization_and_workload_allocation():
    with pytest.raises(providers.ProviderValidationError, match="allow_materialization"):
        providers.ProbePolicy(allow_materialization=True)
    with pytest.raises(providers.ProviderValidationError, match="workload"):
        providers.ProbePolicy(allow_workload_allocation=True)


def test_report_round_trip_and_provider_fragment_sources():
    identity = providers.ProviderIdentity("fake", "1")
    request = providers.OperationInspectionRequest(operation_spec=operation_spec())
    fragment = ann.AnnotationFragment("world", "requirement", {"roles": {"main": {"resources": {"cpus": {"min": 1}}}}}, ann.SourceTrace("provider"))
    report = providers.OperationInspectionReport(identity, "ok", request.key, request.operation_id, fragments=(fragment,))
    probe_report = providers.ProbeReport(request=request.to_data(), reports=(report,), operation_id=request.operation_id)

    round_trip = providers.ProbeReport.from_data(probe_report.to_data())
    fresh = round_trip.annotation_fragments(cached=False)[0]
    cached = round_trip.annotation_fragments(cached=True)[0]
    assert fresh.source.kind == "provider"
    assert cached.source.kind == "cached_probe"
    assert cached.source.metadata["provider_version"] == "1"


def test_structured_failed_report():
    identity = providers.ProviderIdentity("fake")
    report = providers.OperationInspectionReport.failed(identity, None, "boom", exception=RuntimeError("boom"))
    assert report.status == "failed"
    assert report.issues[0].exception_type == "RuntimeError"
