from dryml.core2.store.dir import DirStore

import dryml.operations as ops
import dryml.providers as providers


def test_probe_report_record_round_trip_and_stable_id(tmp_path):
    request = providers.OperationInspectionRequest(operation_spec=ops.make_function_call_spec("providers.fake_provider:target_fn"))
    report = providers.ProbeReport(request=request.to_data(), operation_id=request.operation_id, reports=(providers.OperationInspectionReport(providers.ProviderIdentity("fake", "1"), "ok", request.key, request.operation_id),))
    left = providers.make_probe_report_record(report)
    right = providers.make_probe_report_record(providers.ProbeReport.from_data(report.to_data()))

    from dryml.records import RecordStoreIO

    io = RecordStoreIO(DirStore(tmp_path / "store"))
    ref = providers.write_probe_report(io, report)
    loaded = providers.probe_report_from_record(io.read_record(ref.record_id))

    assert left["kind"] == "probe_report"
    assert providers.make_probe_report_record(report) == left
    assert left == right
    assert loaded.operation_id == request.operation_id
    assert loaded.report_id == ref.record_id


def test_invalid_probe_report_record_rejected():
    from dryml.records import make_record

    import pytest

    with pytest.raises(providers.ProviderReportError):
        providers.validate_probe_report_record(make_record(kind="stored_state", payload={}))


def test_probe_cache_hit_miss_by_provider_operation_and_environment():
    report = providers.ProbeReport(operation_id="op-v1-a", environment_spec_id="envspec-v1-a", runtime_id="runtime-v1-a", probe_policy={})
    key = providers.ProbeCacheKey("operation_inspection", "op-v1-a", "envspec-v1-a", "provider-v1-a", "runtime-v1-a", providers.hash_json_payload({}), providers.hash_json_payload({}))
    cache = providers.ProbeCache()
    cache.put(key, report)

    assert cache.get(key) is report
    assert cache.get(providers.ProbeCacheKey("operation_inspection", "op-v1-b", "envspec-v1-a", "provider-v1-a", "runtime-v1-a", key.probe_policy_hash, key.provider_options_hash)) is None
    assert cache.get(providers.ProbeCacheKey("operation_inspection", "op-v1-a", "envspec-v1-b", "provider-v1-a", "runtime-v1-a", key.probe_policy_hash, key.provider_options_hash)) is None
    assert cache.get(providers.ProbeCacheKey("operation_inspection", "op-v1-a", "envspec-v1-a", "provider-v1-b", "runtime-v1-a", key.probe_policy_hash, key.provider_options_hash)) is None
