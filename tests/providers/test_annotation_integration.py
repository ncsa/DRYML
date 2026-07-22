import dryml.annotations as ann
import dryml.operations as ops
import dryml.providers as providers


def target():
    pass


def fake_report():
    from providers.fake_provider import Provider

    request = providers.OperationInspectionRequest(operation_spec=ops.make_function_call_spec("providers.fake_provider:target_fn"))
    return providers.ProbeReport(request=request.to_data(), operation_id=request.operation_id, reports=(Provider().inspect_operation(request),))


def test_provider_world_requirement_fragment_merges_and_enforces():
    report = fake_report()
    result = ann.resolve(target, provider_fragments=report.annotation_fragments(), overrides={"world": {"roles": {"main": {"process": {"resources": {"cpus": 0}}}}}})

    assert not result.report.ok
    assert any(source.kind == "provider" for issue in result.report.issues for source in issue.sources)


def test_provider_runtime_default_fragment_merges():
    report = fake_report()
    runtime = ann.resolve_runtime_default(target, provider_fragments=report.annotation_fragments())

    assert runtime.frameworks["plain"]["provider"] == "fake"


def test_static_and_provider_source_traces_appear_in_conflict():
    @ann.default(namespace="runtime", fragment={"frameworks": {"plain": {"provider": "static"}}}, merge_policy="error_on_conflict", priority=1)
    def configured():
        pass

    result = ann.resolve(configured, provider_fragments=fake_report().annotation_fragments())
    source_kinds = {source.kind for issue in result.report.issues for source in issue.sources}
    assert not result.report.ok
    assert "provider" in source_kinds
    assert "decorator" in source_kinds


def test_cached_probe_source_traces():
    report = fake_report()
    assert report.annotation_fragments(cached=True)[0].source.kind == "cached_probe"
