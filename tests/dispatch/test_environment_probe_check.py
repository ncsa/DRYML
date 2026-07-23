from __future__ import annotations

import sys

import dryml

from dryml.code.analysis import CodeAnalysisResult
from dryml.code.probe import CodeProbeResult
from dryml.code.targets import CodeTargetSpec
from dryml.dispatch import NormalizedDispatchTarget, normalize_user_operation, resolve_dispatch_plan
from dryml.operations import make_function_call_spec


def test_live_complete_discovery_skips_code_probe_and_environment_probe_without_requirement(monkeypatch):
    import dryml.dispatch.requirements as requirements

    code_probes = []
    environment_probes = []

    def unexpected_code_probe(*_args, **_kwargs):
        code_probes.append(True)
        raise AssertionError("complete planning must not code-probe")

    def unexpected_environment_probe(*_args, **_kwargs):
        environment_probes.append(True)
        raise AssertionError("complete planning must not environment-probe")

    monkeypatch.setattr(requirements, "probe_target", unexpected_code_probe)
    monkeypatch.setattr(requirements.environments, "probe", unexpected_environment_probe)

    @dryml.world.default(cpus=1)
    def target():
        return None

    resolution = resolve_dispatch_plan(normalize_user_operation(target, allow_pickle=True), requirement_policy="strict")
    assert resolution.code_probe is None
    assert resolution.environment_record is None
    assert resolution.environment_check.status == "not_required"
    assert code_probes == []
    assert environment_probes == []


def test_environment_requirement_is_checked_against_selected_candidate_record():
    @dryml.env.req(requirements=("package-that-cannot-exist-for-dryml-test>=1",))
    def target():
        return None

    resolution = resolve_dispatch_plan(normalize_user_operation(target, allow_pickle=True), requirement_policy="strict")
    assert resolution.environment_record is not None
    assert resolution.environment_check.status == "incompatible"
    assert resolution.launchable is False


def test_missing_explicit_final_environment_is_structurally_blocking_without_requirement():
    normalized = normalize_user_operation(make_function_call_spec("operator:add"))
    resolution = resolve_dispatch_plan(
        normalized,
        environment={"kind": "python", "executable": "/definitely/missing/python"},
        requirement_policy="ignore",
    )

    assert resolution.environment_check.status == "error"
    assert resolution.launchable is False
    assert resolution.code_probe is not None


def test_final_probe_uses_import_path_even_when_target_kind_is_function(monkeypatch, sample_environment_record):
    import dryml.dispatch.requirements as requirements

    target = CodeTargetSpec("function", import_path="operator:add")
    normalized = NormalizedDispatchTarget(make_function_call_spec("operator:add"), code_target=target, transport="import_path")
    calls = []

    def fake_probe(code_target, **kwargs):
        calls.append((code_target, kwargs))
        result = CodeProbeResult(True, CodeAnalysisResult(target), sample_environment_record)
        return CodeProbeResult.from_data(result.to_data())

    monkeypatch.setattr(requirements, "probe_target", fake_probe)
    monkeypatch.setattr(requirements.environments, "probe", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("final probe record must be reused")))
    resolution = resolve_dispatch_plan(
        normalized,
        environment={"kind": "python", "executable": sys.executable},
        requirement_policy="strict",
    )

    assert len(calls) == 1
    assert "algorithms" not in calls[0][1]
    assert resolution.final_code_probe is not None
    assert resolution.environment_record is not None
