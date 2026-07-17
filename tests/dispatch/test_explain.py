from __future__ import annotations

import json
import math

import dryml

from dryml.core2.store.dir import DirStore
from dryml.dispatch import Dispatcher
from dryml.dispatch.errors import DispatchPlanningError
from dryml.worlds import LocalResourceInventory


@dryml.world.req(accelerators={"gpu": {"min": 1}})
def gpu_target():
    return None


def test_explain_returns_nonlaunching_structured_failure_with_plan_parity(tmp_path):
    store = DirStore(tmp_path / "store", query_index="none")
    world = {"roles": {"main": {"replicas": 1, "process": {}}}}
    dispatcher = Dispatcher(store=store)

    explanation = dispatcher.explain(gpu_target, allow_pickle=True, world=world)
    assert explanation.launchable is False
    assert explanation.resolution.world_check.status == "incompatible"
    assert json.loads(json.dumps(explanation.to_data()))["launchable"] is False
    with __import__("pytest").raises(DispatchPlanningError, match="not launchable"):
        dispatcher.plan(gpu_target, allow_pickle=True, world=world)


def test_explain_does_not_write_operation_records_for_importable_target(tmp_path):
    store = DirStore(tmp_path / "store", query_index="none")

    explanation = Dispatcher(store=store).explain(
        gpu_target,
        allow_pickle=True,
        inventory=LocalResourceInventory((0,)),
        requirement_policy="ignore",
    )
    assert explanation.launchable is True
    assert any(item.code == "dryml.dispatch.world_synthesis_failed" for item in explanation.resolution.diagnostics)
    assert not store.records.specs_dir.exists()


def test_explain_does_not_require_store_for_ordinary_callable_or_leak_pickle(monkeypatch):
    import dryml.dispatch.planner as planner

    captured = {}
    original = planner.normalize_user_operation

    def capture(*args, **kwargs):
        normalized = original(*args, **kwargs)
        captured["paths"] = tuple(normalized.launch.get("cleanup_paths") or ())
        return normalized

    monkeypatch.setattr(planner, "normalize_user_operation", capture)
    explanation = Dispatcher().explain(lambda: None, allow_pickle=True, requirement_policy="ignore")

    assert explanation.launchable is True
    assert captured["paths"]
    assert all(not __import__("os").path.exists(path) for path in captured["paths"])


def test_failed_bootstrap_target_probe_remains_blocking_under_relaxed_policies():
    from dryml.operations import make_function_call_spec

    for policy in ("warn", "ignore"):
        explanation = Dispatcher().explain(
            make_function_call_spec("missing_audit_module:fn"),
            requirement_policy=policy,
        )

        assert not explanation.launchable
        assert explanation.resolution.bootstrap_code_probe is not None
        assert not explanation.resolution.bootstrap_code_probe.ok


def test_successful_final_probe_supersedes_failed_bootstrap_probe(monkeypatch, sample_environment_record):
    import sys

    import dryml.dispatch.requirements as requirements
    from dryml.code.analysis import CodeAnalysisResult
    from dryml.code.facts import DiagnosticFact
    from dryml.code.probe import CodeProbeResult
    from dryml.dispatch import normalize_user_operation, resolve_dispatch_plan
    from dryml.environments import PythonExecutableSpec
    from dryml.operations import make_function_call_spec

    normalized = normalize_user_operation(make_function_call_spec("operator:add", args=[1, 2]))
    incomplete = CodeAnalysisResult(
        normalized.code_target,
        diagnostics=(DiagnosticFact(severity="error", code="dryml.code.algorithm_not_applicable", message="force bootstrap probe"),),
    )
    probes = []

    def fake_probe(target, **_kwargs):
        probes.append(target)
        if len(probes) == 1:
            return CodeProbeResult(False, None, None, (DiagnosticFact(severity="error", code="code_probe.import_failed", message="bootstrap failed"),))
        result = CodeProbeResult(True, CodeAnalysisResult(target), sample_environment_record)
        return CodeProbeResult.from_data(result.to_data())

    monkeypatch.setattr(requirements, "analyze", lambda *_args, **_kwargs: incomplete)
    monkeypatch.setattr(requirements, "probe_target", fake_probe)

    resolution = resolve_dispatch_plan(
        normalized,
        environment_candidates=(PythonExecutableSpec(sys.executable),),
        requirement_policy="strict",
    )

    assert resolution.launchable
    assert resolution.final_code_probe is not None and resolution.final_code_probe.ok
    assert len(probes) == 2


def test_final_probe_environment_default_does_not_retain_resolver_candidate(monkeypatch, sample_environment_record):
    import sys

    import dryml.dispatch.requirements as requirements
    from dryml import annotations
    from dryml.code.analysis import CodeAnalysisResult
    from dryml.code.facts import AnnotationFact, DiagnosticFact
    from dryml.code.probe import CodeProbeResult
    from dryml.dispatch import normalize_user_operation, resolve_dispatch_plan
    from dryml.environments import EnvironmentProbeResult, PythonExecutableSpec
    from dryml.operations import make_function_call_spec

    @dryml.env.default(PythonExecutableSpec("/final/default/python").to_data())
    def final_default_target():
        return None

    final_analysis = CodeAnalysisResult(
        normalize_user_operation(make_function_call_spec("operator:add", args=[1, 2])).code_target,
        facts=(AnnotationFact(data=annotations.fragments_for(final_default_target)[0].to_data()),),
    )
    incomplete = CodeAnalysisResult(
        final_analysis.target,
        diagnostics=(DiagnosticFact(severity="error", code="dryml.code.algorithm_not_applicable", message="force bootstrap probe"),),
    )
    probes = []

    def fake_probe(target, **_kwargs):
        probes.append(target)
        if len(probes) == 1:
            return CodeProbeResult(False, None, None, (DiagnosticFact(severity="error", code="code_probe.import_failed", message="bootstrap failed"),))
        result = CodeProbeResult(True, final_analysis, sample_environment_record)
        return CodeProbeResult.from_data(result.to_data())

    monkeypatch.setattr(requirements, "analyze", lambda *_args, **_kwargs: incomplete)
    monkeypatch.setattr(requirements, "probe_target", fake_probe)
    environment_probes = []
    monkeypatch.setattr(
        requirements.environments,
        "probe",
        lambda spec, **_kwargs: environment_probes.append(spec) or EnvironmentProbeResult(spec=spec, ok=True, record=sample_environment_record),
    )

    resolution = resolve_dispatch_plan(
        normalize_user_operation(make_function_call_spec("operator:add", args=[1, 2])),
        environment_candidates=(PythonExecutableSpec(sys.executable),),
        requirement_policy="strict",
    )

    assert resolution.environment_selection.source == "resolver"
    assert not resolution.launchable
    assert any(item.code == "dryml.dispatch.final_probe_annotation_mismatch" for item in resolution.diagnostics)
    assert len(probes) == 2
    assert environment_probes == []


def test_implicit_fallback_inventory_failure_has_explain_plan_parity(monkeypatch, tmp_path):
    import dryml.dispatch.requirements as requirements
    from dryml.operations import make_function_call_spec

    calls = []

    def fail_inventory(*_args, **_kwargs):
        calls.append(True)
        raise RuntimeError("local inventory unavailable")

    monkeypatch.setattr(requirements.worlds, "local_inventory", fail_inventory)
    operation = make_function_call_spec("operator:add", args=[1, 2])
    dispatcher = Dispatcher(store=DirStore(tmp_path / "store", query_index="none"))

    explanation = dispatcher.explain(operation, requirement_policy="ignore")

    assert not explanation.launchable
    assert any(item.code == "dryml.dispatch.local_allocation_failed" for item in explanation.resolution.diagnostics)
    with __import__("pytest").raises(DispatchPlanningError, match="local inventory"):
        dispatcher.plan(operation, requirement_policy="ignore")
    assert len(calls) == 2


def test_implicit_fallback_plan_discovers_inventory_once(monkeypatch, tmp_path):
    import dryml.dispatch.requirements as requirements
    from dryml.operations import make_function_call_spec

    calls = []
    inventory = LocalResourceInventory((0,))
    monkeypatch.setattr(requirements.worlds, "local_inventory", lambda **_kwargs: calls.append(True) or inventory)

    Dispatcher(store=DirStore(tmp_path / "store", query_index="none")).plan(
        make_function_call_spec("operator:add", args=[1, 2]),
        requirement_policy="ignore",
    )

    assert len(calls) == 1


def test_resolver_reuses_matching_bootstrap_environment_record(monkeypatch, sample_environment_record):
    import dryml.dispatch.requirements as requirements
    from dryml.code.analysis import CodeAnalysisResult
    from dryml.code.probe import CodeProbeResult
    from dryml.code.targets import CodeTargetSpec
    from dryml.environments import CurrentEnvironmentSpec, EnvironmentRequirement

    bootstrap = CodeProbeResult(True, CodeAnalysisResult(CodeTargetSpec.from_import_path("operator:add")), sample_environment_record)
    monkeypatch.setattr(requirements.environments, "probe", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected reprobe")))

    _selection, _data, resolution = requirements._select_environment(
        None,
        None,
        requirement=EnvironmentRequirement(),
        bootstrap_probe=bootstrap,
        bootstrap_environment=CurrentEnvironmentSpec().to_data(),
    )

    assert resolution is not None and resolution.ok
    assert resolution.selected_source == "current"


def test_final_probe_topology_requirement_is_rechecked_after_failed_bootstrap(monkeypatch, sample_environment_record):
    import sys

    import dryml.dispatch.requirements as requirements
    from dryml.code.analysis import CodeAnalysisResult
    from dryml.code.facts import AnnotationFact, DiagnosticFact
    from dryml.code.probe import CodeProbeResult
    from dryml.dispatch import normalize_user_operation, resolve_dispatch_plan
    from dryml.environments import PythonExecutableSpec
    from dryml.operations import make_function_call_spec

    normalized = normalize_user_operation(make_function_call_spec("operator:add", args=[1, 2]))
    incomplete = CodeAnalysisResult(
        normalized.code_target,
        diagnostics=(DiagnosticFact(severity="error", code="dryml.code.algorithm_not_applicable", message="force bootstrap probe"),),
    )
    final_analysis = CodeAnalysisResult(
        normalized.code_target,
        facts=(AnnotationFact(data={
            "namespace": "world",
            "kind": "requirement",
            "fragment": {"roles": {"main": {"topology": {"collectives": True}}}},
            "source": {"kind": "synthetic"},
        }),),
    )
    probes = []

    def fake_probe(target, **_kwargs):
        probes.append(target)
        if len(probes) == 1:
            return CodeProbeResult(False, None, None, (DiagnosticFact(severity="error", code="code_probe.import_failed", message="bootstrap failed"),))
        result = CodeProbeResult(True, final_analysis, sample_environment_record)
        return CodeProbeResult.from_data(result.to_data())

    monkeypatch.setattr(requirements, "analyze", lambda *_args, **_kwargs: incomplete)
    monkeypatch.setattr(requirements, "probe_target", fake_probe)

    resolution = resolve_dispatch_plan(
        normalized,
        environment_candidates=(PythonExecutableSpec(sys.executable),),
        requirement_policy="ignore",
    )

    assert not resolution.launchable
    assert any(item.code == "dryml.dispatch.local_world_topology_unsupported" for item in resolution.diagnostics)
    assert len(probes) == 2


def test_plan_cleans_pickle_artifacts_when_marshalling_fails(monkeypatch):
    import dryml.dispatch.planner as planner

    captured = {}
    original = planner.normalize_user_operation

    def capture(*args, **kwargs):
        normalized = original(*args, **kwargs)
        captured["paths"] = tuple(normalized.launch.get("cleanup_paths") or ())
        return normalized

    monkeypatch.setattr(planner, "normalize_user_operation", capture)
    with __import__("pytest").raises(Exception):
        Dispatcher(store=object()).plan(lambda: None, allow_pickle=True, requirement_policy="ignore")

    assert captured["paths"]
    assert all(not __import__("os").path.exists(path) for path in captured["paths"])


def test_plan_cleans_pickle_artifacts_when_interrupted(monkeypatch, tmp_path):
    import dryml.dispatch.planner as planner

    captured = {}
    original = planner.normalize_user_operation

    def capture(*args, **kwargs):
        normalized = original(*args, **kwargs)
        captured["paths"] = tuple(normalized.launch.get("cleanup_paths") or ())
        return normalized

    monkeypatch.setattr(planner, "normalize_user_operation", capture)
    monkeypatch.setattr(planner, "resolve_dispatch_plan", lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt()))
    with __import__("pytest").raises(KeyboardInterrupt):
        Dispatcher(store=DirStore(tmp_path / "store", query_index="none")).plan(lambda: None, allow_pickle=True)

    assert captured["paths"]
    assert all(not __import__("os").path.exists(path) for path in captured["paths"])


def test_explain_rejects_invalid_inventory_policy_before_normalization(monkeypatch):
    import dryml.dispatch.planner as planner

    monkeypatch.setattr(planner, "normalize_user_operation", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not normalize")))

    with __import__("pytest").raises(DispatchPlanningError, match="invalid inventory_policy"):
        Dispatcher().explain(lambda: None, allow_pickle=True, inventory_policy="invalid")


def test_explain_rejects_invalid_injected_inventory_before_planning():
    from dryml.operations import make_function_call_spec

    with __import__("pytest").raises(DispatchPlanningError, match="LocalResourceInventory"):
        Dispatcher().explain(make_function_call_spec("operator:add", args=[1, 2]), inventory=object())


def test_explain_rejects_nonfinite_probe_timeout_before_discovery():
    with __import__("pytest").raises(DispatchPlanningError, match="probe_timeout_s must be a positive number"):
        Dispatcher().explain(
            lambda: None,
            allow_pickle=True,
            analysis_policy={"probe_timeout_s": math.nan},
        )


def test_plan_world_cleans_pickle_artifacts_when_marshalling_fails(monkeypatch):
    import dryml.dispatch.planner as planner

    captured = {}
    original = planner.normalize_user_operation

    def capture(*args, **kwargs):
        normalized = original(*args, **kwargs)
        captured["paths"] = tuple(normalized.launch.get("cleanup_paths") or ())
        return normalized

    monkeypatch.setattr(planner, "normalize_user_operation", capture)
    with __import__("pytest").raises(Exception):
        Dispatcher(store=object()).plan_world(lambda: None, world={"roles": {"main": {"replicas": 1, "process": {}}}}, allow_pickle=True, requirement_policy="ignore")

    assert captured["paths"]
    assert all(not __import__("os").path.exists(path) for path in captured["paths"])


def test_plan_world_cleans_pickle_artifacts_when_recipe_build_fails(monkeypatch, tmp_path):
    import dryml.dispatch.planner as planner

    captured = {}
    original = planner.normalize_user_operation

    def capture(*args, **kwargs):
        normalized = original(*args, **kwargs)
        captured["paths"] = tuple(normalized.launch.get("cleanup_paths") or ())
        return normalized

    monkeypatch.setattr(planner, "normalize_user_operation", capture)
    monkeypatch.setattr(planner, "make_execution_recipe", lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("recipe failure")))
    world = {"roles": {"main": {"replicas": 1, "process": {}}}}
    with __import__("pytest").raises(RuntimeError, match="recipe failure"):
        Dispatcher(store=DirStore(tmp_path / "store", query_index="none")).plan_world(lambda: None, world=world, allow_pickle=True, requirement_policy="ignore")

    assert captured["paths"]
    assert all(not __import__("os").path.exists(path) for path in captured["paths"])
    assert not (tmp_path / "store" / "records" / "specs" / "world").exists()
    assert not (tmp_path / "store" / "records" / "specs" / "world_allocation").exists()
