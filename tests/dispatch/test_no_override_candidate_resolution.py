from __future__ import annotations

from dataclasses import replace

import dryml
import pytest

from dryml.core2.store.dir import DirStore
from dryml.dispatch import Dispatcher, normalize_user_operation, resolve_dispatch_plan
from dryml.environments import ContainerEnvironmentSpec, CurrentEnvironmentSpec, EnvironmentRegistry, EnvironmentRequirement, PythonExecutableSpec, inspect_current, resolve, use
from dryml.operations import make_function_call_spec
from dryml.worlds import LocalResourceInventory, WorldSpec, use as use_world


@dryml.world.req(cpus={"min": 2})
def cpu_target():
    return None


@dryml.world.default(cpus=2)
def default_world_target():
    return None


@dryml.world.req(roles={"trainer": {"replicas": {"exact": 2}, "resources": {"cpus": {"exact": 1}}}})
def multi_worker_target():
    return None


@dryml.env.req(tags=("resolved",))
def resolver_target():
    return None


@dryml.world.default(cpus=1)
@dryml.world.req(cpus={"min": 2})
def incompatible_default_world_target():
    return None


@dryml.env.default(CurrentEnvironmentSpec())
@dryml.env.req(tags=("unavailable",))
def incompatible_default_environment_target():
    return None


@dryml.env.req(tags=("unavailable",))
def incompatible_current_environment_target():
    return None


@dryml.world.req(cpus={"min": 2})
def incompatible_current_world_target():
    return None


def test_no_override_hard_world_requirement_is_synthesized_once():
    inventory = LocalResourceInventory((0, 1, 2, 3))

    resolution = resolve_dispatch_plan(
        normalize_user_operation(cpu_target, allow_pickle=True),
        inventory=inventory,
        requirement_policy="strict",
        single_worker_only=True,
    )

    assert resolution.world_selection.source == "synthesized"
    assert resolution.world_synthesis is not None and resolution.world_synthesis.ok
    assert resolution.inventory_summary == inventory.summary()
    assert resolution.launchable


def test_dispatch_reports_inventory_discovery_failure_as_structured_synthesis_failure(monkeypatch):
    import dryml.worlds.synthesis as synthesis

    def fail_inventory(*_args, **_kwargs):
        raise RuntimeError("malformed local inventory")

    monkeypatch.setattr(synthesis, "local_inventory", fail_inventory)
    explanation = Dispatcher().explain(cpu_target, allow_pickle=True)

    assert explanation.launchable is False
    assert explanation.resolution.world_synthesis is not None
    assert explanation.resolution.world_synthesis.status == "error"
    assert explanation.resolution.world_synthesis.diagnostics[0].code == "inventory_discovery_failed"


def test_failed_inventory_discovery_is_reused_during_reconciliation(monkeypatch):
    import dryml.worlds.synthesis as synthesis
    from dryml.dispatch.requirements import _select_world
    from dryml.worlds import WorldRequirement

    calls = []
    monkeypatch.setattr(
        synthesis,
        "local_inventory",
        lambda **_kwargs: calls.append(True) or (_ for _ in ()).throw(RuntimeError("inventory unavailable")),
    )
    requirement = WorldRequirement.from_data({"roles": {"main": {"resources": {"cpus": {"min": 1}}}}})

    _selection, _world, first = _select_world(None, None, requirement=requirement)
    _selection, _world, second = _select_world(
        None,
        None,
        requirement=requirement,
        inventory_discovery_error=first.inventory_discovery_error,
    )

    assert len(calls) == 1
    assert second.inventory_source == "discovery_failed"
    assert second.inventory_policy == "lightweight"


def test_unsupported_requirement_free_resolver_candidate_falls_back_to_current():
    explanation = Dispatcher().explain(
        make_function_call_spec("operator:add", args=[1, 2]),
        environment_candidates=(ContainerEnvironmentSpec("example/image"),),
        requirement_policy="ignore",
    )

    assert explanation.resolution.environment_selection.source == "resolver"
    assert explanation.resolution.environment_resolution is not None
    assert explanation.resolution.environment_resolution.selected_source == "current"
    assert explanation.resolution.environment_resolution.attempts[0].status == "unsupported"
    assert explanation.launchable is True


@pytest.mark.parametrize("policy", ("strict", "warn", "ignore"))
def test_dispatch_blocks_incomplete_environment_resolution_without_fallback_probe(monkeypatch, policy):
    import dryml.dispatch.requirements as requirements

    rejected = PythonExecutableSpec("/rejected/python")
    calls = []

    def probe(spec, **_kwargs):
        calls.append(spec)
        if spec == rejected:
            return requirements.environments.EnvironmentProbeResult(spec, False)
        raise AssertionError("incomplete resolver input must not probe the fallback")

    monkeypatch.setattr(
        requirements.environments,
        "probe",
        probe,
    )

    dispatch_resolution = resolve_dispatch_plan(
        normalize_user_operation(resolver_target, allow_pickle=True),
        environment_candidates=(rejected, *((rejected.to_data(),) * 300)),
        requirement_policy=policy,
    )

    assert dispatch_resolution.environment_resolution is not None
    assert dispatch_resolution.environment_resolution.status == "incomplete"
    assert not dispatch_resolution.launchable
    assert calls == [rejected]


@pytest.mark.parametrize("policy", ("strict", "warn", "ignore"))
def test_dispatch_blocks_probe_deadline_exhaustion_without_relaxed_fallback(monkeypatch, policy):
    import dryml.dispatch.requirements as requirements
    from dryml.environments.resolution import resolve as resolve_environments

    candidate = PythonExecutableSpec("/slow-candidate/python")
    clock_state = {"now": 0.0}

    def delayed_probe(spec, *, timeout):
        clock_state["now"] = 1.0
        return requirements.environments.EnvironmentProbeResult(
            spec,
            True,
            record=replace(inspect_current(), tags=("resolved",)),
        )

    def resolve_with_short_deadline(requirement, **kwargs):
        kwargs.pop("probe_runner")
        return resolve_environments(
            requirement,
            **kwargs,
            total_timeout=0.5,
            clock=lambda: clock_state["now"],
            probe_runner=delayed_probe,
        )

    monkeypatch.setattr(requirements.environments, "resolve", resolve_with_short_deadline)
    dispatch_resolution = resolve_dispatch_plan(
        normalize_user_operation(resolver_target, allow_pickle=True),
        environment_candidates=(candidate,),
        requirement_policy=policy,
    )

    assert dispatch_resolution.environment_resolution is not None
    assert dispatch_resolution.environment_resolution.status == "incomplete"
    assert not dispatch_resolution.launchable


@pytest.mark.parametrize(("policy", "launchable"), (("strict", False), ("warn", True), ("ignore", True)))
def test_dispatch_completed_resolver_no_match_obeys_requirement_policy(monkeypatch, policy, launchable):
    import dryml.dispatch.requirements as requirements

    candidate = PythonExecutableSpec("/incompatible/python")
    monkeypatch.setattr(
        requirements.environments,
        "probe",
        lambda spec, **_kwargs: requirements.environments.EnvironmentProbeResult(
            spec, True, record=replace(inspect_current(), tags=())
        ),
    )

    resolution = resolve_dispatch_plan(
        normalize_user_operation(resolver_target, allow_pickle=True),
        environment_candidates=(candidate,),
        requirement_policy=policy,
    )

    assert resolution.environment_resolution is not None
    assert resolution.environment_resolution.status == "no_match"
    assert resolution.launchable is launchable


def test_attached_record_does_not_bypass_unsupported_environment_launch():
    explanation = Dispatcher().explain(
        make_function_call_spec("operator:add", args=[1, 2]),
        environment={"spec": ContainerEnvironmentSpec("example/image").to_data(), "record": inspect_current().to_data()},
        requirement_policy="ignore",
    )

    assert explanation.launchable is False
    assert any(item.code == "dryml.dispatch.environment_launch_unsupported" for item in explanation.resolution.diagnostics)


def test_explanation_formats_synthesized_inventory_summary():
    explanation = Dispatcher().explain(
        cpu_target,
        allow_pickle=True,
        inventory=LocalResourceInventory((2, 3), {"gpu": ("gpu-a",)}),
        requirement_policy="strict",
    )

    assert "inventory_cpus=2" in str(explanation)
    assert "inventory_accelerators=['gpu']" in str(explanation)


def test_explanation_formats_total_resolver_counts_not_bounded_trace_length():
    explanation = Dispatcher().explain(make_function_call_spec("operator:add", args=[1, 2]))
    resolution = explanation.resolution
    environment_resolution = resolve(None, candidates=(CurrentEnvironmentSpec(),), include_current=False)

    reported = replace(
        explanation,
        resolution=replace(
            resolution,
            environment_resolution=replace(
                environment_resolution,
                attempts=environment_resolution.attempts[:1],
                attempt_count=37,
                probe_count=11,
            ),
        ),
    )

    assert "environment_attempts=37" in str(reported)
    assert "environment_probes=11" in str(reported)


def test_dispatch_reuses_truncated_current_resolver_evidence(monkeypatch):
    import dryml.dispatch.requirements as requirements

    candidates = tuple(PythonExecutableSpec(f"/candidate-{index}") for index in range(31))
    record = replace(inspect_current(), tags=())
    resolver_result = resolve(
        EnvironmentRequirement(tags=("resolved",)),
        candidates=candidates,
        max_candidates=32,
        probe_runner=lambda spec, *, timeout: requirements.environments.EnvironmentProbeResult(
            spec,
            True,
            record=record,
        ),
    )
    assert resolver_result.fallback_record is not None
    assert len(resolver_result.attempts) == 32
    monkeypatch.setattr(requirements.environments, "resolve", lambda *_args, **_kwargs: resolver_result)
    monkeypatch.setattr(requirements.environments, "probe", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("fallback must reuse resolver evidence")))

    resolution = resolve_dispatch_plan(
        normalize_user_operation(resolver_target, allow_pickle=True),
        requirement_policy="warn",
    )

    assert not any(item.code == "dryml.dispatch.environment_probe_failed" for item in resolution.diagnostics)


def test_plan_allocates_a_synthesized_one_worker_world(tmp_path):
    plan = Dispatcher(store=DirStore(tmp_path / "store", query_index="none")).plan(
        cpu_target,
        allow_pickle=True,
        inventory=LocalResourceInventory((4, 5)),
        requirement_policy="strict",
    )

    assert plan.resolution.world_selection.source == "synthesized"
    assert plan.envelope.allocation_view["cpus"] == [4, 5]


def test_plan_allocates_annotation_default_and_context_current_worlds(tmp_path):
    dispatcher = Dispatcher(store=DirStore(tmp_path / "store", query_index="none"))
    inventory = LocalResourceInventory((4, 5))

    default_plan = dispatcher.plan(
        default_world_target,
        allow_pickle=True,
        inventory=inventory,
        requirement_policy="strict",
    )
    current_world = WorldSpec.from_data({"roles": {"main": {"replicas": 1, "process": {"resources": {"cpus": 2}}}}})
    with use_world(current_world):
        current_plan = dispatcher.plan(
            lambda: None,
            allow_pickle=True,
            inventory=inventory,
            requirement_policy="strict",
        )

    assert default_plan.resolution.world_selection.source == "annotation_default"
    assert current_plan.resolution.world_selection.source == "current"
    assert default_plan.envelope.allocation_view["cpus"] == [4, 5]
    assert current_plan.envelope.allocation_view["cpus"] == [4, 5]
    assert default_plan.resolution.world_allocation_summary["backend"] == "local_subprocess"
    assert current_plan.resolution.world_allocation_summary["backend"] == "local_subprocess"


def test_successful_resolver_synthesis_plan_and_explain_have_matching_resolution(tmp_path):
    dispatcher = Dispatcher(store=DirStore(tmp_path / "store", query_index="none"))
    inventory = LocalResourceInventory((4, 5))
    kwargs = {
        "allow_pickle": True,
        "environment_candidates": (CurrentEnvironmentSpec(),),
        "inventory": inventory,
        "requirement_policy": "strict",
    }

    plan = dispatcher.plan(cpu_target, **kwargs)
    explanation = dispatcher.explain(cpu_target, **kwargs)

    assert plan.resolution.environment_selection.source == explanation.resolution.environment_selection.source == "resolver"
    assert plan.resolution.world_selection.source == explanation.resolution.world_selection.source == "synthesized"
    assert plan.resolution.environment_resolution.to_data() == explanation.resolution.environment_resolution.to_data()
    assert plan.resolution.world_synthesis.to_data() == explanation.resolution.world_synthesis.to_data()
    assert plan.resolution.metadata()["dryml.environment_resolution"] == explanation.resolution.metadata()["dryml.environment_resolution"]
    assert plan.resolution.metadata()["dryml.world_synthesis"] == explanation.resolution.metadata()["dryml.world_synthesis"]


def test_plan_world_synthesizes_an_omitted_multi_worker_world(tmp_path):
    inventory = LocalResourceInventory((0, 1))
    plan = Dispatcher(store=DirStore(tmp_path / "store", query_index="none")).plan_world(
        multi_worker_target,
        allow_pickle=True,
        inventory=inventory,
        requirement_policy="strict",
    )

    assert len(plan.worker_plans) == 2
    assert len(plan.world_spec["payload"]["roles"]["trainer"]) == 2
    assert plan.dispatch_spec["payload"]["metadata"]["dryml.local_inventory"] == inventory.summary()


def test_warn_synthesis_failure_keeps_a_human_blocking_action():
    explanation = Dispatcher().explain(
        cpu_target,
        allow_pickle=True,
        inventory=LocalResourceInventory((0,)),
        requirement_policy="warn",
    )

    assert not explanation.launchable
    assert explanation.blocking_diagnostics
    assert "blocking_action=" in str(explanation)


def test_per_call_none_clears_configured_registry_default():
    registry = EnvironmentRegistry()
    registry.register("configured", PythonExecutableSpec("/configured/python"))
    dispatcher = Dispatcher(environment_registry=registry)

    configured = dispatcher.explain(make_function_call_spec("operator:add", args=[1, 2]))
    cleared = dispatcher.explain(make_function_call_spec("operator:add", args=[1, 2]), environment_registry=None)

    assert configured.resolution.environment_selection.source == "resolver"
    assert cleared.resolution.environment_selection.source == "fallback"


def test_dispatcher_rejects_retained_one_shot_candidate_iterators():
    with pytest.raises(TypeError, match="re-iterable"):
        Dispatcher(environment_candidates=iter((PythonExecutableSpec("/candidate/python"),)))


def test_dispatcher_rejects_retained_one_shot_candidate_iterable_wrappers():
    class OneShotCandidates:
        def __init__(self):
            self._iterator = iter((PythonExecutableSpec("/candidate/python"),))

        def __iter__(self):
            return self._iterator

    with pytest.raises(TypeError, match="re-iterable"):
        Dispatcher(environment_candidates=OneShotCandidates())


def test_dispatcher_rejects_retained_distinct_iterators_over_a_shared_cursor():
    class SharedCursorCandidates:
        def __init__(self):
            self._cursor = iter((PythonExecutableSpec("/candidate/python"),))

        def __iter__(self):
            return (candidate for candidate in self._cursor)

    with pytest.raises(TypeError, match="re-iterable"):
        Dispatcher(environment_candidates=SharedCursorCandidates())


def test_dispatch_reuses_selected_resolver_record_without_a_second_probe(monkeypatch):
    import dryml.dispatch.requirements as requirements

    candidate = PythonExecutableSpec("/resolved/python")
    calls = []
    record = replace(inspect_current(), tags=("resolved",))

    def probe(spec, **_kwargs):
        calls.append(spec)
        return requirements.environments.EnvironmentProbeResult(spec, True, record=record)

    monkeypatch.setattr(requirements.environments, "probe", probe)
    resolution = resolve_dispatch_plan(
        normalize_user_operation(resolver_target, allow_pickle=True),
        environment_candidates=(candidate,),
        requirement_policy="strict",
    )

    assert resolution.environment_selection.source == "resolver"
    assert resolution.environment_record == record
    assert calls == [candidate]
    metadata = resolution.metadata()
    assert metadata["dryml.environment_resolution"]["selected"]["executable"] == "/resolved/python"
    assert metadata["dryml.environment_probe"] is not None


def test_annotation_default_world_is_not_replaced_by_synthesis(monkeypatch):
    import dryml.dispatch.requirements as requirements

    monkeypatch.setattr(requirements.worlds, "synthesize", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("synthesis must not run")))
    resolution = resolve_dispatch_plan(
        normalize_user_operation(incompatible_default_world_target, allow_pickle=True),
        inventory=LocalResourceInventory((0, 1)),
        requirement_policy="strict",
    )

    assert resolution.world_selection.source == "annotation_default"
    assert resolution.world_check.status == "incompatible"


def test_incompatible_environment_default_and_current_do_not_start_resolver_search():
    def candidates():
        raise AssertionError("resolver must not inspect candidates")
        yield PythonExecutableSpec("/unreachable/python")

    default = resolve_dispatch_plan(
        normalize_user_operation(incompatible_default_environment_target, allow_pickle=True),
        environment_candidates=candidates(),
        requirement_policy="strict",
    )
    with use(CurrentEnvironmentSpec()):
        current = resolve_dispatch_plan(
            normalize_user_operation(incompatible_current_environment_target, allow_pickle=True),
            environment_candidates=candidates(),
            requirement_policy="strict",
        )

    assert default.environment_selection.source == "annotation_default"
    assert default.environment_resolution is None
    assert current.environment_selection.source == "current"
    assert current.environment_resolution is None


def test_incompatible_current_world_is_not_replaced_by_synthesis(monkeypatch):
    import dryml.dispatch.requirements as requirements

    monkeypatch.setattr(requirements.worlds, "synthesize", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("synthesis must not run")))
    current_world = WorldSpec.from_data({"roles": {"main": {"replicas": 1, "process": {"resources": {"cpus": 1}}}}})

    with use_world(current_world):
        resolution = resolve_dispatch_plan(
            normalize_user_operation(incompatible_current_world_target, allow_pickle=True),
            inventory=LocalResourceInventory((0, 1)),
            requirement_policy="strict",
        )

    assert resolution.world_selection.source == "current"
    assert resolution.world_check.status == "incompatible"


def test_incompatible_explicit_candidates_do_not_start_lower_precedence_search(monkeypatch):
    import dryml.dispatch.requirements as requirements

    def candidates():
        raise AssertionError("resolver must not inspect candidates")
        yield PythonExecutableSpec("/unreachable/python")

    monkeypatch.setattr(requirements.worlds, "synthesize", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("synthesis must not run")))
    environment = resolve_dispatch_plan(
        normalize_user_operation(incompatible_current_environment_target, allow_pickle=True),
        environment=CurrentEnvironmentSpec(),
        environment_candidates=candidates(),
        requirement_policy="strict",
    )
    world = resolve_dispatch_plan(
        normalize_user_operation(cpu_target, allow_pickle=True),
        world=WorldSpec.from_data({"roles": {"main": {"replicas": 1, "process": {"resources": {"cpus": 1}}}}}),
        inventory=LocalResourceInventory((0, 1)),
        requirement_policy="strict",
    )

    assert environment.environment_selection.source == "explicit"
    assert environment.environment_resolution is None
    assert world.world_selection.source == "explicit"
    assert world.world_synthesis is None
