from __future__ import annotations

import json

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
