"""Facade transition safety contracts."""

from __future__ import annotations

import sys
import uuid
import warnings
from contextvars import copy_context

import pytest

from dryml import session
from dryml.dispatch import normalize_user_operation, resolve_dispatch_plan
from dryml.environments import PythonExecutableSpec
from dryml._framework_imports import finder
from dryml.core import Object, definition_mode, selector_mode, space_mode
from dryml.core.session import config, configure, current_object_mode, get_config, status
from dryml.runtime import NoAllocation, RuntimeAllocationView, RuntimeEnforcement, RuntimeMode, RuntimeState, active_runtime, assert_control_plane_target_execution_allowed, assert_object_materialization_allowed, enter_runtime, plain
from dryml.runtime.guards import internal_construction_admitted
from dryml.runtime.errors import RuntimeTransitionError
from dryml.runtime.frameworks import FrameworkRegistration, framework_registry
from dryml.runtime.publication import PublicationService
from dryml.worlds import LocalResourceInventory


@pytest.fixture(autouse=True)
def isolated_session(monkeypatch):
    import dryml.session.state as state

    affinity = {0, 1}
    service = PublicationService(
        environ={},
        affinity_getter=lambda: affinity,
        affinity_setter=lambda cpus: (affinity.clear(), affinity.update(cpus)),
    )
    service.initialize(RuntimeState(enforcement=RuntimeEnforcement.OFF))
    monkeypatch.setattr(state, "publication", service)
    import dryml.runtime.context as context
    import dryml.core.session as core_session
    import dryml.runtime.guards as guards

    monkeypatch.setattr(context, "publication", service)
    monkeypatch.setattr(core_session, "publication", service, raising=False)
    monkeypatch.setattr(guards, "publication", service, raising=False)
    monkeypatch.setattr(state, "local_inventory", lambda: LocalResourceInventory((0, 1), {"gpu": (0, 1)}, memory=4 * 1024**3))
    core_session.reset_config()


class FloorObject(Object):
    initialized = 0

    def __init__(self, value):
        super().__init__()
        type(self).initialized += 1
        self.value = value


def test_orchestrator_object_mode_floor_projects_and_rejects_materializing_contexts():
    configure(object_mode="load_or_build")
    before = get_config()
    session.set_mode("orchestrator")

    assert current_object_mode() == "definition"
    assert status()["object_mode"] == "definition"
    assert isinstance(FloorObject(1), __import__("dryml").Definition)

    with definition_mode(concrete=True):
        assert current_object_mode() == "concrete"
        with selector_mode():
            assert current_object_mode() == "selector"
            with space_mode():
                assert current_object_mode() == "space"
            assert current_object_mode() == "selector"
        assert current_object_mode() == "concrete"

    for enter in (
        lambda: config(object_mode="fresh"),
        lambda: config(object_mode="load_or_build"),
        lambda: definition_mode(False),
        lambda: selector_mode(False),
        lambda: space_mode(False),
    ):
        with pytest.raises(RuntimeTransitionError, match="orchestration"):
            with enter():
                pass
        assert get_config() == before

    session.reset()
    assert current_object_mode() == "load_or_build"


def test_orchestrator_worker_environment_and_zero_gpu_world_remain_independent_requests():
    requested_environment = PythonExecutableSpec(sys.executable)
    session.set_mode("orchestrator")
    session.worker_env_request(requested_environment)
    snapshot = session.worker_world_request(cpus=2, gpus=0)

    resolution = resolve_dispatch_plan(
        normalize_user_operation(lambda: None, allow_pickle=True),
        session_snapshot=snapshot,
    )

    assert snapshot.runtime is not None
    assert snapshot.runtime.allocation is NoAllocation
    assert snapshot.requested_environment == requested_environment
    assert snapshot.requested_world is not None
    assert resolution.environment_selection.source == "session_requested"
    assert resolution.environment_selection.candidate["kind"] == "python"
    assert resolution.world_selection.source == "session_requested"
    assert resolution.world_selection.candidate["roles"]["worker"]["process"]["resources"] == {"cpus": 2}


def test_definition_build_guard_has_stable_diagnostic_and_warn_off_admission():
    definition = FloorObject.defn(1)
    session.set_mode("orchestrator")

    with pytest.raises(RuntimeTransitionError, match="Orchestration mode prohibits Object materialization") as exc_info:
        definition.build()
    assert exc_info.value.context == {
        "mode": "orchestrator",
        "enforcement": "strict",
        "operation": "definition_build",
        "fix": "use Definition/CDef APIs for metadata, or execute in a managed inline session or dispatched worker",
    }
    assert FloorObject.initialized == 0

    with enter_runtime(RuntimeMode.ORCHESTRATOR, enforcement="warn"):
        with pytest.warns(RuntimeWarning, match="Orchestration mode prohibits Object materialization") as caught:
            built = definition.build()
    assert isinstance(built, FloorObject)
    assert len(caught) == 1
    assert status()["object_mode"] == "definition"

    with plain():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            built = definition.build()
    assert isinstance(built, FloorObject)
    assert not caught


def test_floor_projects_into_copied_contexts_and_guard_admission_expires():
    configure(object_mode="fresh")
    copied = copy_context()
    session.set_mode("orchestrator")

    assert copied.run(current_object_mode) == "definition"
    captured = None
    with plain():
        with assert_object_materialization_allowed(operation="test_materialization"):
            assert internal_construction_admitted()
            with pytest.raises(RuntimeTransitionError, match="orchestration"):
                with config(object_mode="fresh"):
                    pass
            captured = copy_context()
    assert captured is not None
    assert not captured.run(internal_construction_admitted)


def test_materialization_guard_lease_blocks_effectful_orchestrator_transition():
    with assert_object_materialization_allowed(operation="test_materialization"):
        with pytest.raises(RuntimeTransitionError, match="lease"):
            session.set_mode("orchestrator")


def test_local_execution_guard_uses_a_distinct_orchestrator_diagnostic():
    session.set_mode("orchestrator")

    with pytest.raises(RuntimeTransitionError, match="Orchestration mode prohibits local workload execution") as exc_info:
        with assert_control_plane_target_execution_allowed(operation="managed_method_call"):
            pass
    assert "Object materialization" not in str(exc_info.value)
    assert exc_info.value.context["operation"] == "managed_method_call"


def test_exact_allocation_requires_an_unambiguous_process_and_respects_inventory():
    allocation = {
        "schema": "dryml.world_allocation.v1",
        "payload": {"roles": {"worker": [
            {"replica": 0, "rank": 0, "local_rank": 0, "resources": {"cpus": [0], "accelerators": {}}},
            {"replica": 1, "rank": 1, "local_rank": 1, "resources": {"cpus": [1], "accelerators": {}}},
        ]}},
    }
    with pytest.raises(ValueError, match="selectors"):
        session.allocate_world(allocation)
    snapshot = session.allocate_world(allocation, role="worker", replica=1)
    assert snapshot.allocation.process.cpus == (1,)


def test_inventory_drift_restages_without_process_effects(monkeypatch):
    import dryml.session.state as state

    first = LocalResourceInventory((0, 1), {}, memory=4 * 1024**3)
    second = LocalResourceInventory((0,), {}, memory=4 * 1024**3)
    values = iter((first, second, second, second))
    monkeypatch.setattr(state, "local_inventory", lambda: next(values))

    snapshot = session.manage()

    assert snapshot.allocation.process.cpus == (0,)


def test_volatile_memory_drift_does_not_invalidate_visibility_epoch(monkeypatch):
    import dryml.session.state as state

    first = LocalResourceInventory((0, 1), {}, memory=4 * 1024**3)
    second = LocalResourceInventory((0, 1), {}, memory=3 * 1024**3)
    values = iter((first, second))
    monkeypatch.setattr(state, "local_inventory", lambda: next(values))

    snapshot = session.manage()

    assert snapshot.inventory is first
    assert snapshot.allocation.process.memory == 4 * 1024**3


def test_late_framework_import_rejects_visibility_change_without_mutating(monkeypatch):
    import dryml.session.state as state

    session.manage(cpus=1)
    before = session.current()
    monkeypatch.setattr(state, "_loaded_framework_roots", lambda: ("torch",))

    with pytest.raises(RuntimeError, match="restart"):
        session.manage(cpus=2)
    assert session.current() == before


def test_active_sessions_publish_all_builtin_adapter_plans_and_pending_statuses():
    import sys
    import dryml.session.state as state

    before = set(sys.modules)
    snapshot = session.set_mode(mode="orchestrator")
    results = state.publication.current().metadata["framework_results"]

    assert {"tensorflow", "torch", "jax"}.issubset(results)
    assert all(name not in sys.modules or name in before for name in ("tensorflow", "torch", "jax", "jaxlib"))
    assert snapshot.statuses["tensorflow:tensorflow:visibility"] == "pending-import"
    assert snapshot.statuses["torch:torch:threads"] == "pending-import"
    assert snapshot.statuses["jax:jaxlib:allocator"] == "pending-import"
    with pytest.raises(RuntimeError, match="frozen"):
        framework_registry.register(FrameworkRegistration("late", ("dryml_late_framework",), "dryml.tf.runtime:adapter"))


def test_adapter_plan_change_after_a_framework_import_fails_without_publication(monkeypatch):
    import dryml.session.state as state

    session.set_mode("orchestrator")
    before = session.current()
    monkeypatch.setattr(state, "_loaded_framework_roots", lambda: ("torch",))

    with pytest.raises(RuntimeTransitionError, match="restart"):
        session.manage(cpus=1)

    assert session.current() == before


def test_loaded_framework_roots_include_registered_retained_descendants(monkeypatch):
    import dryml.session.state as state

    name = "dryml_fake_" + uuid.uuid4().hex
    registry = type(framework_registry)()
    registry.register(FrameworkRegistration(name, (name,), object()))
    monkeypatch.setattr(state, "framework_registry", registry)
    monkeypatch.setitem(sys.modules, name + ".retained_after_failure", object())

    assert name in state._loaded_framework_roots()


def test_reset_deactivates_builtin_plans_without_removing_the_passive_finder():
    import dryml.session.state as state

    session.set_mode("orchestrator")
    reset = session.reset()
    metadata = state.publication.current().metadata

    assert reset.mode == "python"
    assert not metadata["session_active"]
    assert not {"tensorflow", "torch", "jax"} & set(metadata["framework_results"])
    assert finder in __import__("sys").meta_path


def test_context_override_restores_the_session_baseline_and_cannot_broaden_visibility():
    session.manage(cpus=1)
    baseline = active_runtime()

    with enter_runtime(RuntimeMode.WORKER, RuntimeAllocationView(cpus=(0,)), enforcement="warn"):
        assert active_runtime().enforcement is RuntimeEnforcement.WARN
    assert active_runtime() == baseline

    with pytest.raises(RuntimeTransitionError, match="broadens session CPU visibility"):
        with enter_runtime(RuntimeMode.WORKER, RuntimeAllocationView(cpus=(2,))):
            pass
