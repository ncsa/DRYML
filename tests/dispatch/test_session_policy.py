from __future__ import annotations

from dryml.core.store.dir import DirStore
from dryml.dispatch import Dispatcher, normalize_user_operation, resolve_dispatch_plan
from dryml.environments import EnvironmentRequirement
from dryml.session.model import SessionSnapshot
from dryml.worlds import LocalResourceInventory, WorldSpec


def _managed_snapshot() -> SessionSnapshot:
    return SessionSnapshot(
        mode="managed",
        resources=None,
        allocation=None,
        requested_world=WorldSpec.from_data(
            {
                "roles": {
                    "worker": {
                        "replicas": 1,
                        "process": {
                            "resources": {
                                "cpus": 1,
                                "accelerators": {"gpu": 1},
                            }
                        },
                    }
                }
            }
        ),
        environment=EnvironmentRequirement(requirements=("session-worker-package>=1",)),
        controls={},
        statuses={"visibility": "visibility-enforced"},
        runtime=None,
        generation=17,
        inventory=LocalResourceInventory((0,), {"gpu": ("gpu-a",)}),
    )


def test_managed_session_uses_one_snapshot_for_worker_intent_requirements_and_inventory():
    snapshot = _managed_snapshot()

    resolution = resolve_dispatch_plan(
        normalize_user_operation(lambda: None, allow_pickle=True),
        requirement_policy="ignore",
        session_snapshot=snapshot,
    )

    assert resolution.world_selection.source == "session_requested"
    assert resolution.world_selection.candidate["roles"]["worker"]["process"]["resources"]["accelerators"] == {"gpu": 1}
    assert resolution.requirements.environment_requirement.requirements == (
        "session-worker-package>=1",
    )
    assert resolution.local_inventory == snapshot.inventory
    assert resolution.metadata()["dryml.session"] == {
        "generation": 17,
        "mode": "managed",
        "control_statuses": {"visibility": "visibility-enforced"},
    }


def test_public_planner_uses_the_pinned_session_snapshot(monkeypatch, tmp_path):
    snapshot = _managed_snapshot()
    monkeypatch.setattr("dryml.dispatch.planner._session_snapshot", lambda: snapshot)

    plan = Dispatcher(store=DirStore(tmp_path / "store", query_index="none")).plan(
        {"schema": "dryml.operation.v1", "schema_version": 1, "kind": "function_call", "payload": {"function": "operator:add", "args": [1, 2]}},
        requirement_policy="ignore",
    )

    assert plan.resolution.world_selection.source == "session_requested"
    assert plan.envelope.allocation_view["accelerators"] == {"gpu": ["gpu-a"]}
    assert plan.dispatch_spec["payload"]["metadata"]["dryml.session"]["generation"] == 17
