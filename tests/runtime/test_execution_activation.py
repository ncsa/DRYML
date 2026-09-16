import pytest

from dryml.runtime import (
    ExecutionGrant,
    PublicationFailedError,
    PublicationService,
    RuntimeContextSpec,
    RuntimeMode,
    RuntimeState,
    activation_scope,
)
from dryml.runtime.errors import RuntimeTransitionError
from dryml.execute.models import WorkerSetupContext
from dryml.worlds import ProcessAllocation, WorldAllocation


def _world():
    process = ProcessAllocation(replica=0, rank=0, local_rank=0, cpus=(3, 4))
    return WorldAllocation({"main": (process,)})


def test_exact_and_logical_grants_preserve_their_distinct_evidence():
    exact = ExecutionGrant.from_world_allocation(_world(), role="main")
    logical = ExecutionGrant.ray(logical_cpus=2, native_evidence={"node_id": "node"})

    assert exact.exact_cpus == (3, 4)
    assert exact.world_allocation_id == _world().semantic_id
    assert logical.exact_cpus == ()
    assert logical.world_allocation_id is None
    assert logical.thread_capacity == 2

    service = PublicationService(environ={})
    service.initialize(RuntimeState())
    with pytest.raises(RuntimeTransitionError, match="exact allocation"):
        with activation_scope(
            RuntimeContextSpec(RuntimeMode.INLINE, world_allocation_id=exact.world_allocation_id),
            logical,
            service=service,
        ):
            pass

    affinity = {"value": (9,)}
    exact_service = PublicationService(
        environ={}, affinity_getter=lambda: affinity["value"],
        affinity_setter=lambda value: affinity.__setitem__("value", value),
    )
    exact_service.initialize(RuntimeState())
    with activation_scope(
        RuntimeContextSpec(RuntimeMode.INLINE, world_allocation_id=exact.world_allocation_id),
        exact, service=exact_service,
    ) as active:
        assert active.state.allocation.cpus == (3, 4)
        assert affinity["value"] == (3, 4)
    assert affinity["value"] == (9,)

    logical_service = PublicationService(environ={})
    logical_service.initialize(RuntimeState())
    with activation_scope(
        RuntimeContextSpec(RuntimeMode.INLINE, framework={"torch": {"threads": 2}}), logical,
        service=logical_service,
    ) as active:
        assert active.state.allocation.cpus == ()
        assert active.state.allocation.logical_cpus == 2
        assert active.state.allocation.grant_provenance == "logical"
        assert logical_service.current().statuses["threading"] == "pending-import"
    assert "OMP_NUM_THREADS" not in logical_service._environ

    with pytest.raises(RuntimeTransitionError, match="exceeds verified backend CPU capacity"):
        with activation_scope(
            RuntimeContextSpec(RuntimeMode.INLINE, framework={"torch": {"threads": 3}}), logical,
            service=logical_service,
        ):
            pass
    with pytest.raises(RuntimeTransitionError, match="framework-owned"):
        with activation_scope(
            RuntimeContextSpec(RuntimeMode.INLINE, limits={"threads": 1}), logical,
            service=logical_service,
        ):
            pass


def test_ray_worker_setup_uses_only_verified_logical_resource_evidence():
    context = WorkerSetupContext(
        submission_id="ray-grant",
        backend="ray",
        environment=None,
        allocation=None,
        native_grant={
            "kind": "ray",
            "resources": {"CPU": 2.0},
            "memory_bytes": 4096,
            "accelerator_ids": {"GPU": ["0"]},
        },
    )
    grant = ExecutionGrant.from_worker_setup(context)

    assert grant.exact_cpus == ()
    assert grant.logical_cpus == 2
    assert grant.memory == 4096
    assert grant.accelerators == {"GPU": ("0",)}

    fractional = WorkerSetupContext(
        submission_id="ray-fractional",
        backend="ray",
        environment=None,
        allocation=None,
        native_grant={"kind": "ray", "resources": {"CPU": 0.5}},
    )
    with pytest.raises(RuntimeTransitionError, match="integer logical CPU"):
        ExecutionGrant.from_worker_setup(fractional)

    malformed_accelerators = WorkerSetupContext(
        submission_id="ray-malformed-accelerator",
        backend="ray",
        environment=None,
        allocation=None,
        native_grant={
            "kind": "ray",
            "resources": {"CPU": 1.0},
            "accelerator_ids": {"GPU": "0"},
        },
    )
    with pytest.raises(RuntimeTransitionError, match="accelerator IDs are invalid"):
        ExecutionGrant.from_worker_setup(malformed_accelerators)


def test_activation_restores_owned_effects_and_marks_failed_rollback_terminal():
    environment = {"DRYML_TEST": "old"}
    service = PublicationService(environ=environment)
    service.initialize(RuntimeState())
    grant = ExecutionGrant.ray(logical_cpus=1)
    spec = RuntimeContextSpec(RuntimeMode.INLINE, env={"DRYML_TEST": "new"})

    with activation_scope(spec, grant, service=service):
        assert environment["DRYML_TEST"] == "new"
    assert environment["DRYML_TEST"] == "old"

    service = PublicationService(environ={})
    service.initialize(RuntimeState())
    with pytest.raises(PublicationFailedError):
        with activation_scope(spec, grant, service=service):
            service._environ["DRYML_TEST"] = "lost"
    assert service.current().health == "failed"


def test_activation_rollback_failure_retains_the_original_body_error():
    service = PublicationService(environ={})
    service.initialize(RuntimeState())
    spec = RuntimeContextSpec(RuntimeMode.INLINE, env={"DRYML_TEST": "new"})

    with pytest.raises(PublicationFailedError) as raised:
        with activation_scope(spec, ExecutionGrant.ray(logical_cpus=1), service=service):
            service._environ["DRYML_TEST"] = "lost"
            raise ValueError("workload failed")

    assert isinstance(raised.value.__context__, ValueError)
    assert service.current().health == "failed"


def test_incompatible_baseline_fails_before_entering_store_factory():
    service = PublicationService(environ={})
    service.initialize(RuntimeState(RuntimeMode.ORCHESTRATOR))
    with pytest.raises(RuntimeTransitionError, match="healthy NONE baseline"):
        with activation_scope(RuntimeContextSpec(RuntimeMode.INLINE), ExecutionGrant.ray(logical_cpus=1), service=service):
            raise AssertionError("activation must not enter the body")
