"""Explicit caller-supplied Ray acceptance for the core Execute facade."""

from __future__ import annotations

from concurrent.futures import CancelledError
from pathlib import Path
from time import monotonic, sleep

import pytest

from dryml.core import ConcreteDefinition, Executor as CoreExecutor
from dryml.core import Object, ObjectRef, Repo, Serializable, StateRef
from dryml.core.execute import CoreOptions
from dryml.core.store.dir import DirStore
from dryml.execute.ray import RayBackendConfig, RayFuture
from dryml.managed import managed_operation
from dryml.worlds import CountConstraint, ResourceRequirement, RoleRequirement, WorldRequirement
from .conftest import require_ray_integration
from tests.managed.execution_fixtures import (
    InnerFunctionWrappedManagedValue,
    ManagedMatrixValue,
    MatrixArgument,
    ORDERS,
    OuterWrappedManagedValue,
    matrix_config,
    nested_managed_advance,
)


class RayTrainingValue(Serializable):
    """Persist a deterministic training scalar through the supplied shared Store."""

    def __init__(self, value=0):
        self.value = value

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        """Write the value used to prove shared-path result recovery."""
        Path(dest_dir, "value").write_text(str(self.value), encoding="ascii")

    def restore_state_from_dir_imp(self, src_dir, *, codec):
        """Restore the value published by the one-shot Ray worker."""
        self.value = int(Path(src_dir, "value").read_text(encoding="ascii"))


class RayStatelessResult(Object):
    """Supply a definition-only result alongside a stateful training result."""


class RayTrainingRoot(Serializable):
    """Keep a descendant whose update is projected from its saved root snapshot."""

    def __init__(self, child):
        self.child = child


def _one_cpu_world() -> WorldRequirement:
    """Request the only portable Ray evidence: one scheduler logical CPU."""
    return WorldRequirement({
        "main": RoleRequirement(resources=ResourceRequirement(cpus=CountConstraint(1, 1))),
    })


def _ray_train(value):
    """Produce raw metrics plus stateful, stateless, and reference-valued results."""
    value.value += 5
    return RayTrainingValue(value.value), RayStatelessResult(), value.object_ref, value.value / 5


def _ray_update_child(root):
    """Return a changed child so caller refresh and result authority must agree."""
    root.child.value += 1
    return root.child


def _ray_control_store_path():
    """Read the selected control Store role installed during core worker setup."""
    from dryml.core.execute import current_context

    control_store = current_context().control_store
    return None if control_store is None else control_store.base_dir


def _ray_scalar():
    """Return one importable ordinary result through detached Ray authority."""
    return 5


def _ray_block(marker: Path) -> None:
    """Mark post-GO execution and remain alive until Ray cancels its exact task."""
    marker.write_text("running", encoding="ascii")
    import time

    time.sleep(30)


def _wait_for_marker(marker: Path, future, *, timeout: float = 30) -> None:
    """Wait for user-code authorization rather than relying on task scheduling sleeps."""
    deadline = monotonic() + timeout
    while not marker.exists():
        assert not future.done(), "Ray worker ended before core user code started"
        if monotonic() >= deadline:
            raise TimeoutError("Ray worker did not start core user code")
        sleep(0.01)


def _executor(tmp_path: Path, repo: Repo, control_store: DirStore) -> CoreExecutor:
    """Attach only to the caller-supplied Ray endpoint with bounded local spooling."""
    return CoreExecutor(
        RayBackendConfig(
            address=require_ray_integration(), spool_directory=tmp_path,
            admission_timeout=90, connect_timeout=60, termination_timeout=10,
            invocation_limit_bytes=1_000_000, result_limit_bytes=1_000_000,
        ),
        core=CoreOptions(repo=repo, control_store=control_store, return_objects=False, update_args=True),
    )


def test_existing_ray_core_recovers_training_results_nested_updates_and_control_store(tmp_path: Path):
    """Run the complete shared-Store core flow on the explicit same-host Ray target."""
    repo = Repo(DirStore(tmp_path / "state", query_index="none"))
    control_store = DirStore(tmp_path / "control", query_index="none")
    value = RayTrainingValue(5, repo=repo)
    repo.save(value, deep_capture=True)
    root = RayTrainingRoot(RayTrainingValue(9, repo=repo), repo=repo)
    repo.save(root, deep_capture=True)
    executor = _executor(tmp_path, repo, control_store)
    try:
        future = executor.submit(_ray_train, value, world=_one_cpu_world())
        stateful, stateless, existing, metric = future.result(timeout=30)
        assert isinstance(stateful, StateRef)
        assert isinstance(stateless, ConcreteDefinition)
        assert isinstance(existing, ObjectRef)
        assert metric == 2.0
        assert value.value == 10
        assert isinstance(future.backend_future, RayFuture)
        assert future.backend_future.object_ref is not None
        assert future.backend_future.worker_id is not None
        assert future.backend_future.node_id is not None
        assert executor.resources(timeout=10).allocated.cpus == 1.0
        future.cleanup(timeout=30)

        nested = executor.submit(_ray_update_child, root, world=_one_cpu_world())
        returned = nested.result(timeout=30)
        assert isinstance(returned, StateRef)
        assert root.child.value == 10
        assert returned == root.last_state_ref.at(next(
            path for path, child in root._runtime_projection.items() if child is root.child
        ))
        nested.cleanup(timeout=30)

        control = executor.submit(_ray_control_store_path, world=_one_cpu_world())
        assert control.result(timeout=30) == str(control_store.base_dir)
        control.cleanup(timeout=30)
        assert executor.resources(timeout=10).allocated.cpus == 0.0
    finally:
        executor.close(cancel=True, timeout=30)


def test_existing_ray_core_accepts_a_detached_repo_definition(tmp_path: Path):
    """The supplied Ray path accepts reconstructed detached core authority."""
    store = DirStore(tmp_path / "state", query_index="none")
    definition = Repo(store).to_definition()
    executor = CoreExecutor(
        RayBackendConfig(
            address=require_ray_integration(), spool_directory=tmp_path,
            admission_timeout=90, connect_timeout=60, termination_timeout=10,
            invocation_limit_bytes=1_000_000, result_limit_bytes=1_000_000,
        ),
        core=CoreOptions(repo=definition, return_objects=False),
    )
    try:
        future = executor.submit(_ray_scalar, world=_one_cpu_world())
        assert future.result(timeout=30) == 5
        future.cleanup(timeout=30)
    finally:
        executor.close(cancel=True, timeout=30)


def test_existing_ray_core_cancellation_retains_native_evidence_until_cleanup(tmp_path: Path):
    """Cancel a post-GO core task without fabricating physical CPU affinity evidence."""
    repo = Repo(DirStore(tmp_path / "state", query_index="none"))
    control_store = DirStore(tmp_path / "control", query_index="none")
    executor = _executor(tmp_path, repo, control_store)
    try:
        marker = tmp_path / "running"
        future = executor.submit(_ray_block, marker, world=_one_cpu_world())
        _wait_for_marker(marker, future)
        backend_future = future.backend_future
        assert isinstance(backend_future, RayFuture)
        assert backend_future.object_ref is not None
        assert backend_future.worker_id is not None
        assert backend_future.worker_pid is not None
        assert executor.resources(timeout=10).allocated.cpus == 1.0
        assert future.request_cancel()
        with pytest.raises(CancelledError):
            future.result(timeout=30)
        assert future.snapshot().backend.cancel_requested
        future.cleanup(timeout=30)
        assert backend_future.object_ref is None
        assert executor.resources(timeout=10).allocated.cpus == 0.0
    finally:
        executor.close(cancel=True, timeout=30)


@pytest.mark.parametrize(
    "member", tuple(member for member, _ in ORDERS),
)
def test_existing_ray_core_runs_every_managed_decorator_order_once(
        tmp_path: Path, member: str):
    """Exercise all six A/F/M orders with real StateRef materialization on Ray."""

    repo = Repo(DirStore(tmp_path / "state", query_index="none"))
    control_store = DirStore(tmp_path / "control", query_index="none")
    value = ManagedMatrixValue(repo=repo)
    argument = MatrixArgument(3, repo=repo)
    argument_state = repo.save_object(argument, deep_capture=True)
    repo.save_object(value, deep_capture=True)
    executor = _executor(tmp_path / "ray", repo, control_store)
    try:
        operation = getattr(value, member)
        future = executor.submit(operation, argument_state, world=_one_cpu_world())
        assert isinstance(future.result(timeout=30), StateRef)
        future.cleanup(timeout=30)
        restored = repo.load_state_ref(
            operation.status(
                state_repo=repo, control_store=control_store,
            ).final_state_ref,
            reuse_live="never",
        )
        assert (restored.calls, restored.value, restored.save_calls) == (1, 3, 2)
    finally:
        executor.close(cancel=True, timeout=30)


@pytest.mark.parametrize(
    ("subject_type", "events"),
    (
        (OuterWrappedManagedValue, ["before", "body", "after"]),
        (InnerFunctionWrappedManagedValue, ["before", "body", "after"]),
    ),
)
def test_existing_ray_core_preserves_ordinary_managed_wrapper_composition(
        tmp_path: Path, subject_type, events):
    """Run W(M) and M(W(F)) before any local call on the supplied Ray target."""

    repo = Repo(DirStore(tmp_path / "state", query_index="none"))
    control_store = DirStore(tmp_path / "control", query_index="none")
    value = subject_type(repo=repo)
    repo.save_object(value, deep_capture=True)
    executor = _executor(tmp_path / "ray", repo, control_store)
    try:
        assert value.advance.status(
            state_repo=repo, control_store=control_store,
        ).state == "not_started"
        future = executor.submit(value.advance, 2, world=_one_cpu_world())
        assert future.result(timeout=30) == 12
        future.cleanup(timeout=30)
        restored = repo.load_state_ref(
            value.advance.status(
                state_repo=repo, control_store=control_store,
            ).final_state_ref,
            reuse_live="never",
        )
        assert restored.events == events
    finally:
        executor.close(cancel=True, timeout=30)


def test_existing_ray_core_transports_nested_managed_config(
        tmp_path: Path) -> None:
    """Resolve explicit selected authority from a nested ordinary Ray argument."""

    repo = Repo(DirStore(tmp_path / "state", query_index="none"))
    control_store = DirStore(tmp_path / "control", query_index="none")
    from tests.managed.execution_fixtures import DispatchDiscoveryValue

    value = DispatchDiscoveryValue(repo=repo)
    repo.save_object(value, deep_capture=True)
    executor = _executor(tmp_path / "ray", repo, control_store)
    try:
        nested_config = {"config": [matrix_config(repo, control_store)]}
        future = executor.submit(
            nested_managed_advance, value, 3, nested_config,
            world=_one_cpu_world(),
        )
        assert future.result(timeout=30) == 3
        future.cleanup(timeout=30)
        assert value.advance.status(
            state_repo=repo, control_store=control_store,
        ).state == "completed"
    finally:
        executor.close(cancel=True, timeout=30)
