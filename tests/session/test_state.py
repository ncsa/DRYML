"""Persistent session state-transition contracts."""

import threading

import pytest

from dryml.session import SessionConfigurationError, configure, current, enforce_requirements, manage, mode, reset, set_mode
from dryml.session import state
from dryml.runtime import PublicationFailedError, PublicationService, RuntimeState


def test_mode_transitions_apply_defaults_and_preserve_managed_allocation(session_runtime):
    """Managed repetition keeps exact allocation while mode changes clear it."""

    first = manage(cpus=1, gpus=1, accelerator_memory="1GiB")
    same = set_mode("managed")
    orchestrator = set_mode("orchestrator")

    assert same.allocation == first.allocation
    assert same.requirement_axes == {"environment": True, "runtime": True, "world": True}
    assert orchestrator.allocation is None
    assert mode() == "orchestrator"
    assert reset().mode == "python"


def test_public_session_failures_include_operation_and_category(session_runtime):
    """Diagnostics identify the public operation and failure class."""

    with pytest.raises(SessionConfigurationError) as error:
        set_mode("worker")

    assert error.value.context["operation"] == "session.set_mode"
    assert error.value.context["category"] == "malformed"

    with pytest.raises(SessionConfigurationError) as error:
        state.require_env("api_key=supersecret")
    assert "supersecret" not in str(error.value)
    assert "supersecret" not in repr(error.value.context)


def test_manage_and_allocate_preserve_axes(session_runtime):
    """Only explicit axis replacement changes parity-only state."""

    enforce_requirements(environment=False, world=True, runtime=False)
    snapshot = manage(cpus=1)

    assert snapshot.requirement_axes == {"environment": False, "runtime": False, "world": True}


def test_configure_replaces_categories_and_passes_restage_retries(session_runtime, monkeypatch):
    """Complete replacement defaults omissions and forwards local retry control."""

    calls = []
    original = session_runtime.publish

    def publish(*args, **kwargs):
        calls.append(kwargs["restage_retries"])
        return original(*args, **kwargs)

    monkeypatch.setattr(session_runtime, "publish", publish)
    first = configure(mode="managed", resources={"cpus": 1}, restage_retries=0)
    second = configure(mode="orchestrator", restage_retries=16)

    assert first.resources.cpus == 1
    assert second.resources is None
    assert calls == [0, 16]
    with pytest.raises(Exception):
        configure(mode="python", restage_retries=17)
    assert current().mode == "orchestrator"


def test_configure_delegates_closed_validation_to_normalization():
    """Replacement configuration rejects source-v1 fields through the pure contract."""

    with pytest.raises(Exception, match="source-v1"):
        configure(mode="python", source="v1")


def test_parity_only_update_preserves_current_framework_finalization(session_runtime):
    """Status finalization remains current across an effect-free axis update."""

    manage(cpus=1)
    admission = session_runtime.admit_status_finalization()
    session_runtime.finalize_statuses(admission, {"torch:visibility": "visibility-enforced"})

    snapshot = enforce_requirements(environment=False, world=True, runtime=True)

    assert snapshot.statuses["torch:visibility"] == "visibility-enforced"


def test_equivalent_python_reset_and_configure_preserve_finalized_generation(session_runtime):
    """Semantic baseline no-ops retain both the generation and final statuses."""

    admission = session_runtime.admit_status_finalization()
    finalized = session_runtime.finalize_statuses(admission, {"visibility": "visibility-enforced"})

    assert set_mode("python").generation == finalized.number
    assert configure(mode="python").generation == finalized.number
    assert reset().generation == finalized.number
    assert current().statuses["visibility"] == "visibility-enforced"


def test_managed_effects_own_allocation_environment_affinity_and_restore_on_exit(monkeypatch):
    """Session transitions restore non-visibility effects without touching host affinity."""

    environment = {"CUSTOM": "before", "CUDA_VISIBLE_DEVICES": "all"}
    affinity = {"value": (0, 1)}
    memory = {"value": 16 * 1024**3}
    service = PublicationService(
        environ=environment,
        affinity_getter=lambda: affinity["value"],
        affinity_setter=lambda value: affinity.__setitem__("value", value),
        process_memory_getter=lambda: memory["value"],
        process_memory_setter=lambda value: memory.__setitem__("value", value),
    )
    service.initialize(RuntimeState())
    monkeypatch.setattr(state, "publication", service)
    from dryml.worlds import LocalResourceInventory
    inventory = LocalResourceInventory((0, 1), {"gpu": (0,)}, memory=8 * 1024**3, accelerator_memory={"gpu": {0: 4 * 1024**3}})
    monkeypatch.setattr(state, "local_inventory", lambda: inventory)

    from dryml.worlds import ProcessAllocation, WorldAllocation
    allocation = WorldAllocation({"main": (ProcessAllocation(0, 0, 0, cpus=(0,), memory="1GiB", accelerators={"gpu": (0,)}, env={"CUSTOM": "session"}),)})
    managed = configure(mode="managed", allocation={"value": allocation})

    assert environment["CUSTOM"] == "session"
    assert environment["CUDA_VISIBLE_DEVICES"] == "0"
    assert affinity["value"] == (0,)
    assert memory["value"] == 1024**3
    assert managed.statuses["process_memory"] == "enforced"

    orchestrator = set_mode("orchestrator")
    assert environment["CUSTOM"] == "before"
    assert environment["CUDA_VISIBLE_DEVICES"] == ""
    assert affinity["value"] == (0, 1)
    assert memory["value"] == 16 * 1024**3
    assert orchestrator.statuses["process_memory"] == "not-applicable"

    reset()
    assert environment["CUDA_VISIBLE_DEVICES"] == "all"


def test_process_memory_is_unsupported_without_a_setter(monkeypatch):
    """A declarative memory request remains publishable without a false effect claim."""

    service = PublicationService(environ={}, affinity_getter=lambda: (0,), affinity_setter=lambda value: None)
    service.initialize(RuntimeState())
    monkeypatch.setattr(state, "publication", service)
    from dryml.worlds import LocalResourceInventory
    monkeypatch.setattr(state, "local_inventory", lambda: LocalResourceInventory((0,), memory=2 * 1024**3))

    snapshot = configure(mode="managed", resources={"cpus": 1, "memory": "1GiB"})

    assert snapshot.controls["memory"] == "declarative"
    assert snapshot.statuses["process_memory"] == "unsupported"


def test_terminal_runtime_failure_never_returns_session_success(session_runtime):
    """Facade mutation propagates terminal publication health without a snapshot."""

    failed = session_runtime.fail_status_finalization(None, RuntimeError("synthetic"))
    with pytest.raises(PublicationFailedError, match="restart") as error:
        set_mode("orchestrator")
    assert error.value.context["operation"] == "session.set_mode"
    assert error.value.context["category"] == "terminal"
    assert current().health == "failed"
    assert current().generation == failed.number


def test_concurrent_inspection_never_observes_torn_session_categories(session_runtime):
    """Readers see complete immutable mode/allocation pairs during transitions."""

    failures = []
    finished = threading.Event()

    def mutate():
        try:
            for _ in range(10):
                manage(cpus=1)
                set_mode("orchestrator")
                reset()
        except BaseException as exc:
            failures.append(exc)
        finally:
            finished.set()

    thread = threading.Thread(target=mutate)
    thread.start()
    while not finished.is_set():
        snapshot = current()
        if (snapshot.mode == "managed") != (snapshot.allocation is not None):
            failures.append(AssertionError("torn session categories"))
            break
    thread.join(timeout=3)

    assert not thread.is_alive()
    assert not failures
