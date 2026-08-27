import pytest

from dryml.runtime import EffectPlan, PublicationService, RuntimeState
from dryml.runtime.errors import PublicationError


def test_windows_environment_ownership_is_case_insensitive_and_missing_affinity_is_honest():
    environment = {"Path": "old"}
    service = PublicationService(environ=environment, windows=True, affinity_getter=lambda: (_ for _ in ()).throw(PublicationError("unsupported")))
    service.initialize(RuntimeState())
    service.commit(service.stage(service.current(), RuntimeState()), EffectPlan(environment={"PATH": "new"}))
    assert environment == {"Path": "new"}
    with pytest.raises(PublicationError):
        service.commit(service.stage(service.current(), RuntimeState()), EffectPlan(cpu_affinity=(0,)))


def test_windows_case_collisions_fail_before_ambiguous_ownership():
    with pytest.raises(PublicationError, match="case-colliding"):
        PublicationService(environ={"Path": "one", "PATH": "two"}, windows=True)


def test_injected_process_memory_effect_is_owned_and_resettable():
    memory = {"value": 100}
    service = PublicationService(
        environ={},
        process_memory_getter=lambda: memory["value"],
        process_memory_setter=lambda value: memory.__setitem__("value", value),
    )
    service.initialize(RuntimeState())

    service.commit(service.stage(service.current(), RuntimeState()), EffectPlan(process_memory=50))
    assert memory["value"] == 50
    service.reset(RuntimeState())
    assert memory["value"] == 100


def test_injected_releases_restore_owned_environment_affinity_and_memory():
    """Leaving a controlling runtime restores each prior session-owned value."""

    environment = {"CUSTOM": "before", "CUDA_VISIBLE_DEVICES": "all"}
    affinity = {"value": (0, 1)}
    memory = {"value": 100}
    service = PublicationService(
        environ=environment,
        affinity_getter=lambda: affinity["value"],
        affinity_setter=lambda value: affinity.__setitem__("value", value),
        process_memory_getter=lambda: memory["value"],
        process_memory_setter=lambda value: memory.__setitem__("value", value),
    )
    service.initialize(RuntimeState())
    service.commit(service.stage(service.current(), RuntimeState()), EffectPlan(environment={"CUSTOM": "session", "CUDA_VISIBLE_DEVICES": "0"}, cpu_affinity=(0,), process_memory=50))

    service.commit(service.stage(service.current(), RuntimeState()), EffectPlan(environment={"CUDA_VISIBLE_DEVICES": ""}, release_environment=("CUSTOM",), release_cpu_affinity=True, release_process_memory=True))

    assert environment == {"CUSTOM": "before", "CUDA_VISIBLE_DEVICES": ""}
    assert affinity["value"] == (0, 1)
    assert memory["value"] == 100
    service.reset(RuntimeState())
    assert environment == {"CUSTOM": "before", "CUDA_VISIBLE_DEVICES": "all"}
