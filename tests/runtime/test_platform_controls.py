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
