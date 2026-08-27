import pytest

from dryml.runtime import EffectPlan, LocalResourceInventory, PublicationService, RuntimeAllocationView, RuntimeMode, RuntimeState
from dryml.runtime.errors import PublicationBusyError


def test_lease_blocks_incompatible_effect_publication_and_releases_after_exception():
    service = PublicationService(environ={})
    service.initialize(RuntimeState())
    with service.lease():
        candidate = service.stage(service.current(), RuntimeState())
        with pytest.raises(PublicationBusyError):
            service.commit(candidate, EffectPlan(environment={"X": "1"}))
    candidate = service.stage(service.current(), RuntimeState())
    assert service.commit(candidate, EffectPlan(environment={"X": "1"})).number == 1


def test_lease_blocks_runtime_only_transition_without_explicit_effects():
    service = PublicationService(environ={})
    service.initialize(RuntimeState())
    inline = RuntimeState(RuntimeMode.INLINE, RuntimeAllocationView(role="main", replica=0, cpus=(0,)))

    with service.lease():
        candidate = service.stage(service.current(), inline, inventory=LocalResourceInventory((0,)))
        with pytest.raises(PublicationBusyError):
            service.commit(candidate)

    assert service.commit(candidate).runtime is inline
