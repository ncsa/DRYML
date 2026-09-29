import pytest

from dryml.runtime import EffectPlan, LocalResourceInventory, PublicationService, RuntimeAllocationView, RuntimeContextSpec, RuntimeMode, RuntimeState
from dryml.runtime.errors import PublicationError


def _inline():
    return RuntimeState(RuntimeMode.INLINE, RuntimeAllocationView(role="main", replica=0, cpus=(0,)))


def test_equivalent_publication_preserves_generation_without_effects():
    service = PublicationService(environ={})
    service.initialize(RuntimeState())
    generation = service.publish(_inline(), inventory=LocalResourceInventory((0,)))
    assert service.publish(_inline(), inventory=LocalResourceInventory((0,)), effects=EffectPlan()) is generation
    assert generation.number == 1


def test_stale_inventory_retries_and_rejects_invalid_retry_bounds():
    service = PublicationService(environ={})
    service.initialize(RuntimeState())
    values = iter((LocalResourceInventory((0,)), LocalResourceInventory((1,)), LocalResourceInventory((1,)), LocalResourceInventory((1,))))
    assert service.publish(_inline(), inventory_observer=lambda: next(values)).number == 1
    with pytest.raises(PublicationError):
        service.publish(_inline(), inventory=LocalResourceInventory((1,)), restage_retries=17)


def test_retry_boundaries_cover_zero_and_sixteen_restages():
    service = PublicationService(environ={})
    service.initialize(RuntimeState())
    stale = iter((LocalResourceInventory((0,)), LocalResourceInventory((1,))))
    with pytest.raises(PublicationError):
        service.publish(_inline(), inventory_observer=lambda: next(stale), restage_retries=0)

    observations = []
    for index in range(16):
        observations.extend((LocalResourceInventory((index + 2,)), LocalResourceInventory((index + 3,))))
    observations.extend((LocalResourceInventory((18,)), LocalResourceInventory((18,))))
    values = iter(observations)
    assert service.publish(_inline(), inventory_observer=lambda: next(values), restage_retries=16).number == 1


def test_inventory_comparator_detects_accelerator_and_per_device_memory_changes():
    service = PublicationService(environ={})
    service.initialize(RuntimeState())
    values = iter((
        LocalResourceInventory((0,), {"gpu": ("a",)}, accelerator_memory={"gpu": {"a": 1}}),
        LocalResourceInventory((0,), {"gpu": ("a",)}, accelerator_memory={"gpu": {"a": 2}}),
        LocalResourceInventory((0,), {"gpu": ("b",)}, accelerator_memory={"gpu": {"b": 2}}),
        LocalResourceInventory((0,), {"gpu": ("b",)}, accelerator_memory={"gpu": {"b": 2}}),
    ))
    assert service.publish(_inline(), inventory_observer=lambda: next(values)).inventory.accelerators["gpu"] == ("b",)


def test_none_rejects_inventory_effects_and_managed_control_plans():
    service = PublicationService(environ={})
    service.initialize(RuntimeState())
    with pytest.raises(PublicationError):
        service.publish(RuntimeState(), inventory=LocalResourceInventory((0,)))
    with pytest.raises(PublicationError):
        service.publish(RuntimeState(spec=RuntimeContextSpec(visibility={"policy": "assigned"})))


def test_same_epoch_status_finalization_can_publish_multiple_generations():
    service = PublicationService(environ={})
    service.initialize(RuntimeState())
    admission = service.admit_status_finalization()

    first = service.finalize_statuses(admission, {"visibility": "visibility-enforced"})
    second = service.finalize_statuses(admission, {"threading": "framework-configured"})

    assert first.number == 1
    assert second.number == 2
    assert second.statuses == {"visibility": "visibility-enforced", "threading": "framework-configured"}
