import pytest

from dryml.runtime import EffectPlan, PublicationService, RuntimeState
from dryml.runtime.errors import PublicationError, PublicationFailedError


def test_reversible_failure_rolls_back_and_preserves_prior_generation():
    environment = {"A": "old"}
    service = PublicationService(environ=environment)
    service.initialize(RuntimeState())
    candidate = service.stage(service.current(), RuntimeState())
    with pytest.raises(RuntimeError):
        service.commit(candidate, EffectPlan(environment={"A": "new"}), validator=lambda: (_ for _ in ()).throw(RuntimeError("stop")))
    assert environment["A"] == "old"
    assert service.current().number == 0


def test_precondition_failure_before_irreversible_effect_preserves_prior_authority():
    service = PublicationService(environ={})
    service.initialize(RuntimeState())
    candidate = service.stage(service.current(), RuntimeState())
    with pytest.raises(RuntimeError):
        service.commit(candidate, EffectPlan(irreversible_outcome="native"), validator=lambda: (_ for _ in ()).throw(RuntimeError("stop")))
    assert service.current().health == "healthy"


def test_stale_irreversible_plan_does_not_poison_authority_before_effects():
    service = PublicationService(environ={})
    service.initialize(RuntimeState())
    stale = service.stage(service.current(), RuntimeState())
    service.commit(service.stage(service.current(), RuntimeState()), EffectPlan(environment={"A": "one"}))

    with pytest.raises(PublicationError, match="stale"):
        service.commit(stale, EffectPlan(irreversible_outcome="native"))

    assert service.current().health == "healthy"


def test_reset_restores_the_value_before_all_owned_effect_updates():
    environment = {"A": "old"}
    service = PublicationService(environ=environment)
    service.initialize(RuntimeState())
    service.commit(service.stage(service.current(), RuntimeState()), EffectPlan(environment={"A": "one"}))
    service.commit(service.stage(service.current(), RuntimeState()), EffectPlan(environment={"A": "two"}))
    service.reset(RuntimeState())
    assert environment["A"] == "old"
