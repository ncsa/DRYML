import pytest

from dryml.runtime import PublicationService, RuntimeState
from dryml.runtime.errors import ForkSafetyError


def test_pristine_child_reinitializes_but_observed_child_rejects_before_lock():
    service = PublicationService(environ={}, pid_getter=lambda: 10)
    service.initialize(RuntimeState())
    service._pid_getter = lambda: 11
    assert service.current().number == 0
    service.current()
    service._pid_getter = lambda: 12
    with pytest.raises(ForkSafetyError):
        service.current()
