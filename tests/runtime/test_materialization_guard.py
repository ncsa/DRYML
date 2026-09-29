import threading
import warnings

import pytest

from dryml import session
from dryml.runtime import materialization_admission, materialization_scope
from dryml.runtime.context import publication
from dryml.runtime.errors import PublicationBusyError, RuntimeTransitionError


@pytest.fixture(autouse=True)
def reset_runtime():
    session.reset()
    yield
    session.reset()


def test_strict_admission_rejects_with_actionable_operation_diagnostic():
    session.set_mode("orchestrator")

    with pytest.raises(RuntimeTransitionError, match="prohibits Object materialization") as error:
        with materialization_admission(operation="sentinel_operation"):
            raise AssertionError("guard did not reject")

    assert error.value.context["operation"] == "sentinel_operation"
    assert "Definition/CDef" in error.value.context["fix"]


def test_warn_and_off_are_explicit_and_nested_warns_once():
    session.set_mode("orchestrator")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with materialization_scope("warn"):
            with materialization_admission(operation="outer"):
                with materialization_admission(operation="inner"):
                    pass
    assert len(caught) == 1

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with materialization_scope("off"):
            with materialization_admission(operation="silent"):
                pass
    assert caught == []


def test_nested_admission_applies_innermost_action():
    session.set_mode("orchestrator")

    with materialization_scope("off"):
        with materialization_admission(operation="outer"):
            with materialization_scope("strict"):
                with pytest.raises(RuntimeTransitionError):
                    with materialization_admission(operation="inner_strict"):
                        pass

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                with materialization_scope("warn"):
                    with materialization_admission(operation="inner_warn"):
                        with materialization_admission(operation="inner_warn_again"):
                            pass

    assert len(caught) == 1


def test_nested_admission_survives_same_control_epoch_status_finalization():
    session.set_mode("orchestrator")
    status_admission = publication.admit_status_finalization()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with materialization_scope("warn"):
            with materialization_admission(operation="outer"):
                publication.finalize_statuses(status_admission, {"fake": "enforced"})
                with materialization_admission(operation="inner"):
                    pass

    assert len(caught) == 1


def test_warn_admission_lease_blocks_incompatible_session_transition():
    session.set_mode("orchestrator")
    entered = threading.Event()
    release = threading.Event()

    def hold_lease():
        with materialization_scope("warn"):
            with materialization_admission(operation="threaded"):
                entered.set()
                release.wait(timeout=5)

    thread = threading.Thread(target=hold_lease)
    thread.start()
    assert entered.wait(timeout=5)
    try:
        with pytest.raises(PublicationBusyError, match="lease"):
            session.set_mode("python")
    finally:
        release.set()
        thread.join(timeout=5)
    assert not thread.is_alive()
