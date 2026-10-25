"""Spawn/fork operation ownership and metadata recovery failure boundaries."""

from dataclasses import replace
import os

import pytest

from dryml.core.store.authority import StoreMutationError
from dryml.managed.errors import ManagedConflictError
from tests.managed.test_u14_authority import adapter, facts, receiver, running


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork unavailable")
def test_inherited_operation_owner_cannot_release_or_bypass_parent(tmp_path):
    control, identity = adapter(tmp_path), facts(receiver())
    with control.admission(identity) as owner:
        read_fd, write_fd = os.pipe()
        child = os.fork()
        if child == 0:
            os.close(read_fd)
            try:
                assert not owner.active and not owner.release()
                try: control.admission(identity)
                except ManagedConflictError: pass
                else: os._exit(2)
                os.write(write_fd, b"fenced")
                os._exit(0)
            except BaseException: os._exit(3)
        os.close(write_fd)
        try:
            assert os.read(read_fd, 16) == b"fenced"
            assert os.waitstatus_to_exitcode(os.waitpid(child, 0)[1]) == 0
            assert owner.active
        finally: os.close(read_fd)


@pytest.mark.parametrize("boundary", ["before_current", "after_current", "projection"])
def test_failed_cas_retains_journal_and_reconciliation_cannot_execute(tmp_path, monkeypatch, boundary):
    control, identity = adapter(tmp_path, index="sqlite"), facts(receiver())
    with control.admission(identity) as owner:
        initial = control.create_initial(running(identity, owner), owner=owner)
        original, finish = control._replace_file, control._finish_projection
        fired = []
        def fail(path, payload):
            if path.endswith("current.json") and not fired:
                fired.append(True)
                if boundary == "after_current": original(path, payload)
                raise OSError("controlled current failure")
            return original(path, payload)
        def fail_projection(*args):
            fired.append(True)
            raise OSError("controlled projection failure")
        with monkeypatch.context() as patch:
            if boundary == "projection": patch.setattr(control, "_finish_projection", fail_projection)
            else: patch.setattr(control, "_replace_file", fail)
            with pytest.raises((StoreMutationError, OSError)):
                control.transition_running_owner(identity.operation_id, owner=owner,
                    run_id=owner.run_id, invocation_id=owner.invocation_id, expected_generation=1,
                    build=lambda current: replace(current, generation=2))
        assert fired and control.control_store.authority_cut().pending
    with control.admission(identity) as recovery:
        assert recovery.reconciled
        assert recovery.current.generation == (1 if boundary == "before_current" else 2)
        with pytest.raises(ManagedConflictError, match="reconciliation_only"):
            recovery.realize(lambda node: pytest.fail("recovery fell through into execution"))
    assert not control.control_store.authority_cut().pending


def test_first_publication_failure_never_exposes_currentless_operation(tmp_path, monkeypatch):
    control, identity = adapter(tmp_path), facts(receiver())
    original = control._replace_file
    def fail_first_current(path, payload):
        if path.endswith("current.json"):
            raise OSError("controlled first-current failure")
        return original(path, payload)
    with control.admission(identity) as owner:
        with monkeypatch.context() as patch:
            patch.setattr(control, "_replace_file", fail_first_current)
            with pytest.raises(StoreMutationError):
                control.create_initial(running(identity, owner), owner=owner)
    with control.admission(identity) as recovery:
        assert recovery.current is None and recovery.reconciled
        with pytest.raises(ManagedConflictError, match="reconciliation_only"):
            recovery.realize(lambda node: pytest.fail("failed initial publication executed"))
    with control.admission(identity) as fresh:
        assert not fresh.reconciled
        control.create_initial(running(identity, fresh), owner=fresh)
