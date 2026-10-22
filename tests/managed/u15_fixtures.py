"""Current-format admission fixtures; callbacks stop at the U15 execution seam."""

from dataclasses import replace
from uuid import uuid4

from dryml.core import Repo
from dryml.core.object import Pickleable
from dryml.core.store.dir import DirStore
from dryml.managed import ManagedConfig, managed_operation
from dryml.managed.control import ControlIntent, ControlSnapshot, ManagedControlStore
from dryml.managed.identity import AdmissionIdentity
from dryml.managed import runtime
from dryml.managed.results import publish_result
from dryml.managed.result_codec import encode_result


class Value(Pickleable):
    restores = 0
    captures = 0
    bodies = 0
    fail_restore = False

    def __init__(self, value=1):
        self.value = value

    def restore_state_from_dir_imp(self, directory, *, codec):
        type(self).restores += 1
        if type(self).fail_restore:
            raise ValueError("hook failure")
        return super().restore_state_from_dir_imp(directory, codec=codec)

    def save_state_to_dir_imp(self, directory, *, codec):
        type(self).captures += 1
        return super().save_state_to_dir_imp(directory, codec=codec)

    @managed_operation(resumable=True)
    def run(self, amount=2, *, managed):
        type(self).bodies += 1
        raise AssertionError("U15 must not execute an authored body")

    @managed_operation()
    def fixed(self, amount=2, *, managed):
        type(self).bodies += 1
        raise AssertionError("U15 must not execute an authored body")


def setup(tmp_path, *, index="none"):
    store = DirStore(tmp_path / "state", query_index=index)
    repo = Repo(store)
    live = Value(repo=repo)
    saved = repo.save(live)
    control = ManagedControlStore(store, repo)
    Value.restores = Value.captures = Value.bodies = 0
    return control, live, saved


def identity(control, live, *, descriptor=Value.run, amount=2):
    return AdmissionIdentity.from_call(control.state_repo.describe(live), descriptor, (), {"amount": amount})


def seed(control, live, saved, *, state="failed", checkpoint=True, initial=None, descriptor=Value.run):
    control.initialize()
    reference = publish_result(control.control_store, encode_result(None)) if state == "completed" else None
    facts = identity(control, live, descriptor=descriptor)
    with control.admission(facts) as owner:
        intents = () if initial is None else (ControlIntent(
            uuid4().hex, owner.run_id, owner.invocation_id, 1, "adopted",
            kind="initial", selection=initial),)
        snapshot = ControlSnapshot(facts.receiver, facts.member, facts.argument_digest,
            owner.run_id, owner.invocation_id, 1, state,
            checkpoint=saved if checkpoint else None,
            final=saved if state == "completed" else None,
            producer_invocation_id=owner.invocation_id if state == "completed" else None,
            failure_code="method_error" if state == "failed" else None, intents=intents, result_reference=reference)
        return control.create_initial(snapshot, owner=owner)


def enter(control, live, *, config=None, descriptor=Value.run, amount=2, execute=None, realize=None):
    return runtime.admit_execution(identity(control, live, descriptor=descriptor, amount=amount),
        descriptor, (config or ManagedConfig()).snapshot(), control,
        realize=realize or (lambda node: live),
        execute=execute or (lambda owner, snapshot, obj, resuming: (snapshot, obj.value, resuming)))
