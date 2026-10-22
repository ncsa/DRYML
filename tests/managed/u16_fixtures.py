"""Real authored lifecycle fixtures and the result-owner handoff used by U17."""

from dryml.core import Repo
from dryml.core.object import Pickleable
from dryml.core.store.dir import DirStore
from dryml.managed import ManagedConfig, managed_operation
from dryml.managed import runtime
from dryml.managed.control import ManagedControlStore
from dryml.managed.identity import AdmissionIdentity, _argument_digest_bound


class Value(Pickleable):
    captures = restores = bodies = 0
    action = None

    def __init__(self, value=0):
        self.value = value

    def save_state_to_dir_imp(self, directory, *, codec):
        type(self).captures += 1
        return super().save_state_to_dir_imp(directory, codec=codec)

    def restore_state_from_dir_imp(self, directory, *, codec):
        type(self).restores += 1
        return super().restore_state_from_dir_imp(directory, codec=codec)

    @managed_operation(resumable=True)
    def run(self, amount=1, *, managed):
        type(self).bodies += 1
        return type(self).action(self, managed)


def setup(tmp_path, *, index="none", cls=Value):
    store = DirStore(tmp_path / "state", query_index=index)
    repo = Repo(store)
    live = cls(repo=repo)
    cls.captures = cls.restores = cls.bodies = 0
    return ManagedControlStore(store, repo), live


def enter(control, live, *, config=None, finish=None, descriptor=None):
    descriptor = descriptor or type(live).run
    config = (config or ManagedConfig()).snapshot()
    boundary = descriptor.signature_plan().prepare_bound(
        descriptor.bind_arguments(live, (), {}), repo=control.state_repo)
    identity = AdmissionIdentity(control.state_repo.describe(live), descriptor.member,
        _argument_digest_bound(tuple(boundary.authority.items())))
    return runtime.admit_execution(identity, descriptor, config, control,
        realize=lambda _: live,
        execute=lambda owner, current, obj, resuming: runtime._execute_authored(
            owner, current, obj, resuming, descriptor, boundary, config,
            control=control, finish=finish or (lambda context, raw: (raw, context._finalize_state()))))
