from pathlib import Path

from dryml.core import Object, Repo, Serializable
from dryml.core.store.dir import DirStore


class FederationState(Serializable):
    def __init__(self, value):
        self.value = value

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        Path(dest_dir, "value.txt").write_text(str(self.value))


class FederationRoot(Object):
    def __init__(self, child):
        self.child = child


def test_default_save_copies_reused_dependencies_but_federated_retains_them(tmp_path):
    source = DirStore(tmp_path / "source")
    copied = DirStore(tmp_path / "copied")
    federated = DirStore(tmp_path / "federated")
    repo = Repo([source, copied, federated])
    child = FederationState("child", repo=repo)
    child.save(repo=repo, store=source)
    root = FederationRoot(child, repo=repo)

    copied_state, copied_report = root.save(repo=repo, store=copied, report_stores=True)
    child_path = next(path for path, obj_id in copied_state.object.objects.items() if obj_id == child.object_id)
    assert copied_report.required_stores == (copied,)
    copied.validate_local_state(child.definition, copied_state.states[child_path])

    federated_state, report = root.save(repo=repo, store=federated, federated=True, report_stores=True)
    child_path = next(path for path, obj_id in federated_state.object.objects.items() if obj_id == child.object_id)
    assert report.state_stores[child_path] is source
    assert report.required_stores == (federated, source)
