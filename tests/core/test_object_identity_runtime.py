from dryml.core import Object, Repo, Serializable


class Ephemeral(Object):
    def __init__(self):
        super().__init__()


class Stateful(Serializable):
    def __init__(self, child=None):
        super().__init__()
        self.child = child


def test_only_serializable_nodes_receive_runtime_object_ids():
    repo = Repo()
    child = Ephemeral(repo=repo)
    parent = Stateful(child, repo=repo)

    assert parent.object_id is not None
    assert child.object_id is None
    assert parent.object_ref.object_id == parent.object_id
    assert parent._last_state_hash is None
