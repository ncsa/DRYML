from dryml.core import Object, Repo, Serializable
from dryml.core.repo_plan import build_runtime_binding


class BindingLeaf(Object):
    def __init__(self, value):
        super().__init__()
        self.value = value


class BindingParent(Serializable):
    def __init__(self, child, values):
        super().__init__()
        values["changed"] = True
        self.child = child


def test_runtime_binding_retains_supplied_object_and_defensive_values():
    repo = Repo()
    child = BindingLeaf("child", repo=repo)
    values = {"source": [1]}

    parent = BindingParent(child, values, repo=repo)
    values["source"].append(2)

    assert parent.graph_at(parent.definition.graph_path("$") and "$") is parent
    assert parent.graph_at('$[@param("child")]') is child
    assert parent.graph_at('$[@param("values")]') == {"source": [1]}
    assert parent.graph_at('$[@param("values")]') is not parent.graph_at('$[@param("values")]')
    assert parent.object_id is not None
    assert child.object_id is None
    binding = build_runtime_binding(repo, parent)
    assert binding.objects[parent.definition] is parent
