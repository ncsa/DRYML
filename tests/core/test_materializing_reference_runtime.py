from dryml.core import Definition, Object, Repo, Serializable
from dryml.core.links import Ref


class ReferenceLeaf(Object):
    def __init__(self, value):
        super().__init__()
        self.value = value


class ReferenceParent(Object):
    def __init__(self, target):
        super().__init__()
        self.target = target


class IdentifiedReferenceLeaf(Serializable):
    def __init__(self, value):
        super().__init__()
        self.value = value


def test_ref_exact_value_is_retained_without_materialization():
    repo = Repo()
    target = Definition(ReferenceLeaf, "exact").concretize(repo=repo)
    parent = ReferenceParent(Ref(target), repo=repo)

    assert parent.graph_at('$[@param("target")]') is target


def test_materializing_object_ref_rebinds_the_supplied_object_id():
    repo = Repo()
    original = IdentifiedReferenceLeaf("exact", repo=repo)
    parent = ReferenceParent(original.object_ref, repo=repo)

    assert isinstance(parent.target, IdentifiedReferenceLeaf)
    assert parent.target is not original
    assert parent.target.object_id == original.object_id
