from dryml.core import Definition, Object, Repo


class ScopeLeaf(Object):
    constructed = 0

    def __init__(self, value):
        super().__init__()
        type(self).constructed += 1
        self.value = value


class ScopeParent(Object):
    def __init__(self, left, right):
        super().__init__()
        self.left = left
        self.right = right


def test_aliases_share_but_independent_equal_nodes_do_not():
    repo = Repo()
    shared = Definition(ScopeLeaf, "same")
    aliased = repo.load_or_build(Definition(ScopeParent, shared, shared))
    first = Definition(ScopeLeaf, "same")
    second = Definition(ScopeLeaf, "same")
    distinct = repo.load_or_build(Definition(ScopeParent, first, second), cache="none")

    assert aliased.left is aliased.right
    assert distinct.left is not distinct.right
    assert distinct.left.definition == distinct.right.definition
    assert distinct.left._realization_scope == distinct.right._realization_scope
    assert aliased._realization_scope != distinct._realization_scope
