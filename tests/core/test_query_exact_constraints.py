import pytest

import core_objects as objects
from dryml.core import Definition, Repo, SKIP_ARGS
from dryml.core.definition import ConcreteDefinition
from dryml.core.freeze import FrozenDict, FrozenSet
from dryml.core.query.query import _query_match


class SemanticQueryLeaf(objects.Object):
    def __init__(self, value=3, *, label="default"):
        self.value = value
        self.label = label


class SemanticQueryParent(objects.Object):
    def __init__(self, child=None):
        self.child = child


class SemanticVariadicLeaf(objects.Object):
    def __init__(self, *items, **options):
        self.items = items
        self.options = options


class SemanticQueryStore:
    def catalog_key(self):
        return "semantic-query-store"


def test_memory_query_uses_v2_parameters_for_partial_selector_constraints():
    repo = Repo()
    match = Definition(SemanticQueryLeaf, 7, label="match").concretize()
    default = Definition(SemanticQueryLeaf).concretize()
    store = SemanticQueryStore()
    repo._query_catalog.register_stored(match, store)
    repo._query_catalog.register_stored(default, store)

    positional = tuple(repo.query(Definition(SemanticQueryLeaf, 7)).stored(refresh=False).defs())
    keyword = tuple(repo.query(Definition(SemanticQueryLeaf, value=7)).stored(refresh=False).defs())
    explicit_default = tuple(repo.query(Definition(SemanticQueryLeaf, value=3)).stored(refresh=False).defs())
    omitted = tuple(repo.query(Definition(SemanticQueryLeaf)).stored(refresh=False).defs())

    assert positional == keyword == (match,)
    assert explicit_default == (default,)
    assert set(omitted) == {match, default}


def test_nested_v2_selector_falls_back_to_authoritative_verification():
    repo = Repo()
    match_child = Definition(SemanticQueryLeaf, 7).concretize()
    other_child = Definition(SemanticQueryLeaf, 8).concretize()
    match = Definition(SemanticQueryParent, match_child).concretize()
    other = Definition(SemanticQueryParent, other_child).concretize()
    store = SemanticQueryStore()
    repo._query_catalog.register_stored(match, store)
    repo._query_catalog.register_stored(other, store)

    query = repo.query(
        Definition(SemanticQueryParent, child=Definition(SemanticQueryLeaf, value=7))
    ).stored(refresh=False)

    assert query.selector is not None
    assert tuple(query.defs()) == (match,)


def test_v2_variadic_selectors_are_not_pruned_by_legacy_feature_paths():
    repo = Repo()
    target = Definition(SemanticVariadicLeaf, "first", "second", enabled=True).concretize()
    store = SemanticQueryStore()
    repo._query_catalog.register_stored(target, store)

    positional = repo.query(
        Definition(SemanticVariadicLeaf, "first", "second")
    ).stored(refresh=False)
    keyword = repo.query(
        Definition(SemanticVariadicLeaf, enabled=True)
    ).stored(refresh=False)

    assert tuple(positional.defs()) == (target,)
    assert tuple(keyword.defs()) == (target,)


def test_exact_root_rejects_structurally_compatible_extra_kwargs():
    repo = Repo()
    base = objects.TestNest3(a=1, repo=repo)
    extra = objects.TestNest3(a=1, b=2, repo=repo)
    repo.add_objects(base, extra)

    results = repo.query(base.definition).known(refresh=False).defs()

    assert list(results) == [base.definition]


def test_exact_root_distinguishes_unique_ids():
    repo = Repo()
    first = objects.TestClass4(1, repo=repo)
    second = objects.TestClass4(1, repo=repo)
    repo.add_objects(first, second)

    assert list(repo.query(first.definition).known(refresh=False).defs()) == [first.definition]


def test_nested_exact_accepts_identical_and_rejects_compatible_subtree():
    repo = Repo()
    child_a = objects.TestClass4(1, repo=repo)
    child_b = objects.TestClass4(1, repo=repo)
    parent_a = objects.TestNest3(child=child_a, label="same", repo=repo)
    parent_b = objects.TestNest3(child=child_b, label="same", repo=repo)
    repo.add_objects(parent_a, parent_b)

    results = (
        repo.query(parent_a.definition)
        .categorical(recursive=True)
        .exact(path="child")
        .known(refresh=False)
        .defs()
    )

    assert list(results) == [parent_a.definition]


def test_multiple_exact_constraints_are_conjunctive():
    repo = Repo()
    left = objects.TestClass4(1, repo=repo)
    right = objects.TestClass4(2, repo=repo)
    other_right = objects.TestClass4(2, repo=repo)
    match = objects.TestNest3(left=left, right=right, repo=repo)
    mismatch = objects.TestNest3(left=left, right=other_right, repo=repo)
    repo.add_objects(match, mismatch)

    results = (
        repo.query(match.definition)
        .categorical(recursive=True)
        .exact(path="left")
        .exact(path="right")
        .known(refresh=False)
        .defs()
    )

    assert list(results) == [match.definition]


def test_exact_without_concrete_source_subtree_raises():
    repo = Repo()
    source = Definition(objects.TestNest3, child=Definition(objects.TestNest2, "x"))

    with pytest.raises(TypeError, match="ConcreteDefinition"):
        repo.query(source).exact(path="child")


def test_exact_with_explicit_object_uses_object_definition():
    repo = Repo()
    child = objects.TestClass4(1, repo=repo)
    match = objects.TestNest3(child=child, repo=repo)
    repo.add_objects(match)

    results = (
        repo.query(Definition(objects.TestNest3, SKIP_ARGS, child=Definition(objects.TestClass4, SKIP_ARGS)))
        .exact(child, path="child")
        .known(refresh=False)
        .defs()
    )

    assert list(results) == [match.definition]


def test_exact_constraint_confirms_equality_after_hash_match(monkeypatch):
    repo = Repo()
    child_a = objects.TestClass4(1, repo=repo)
    child_b = objects.TestClass4(1, repo=repo)
    parent_a = objects.TestNest3(child=child_a, repo=repo)
    parent_b = objects.TestNest3(child=child_b, repo=repo)

    selector = repo.query(parent_a.definition).categorical(recursive=True).exact(path="child").selector

    monkeypatch.setattr(ConcreteDefinition, "stable_hash", lambda self: "collision")

    assert _query_match(selector, parent_a.definition, strict=False, class_match="selector")
    assert not _query_match(selector, parent_b.definition, strict=False, class_match="selector")


def test_concrete_definition_inside_set_is_exact():
    repo = Repo()
    exact_child = objects.TestNest3(a=1, repo=repo)
    compatible_child = objects.TestNest3(a=1, b=2, repo=repo)
    exact_parent = objects.TestNest3(members={exact_child}, repo=repo)
    compatible_parent = objects.TestNest3(members={compatible_child}, repo=repo)
    repo.add_objects(exact_parent, compatible_parent)

    selector = Definition(objects.TestNest3, SKIP_ARGS, members={exact_child.definition})
    results = repo.query(selector).known(refresh=False).defs()

    assert list(results) == [exact_parent.definition]


def test_class_match_exact_applies_inside_set_members():
    repo = Repo()
    base_parent = objects.TestNest({objects.TestBase}, repo=repo)
    subclass_parent = objects.TestNest({objects.TestClassA}, repo=repo)
    repo.add_objects(base_parent, subclass_parent)

    selector = Definition(objects.TestNest, {objects.TestBase})
    results = repo.query(selector).class_match("exact").known(refresh=False).defs()

    assert list(results) == [base_parent.definition]


def test_exact_cdef_inside_tuple_inside_set_remains_exact():
    repo = Repo()
    exact_child = objects.TestNest3(a=1, repo=repo)
    compatible_child = objects.TestNest3(a=1, b=2, repo=repo)
    exact_parent = objects.TestNest3(members={(exact_child.definition,)}, repo=repo)
    compatible_parent = objects.TestNest3(members={(compatible_child.definition,)}, repo=repo)
    repo.add_objects(exact_parent, compatible_parent)

    selector = Definition(objects.TestNest3, SKIP_ARGS, members={(exact_child.definition,)})
    results = repo.query(selector).known(refresh=False).defs()

    assert list(results) == [exact_parent.definition]


def test_exact_cdef_inside_frozenset_inside_set_remains_exact():
    repo = Repo()
    exact_child = objects.TestNest3(a=1, repo=repo)
    compatible_child = objects.TestNest3(a=1, b=2, repo=repo)
    template = objects.TestNest3().definition
    exact_parent = ConcreteDefinition(
        template.cls,
        template.args,
        FrozenDict({"members": FrozenSet({FrozenSet({exact_child.definition})})}),
    )
    compatible_parent = ConcreteDefinition(
        template.cls,
        template.args,
        FrozenDict({"members": FrozenSet({FrozenSet({compatible_child.definition})})}),
    )

    class FakeStore:
        def __init__(self, name):
            self.name = name

        def catalog_key(self):
            return self.name

    repo._query_catalog.register_stored(exact_parent, FakeStore("exact"))
    repo._query_catalog.register_stored(compatible_parent, FakeStore("compatible"))

    selector = Definition(objects.TestNest3, SKIP_ARGS, members={FrozenSet({exact_child.definition})})
    results = repo.query(selector).stored(refresh=False).defs()

    assert list(results) == [exact_parent]


def test_unordered_matching_backtracks_for_ambiguous_members():
    repo = Repo()
    target = objects.TestNest({objects.TestClassA, objects.TestClassB}, repo=repo)
    repo.add_objects(target)

    selector = Definition(objects.TestNest, {objects.TestBase, objects.TestClassA})

    assert list(repo.query(selector).known(refresh=False).defs()) == [target.definition]


def test_exact_class_unordered_matching_backtracks():
    repo = Repo()
    target = objects.TestNest({objects.TestClassA, objects.TestClassB}, repo=repo)
    repo.add_objects(target)

    selector = Definition(objects.TestNest, {objects.TestClassA, objects.TestClassB})

    assert list(repo.query(selector).class_match("exact").known(refresh=False).defs()) == [target.definition]
