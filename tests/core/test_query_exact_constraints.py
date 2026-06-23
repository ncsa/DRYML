import pytest

import core2_objects as objects
from dryml.core2 import Definition, Repo, SKIP_ARGS
from dryml.core2.definition import ConcreteDefinition
from dryml.core2.query.fingerprint import collect_exact_constraints
from dryml.core2.query.query import _exact_constraints_match


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
    constraints = collect_exact_constraints(selector)

    monkeypatch.setattr(ConcreteDefinition, "stable_hash", lambda self: "collision")

    assert _exact_constraints_match(parent_a.definition, constraints)
    assert not _exact_constraints_match(parent_b.definition, constraints)


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
