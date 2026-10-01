import pytest

from tests.core import core_objects as objects

from dryml.core import Definition, QueryDomainError, Repo, SKIP_ARGS

pytestmark = pytest.mark.usefixtures("fixed_snapshot_environment")


def test_mixed_categorical_and_exact_query_keeps_only_exact_branch():
    repo = Repo()
    encoder_a = objects.TestClass4(1, discriminator="encoder-a", repo=repo)
    encoder_b = objects.TestClass4(1, discriminator="encoder-b", repo=repo)
    parent_a = objects.TestNest3(model=encoder_a, tag="same", repo=repo)
    parent_b = objects.TestNest3(model=encoder_b, tag="same", repo=repo)
    repo.add_objects(parent_a, parent_b)

    results = (
        repo.query(parent_a.definition)
        .categorical(recursive=True)
        .exact(path="kwargs.model")
        .known()
        .defs()
    )

    assert list(results) == [parent_a.definition]


def test_restore_reinstates_original_concrete_anchor():
    repo = Repo()
    opt_a = objects.TestClass4(1, discriminator="optimizer-a", repo=repo)
    opt_b = objects.TestClass4(1, discriminator="optimizer-b", repo=repo)
    parent_a = objects.TestNest3(model=objects.TestClass4(2, repo=repo), optimizer=opt_a, repo=repo)
    parent_b = objects.TestNest3(model=objects.TestClass4(3, repo=repo), optimizer=opt_b, repo=repo)
    repo.add_objects(parent_a, parent_b)

    results = (
        repo.query(parent_a.definition)
        .categorical(recursive=True)
        .restore(path="kwargs.optimizer")
        .known()
        .defs()
    )

    assert list(results) == [parent_a.definition]


def test_find_defs_scope_nested_returns_distinct_nested_definitions(tmp_path):
    from dryml.core.store.dir import DirStore

    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    leaf = objects.TestNest2("leaf", repo=repo)
    parent = objects.TestNest3(child=leaf, repo=repo)
    repo.save_object(parent)

    repo2 = Repo(stores=DirStore(store.base_dir))
    selector = Definition(objects.TestNest2, SKIP_ARGS)

    assert len(repo2.find_defs(selector, scope="stored")) == 0
    nested_defs = repo2.find_defs(selector, scope="nested")
    assert list(nested_defs) == [leaf.definition]


@pytest.mark.parametrize(
    ("edges", "contains_ref"),
    [
        ("materialize", False),
        ("materialize", True),
        ("ref", False),
        ("ref", True),
        ("all", False),
        ("all", True),
    ],
)
def test_nested_retains_immutable_containment_policy(edges, contains_ref):
    repo = Repo()
    target = objects.TestClass4(1, repo=repo).definition

    query = repo.query(target)
    nested = query.nested(edges=edges, contains_ref=contains_ref, refresh=False)

    assert query.domain is None
    assert query.containment_target == target
    assert query.containment_edges == "materialize"
    assert query.contains_ref is False
    assert nested.domain == "nested"
    assert nested.containment_target == target
    assert nested.containment_edges == edges
    assert nested.contains_ref is contains_ref
    assert nested.refresh_policy is False


@pytest.mark.parametrize("edges", [None, True, "reference", "Materialize"])
def test_nested_rejects_invalid_containment_edge_policy(edges):
    with pytest.raises(ValueError, match="edges"):
        Repo().query().nested(edges=edges)


@pytest.mark.parametrize("contains_ref", [None, 0, 1, "true"])
def test_nested_requires_exact_boolean_reference_filter(contains_ref):
    with pytest.raises(TypeError, match="contains_ref"):
        Repo().query().nested(contains_ref=contains_ref)


def test_nested_default_remains_materialize_only(tmp_path):
    from dryml.core.store.dir import DirStore

    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    child = objects.TestNest2("child", repo=repo)
    parent = objects.TestNest3(child=child, repo=repo)
    repo.save_object(parent)

    query = repo.query(child.definition).nested(refresh=False)

    assert query.containment_edges == "materialize"
    assert query.contains_ref is False
    assert list(query.definitions().defs()) == [child.definition]


def test_nested_cdef_containment_rejects_reference_authority_conversion():
    repo = Repo()
    target = objects.TestClass4(1, repo=repo).definition

    with pytest.raises(QueryDomainError, match="containment"):
        repo.query(target).nested().references()
