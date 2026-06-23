import core2_objects as objects

from dryml.core2 import Definition, Repo, SKIP_ARGS


def test_mixed_categorical_and_exact_query_keeps_only_exact_branch():
    repo = Repo()
    encoder_a = objects.TestClass4(1, repo=repo)
    encoder_b = objects.TestClass4(1, repo=repo)
    parent_a = objects.TestNest3(model=encoder_a, tag="same", repo=repo)
    parent_b = objects.TestNest3(model=encoder_b, tag="same", repo=repo)
    repo.add_objects(parent_a, parent_b)

    results = (
        repo.query(parent_a.definition)
        .categorical(recursive=True)
        .exact(path="model")
        .known()
        .defs()
    )

    assert list(results) == [parent_a.definition]


def test_restore_reinstates_original_concrete_anchor():
    repo = Repo()
    opt_a = objects.TestClass4(1, repo=repo)
    opt_b = objects.TestClass4(1, repo=repo)
    parent_a = objects.TestNest3(model=objects.TestClass4(2, repo=repo), optimizer=opt_a, repo=repo)
    parent_b = objects.TestNest3(model=objects.TestClass4(3, repo=repo), optimizer=opt_b, repo=repo)
    repo.add_objects(parent_a, parent_b)

    results = (
        repo.query(parent_a.definition)
        .categorical(recursive=True)
        .restore(path="optimizer")
        .known()
        .defs()
    )

    assert list(results) == [parent_a.definition]


def test_find_defs_scope_nested_returns_distinct_nested_definitions(tmp_path):
    from dryml.core2.store.dir import DirStore

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
