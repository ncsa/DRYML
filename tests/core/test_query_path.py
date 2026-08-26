import core2_objects as objects
import pytest

from dryml.core2 import Definition, Repo
from dryml.core2.bound_args import BoundArguments
from dryml.core2.cdef_identity import V2_IDENTITY_VERSION
from dryml.core2.definition import ConcreteDefinition
from dryml.core2.freeze import FrozenDict, FrozenList, FrozenTuple
from dryml.core2.query import Arg, DefinitionPath, Index, Key, Kwarg, Parameter, QueryPathError, normalize_path


def test_query_path_parses_root_and_segments():
    assert normalize_path("$") == DefinitionPath()
    assert normalize_path("model.encoder") == DefinitionPath((Kwarg("model"), Kwarg("encoder")))
    assert normalize_path("args[0]") == DefinitionPath((Arg(0),))
    assert normalize_path("model.layer_defs[1]") == DefinitionPath((Kwarg("model"), Kwarg("layer_defs"), Index(1)))
    assert normalize_path('metadata["x.y"]') == DefinitionPath((Kwarg("metadata"), Key("x.y")))


def test_query_path_resolves_concrete_definition_subtrees():
    leaf = objects.TestClass1(10, test="leaf")
    root = objects.TestNest3(leaf, metadata={"x.y": [leaf]})

    from dryml.core2.query.path import get_subtree

    assert get_subtree(root.definition, "args[0]") == leaf.definition
    assert get_subtree(root.definition, 'metadata["x.y"][0]') == leaf.definition


def test_v2_semantic_paths_resolve_without_class_projection():
    leaf = ConcreteDefinition._from_persisted_record(
        objects.TestClass1,
        identity_version=V2_IDENTITY_VERSION,
        parameters=BoundArguments((("value", 10), ("test", "leaf"))),
    )
    root = ConcreteDefinition._from_persisted_record(
        objects.TestNest3,
        identity_version=V2_IDENTITY_VERSION,
        parameters=BoundArguments((("model", leaf),)),
    )
    path = DefinitionPath((Parameter("model"),))

    from dryml.core2.query.path import get_subtree

    assert root.graph_path(path) == leaf
    assert root.graph_path('$[@param("model")]') == leaf
    assert get_subtree(root, path) == leaf
    assert root.parameters["model"] == leaf


def test_v1_keywords_and_v2_parameters_resolve_in_their_own_path_domains():
    leaf = ConcreteDefinition(objects.TestClass1, (10,), {"test": "leaf"})
    legacy = ConcreteDefinition(objects.TestNest3, (), {"model": leaf})
    semantic = ConcreteDefinition._from_persisted_record(
        objects.TestNest3,
        identity_version=V2_IDENTITY_VERSION,
        parameters=BoundArguments((("model", leaf),)),
    )

    assert legacy.graph_path(DefinitionPath((Kwarg("model"),))) == leaf
    assert semantic.graph_path(DefinitionPath((Parameter("model"),))) == leaf
    with pytest.raises(QueryPathError):
        semantic.graph_path(DefinitionPath((Kwarg("model"),)))


def test_v2_variadic_buckets_use_parameter_then_container_paths():
    leaf = ConcreteDefinition._from_persisted_record(
        objects.TestClass1,
        identity_version=V2_IDENTITY_VERSION,
        parameters=BoundArguments((("value", 10), ("test", "leaf"))),
    )
    root = ConcreteDefinition._from_persisted_record(
        objects.TestNest3,
        identity_version=V2_IDENTITY_VERSION,
        parameters=BoundArguments((
            ("sources", FrozenTuple(("first", leaf))),
            ("capabilities", FrozenDict({"encoder": leaf})),
        )),
    )

    assert root.graph_path('$[@param("sources")][1]') == leaf
    assert root.graph_path('$[@param("capabilities")]["encoder"]') == leaf

    from dryml.core2.query.fingerprint import target_local_fingerprint

    assert any(
        token.kind == "CDEF_EDGE_AT_PATH" and str(token.path) == '$[@param("sources")][1]'
        for token in target_local_fingerprint(root).counts
    )


def test_invalid_paths_report_errors():
    obj = objects.TestNest3(child=1)
    from dryml.core2.query.path import get_subtree

    with pytest.raises(QueryPathError):
        normalize_path("$bad")
    with pytest.raises(QueryPathError, match="missing"):
        get_subtree(obj.definition, "missing")
    with pytest.raises(QueryPathError):
        get_subtree(obj.definition, "child.value")


def test_v2_invalid_semantic_path_reports_the_failing_prefix():
    cdef = ConcreteDefinition._from_persisted_record(
        objects.TestClass1,
        identity_version=V2_IDENTITY_VERSION,
        parameters=BoundArguments((("value", 10),)),
    )

    with pytest.raises(QueryPathError, match=r'at \$\[@param\("missing"\)\]'):
        cdef.graph_path('$[@param("missing")]')


def test_replace_subtree_preserves_container_types():
    from dryml.core2.query.path import get_subtree, replace_subtree
    obj = objects.TestNest3(items=("a", "b"), mapping={"k": [1, 2]})
    replaced = replace_subtree(obj.definition, "items[1]", "c")
    replaced = replace_subtree(replaced, "mapping.k[0]", 9)

    assert isinstance(get_subtree(replaced, "items"), tuple)
    assert get_subtree(replaced, "items[1]") == "c"
    assert get_subtree(replaced, "mapping.k[0]") == 9


def test_query_projection_does_not_mutate_source_or_reinject_uid():
    repo = Repo()
    child = objects.TestClass4(1, repo=repo)
    parent = objects.TestNest3(child=child, repo=repo)
    original = parent.definition

    projected = repo.query(original).categorical(path="child", recursive=True)

    assert parent.definition == original
    assert "uid" in original.kwargs["child"].kwargs
    assert "uid" not in projected.selector.kwargs["child"].kwargs


def test_chained_query_methods_return_independent_queries():
    repo = Repo()
    source = objects.TestNest3(child=objects.TestClass4(1, repo=repo), repo=repo).definition
    q1 = repo.query(source)
    q2 = q1.categorical(path="child", recursive=True)
    q3 = q2.restore(path="child")

    assert q1 is not q2
    assert q2 is not q3
    assert "uid" in q1.selector.kwargs["child"].kwargs
    assert "uid" not in q2.selector.kwargs["child"].kwargs
    assert q3.selector.kwargs["child"] == source.kwargs["child"]


def test_restore_frozen_list_branch_preserves_query_soundness():
    repo = Repo()
    source = objects.TestNest3(items=[1, 2], repo=repo).definition
    match = objects.TestNest3(items=[1, 2], repo=repo)
    other = objects.TestNest3(items=[1, 3], repo=repo)
    repo.add_objects(match, other)

    query = repo.query(source).categorical(recursive=True).restore(path="items")

    assert isinstance(query.selector.kwargs["items"], FrozenList)
    assert list(query.known(refresh=False).defs()) == [match.definition]
