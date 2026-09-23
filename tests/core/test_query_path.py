from tests.core import core_objects as objects
import pytest

from dryml.core import Definition, Repo, SKIP_ARGS
from dryml.core.bound_args import BoundArguments
from dryml.core.cdef_identity import V2_IDENTITY_VERSION
from dryml.core.definition import ConcreteDefinition
from dryml.core.freeze import FrozenDict, FrozenList, FrozenSet, FrozenTuple
from dryml.core.cdef_graph import EdgeKind
from dryml.core.links import DefLink
from dryml.core.symbol import ImportRef
from dryml.core.query import Arg, DefinitionPath, Index, Key, Kwarg, Parameter, QueryPathError, normalize_path


class U4SetLeaf(objects.Object):
    def __init__(self, value, *, seed, another=None):
        self.value = value
        self.seed = seed
        self.another = another


class U4SetParent(objects.Object):
    def __init__(self, members):
        self.members = members


class U4ListParent(objects.Object):
    def __init__(self, values):
        self.values = values


class U4VariadicParent(objects.Object):
    def __init__(self, first, /, width=2, *items, seed=3, **options):
        self.first = first
        self.width = width
        self.items = items
        self.seed = seed
        self.options = options


class U4AliasParent(objects.Object):
    def __init__(self, child, sibling):
        self.child = child
        self.sibling = sibling


class U4NoneLeaf(objects.Object):
    def __init__(self, value):
        self.value = value


def test_query_path_parses_root_and_segments():
    assert normalize_path("$") == DefinitionPath()
    assert normalize_path("model.encoder") == DefinitionPath((Kwarg("model"), Kwarg("encoder")))
    assert normalize_path("args[0]") == DefinitionPath((Arg(0),))
    assert normalize_path("model.layer_defs[1]") == DefinitionPath((Kwarg("model"), Kwarg("layer_defs"), Index(1)))
    assert normalize_path('metadata["x.y"]') == DefinitionPath((Kwarg("metadata"), Key("x.y")))


def test_query_path_resolves_concrete_definition_subtrees():
    leaf = objects.TestClass1(10, test="leaf")
    root = objects.TestNest3(leaf, metadata={"x.y": [leaf]})

    from dryml.core.query.path import get_subtree

    assert get_subtree(root.definition, '$[@param("args")][0]') == leaf.definition
    assert get_subtree(root.definition, '$[@param("kwargs")]["metadata"]["x.y"][0]') == leaf.definition


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

    from dryml.core.query.path import get_subtree

    assert root.graph_path(path) == leaf
    assert root.graph_path('$[@param("model")]') == leaf
    assert get_subtree(root, path) == leaf
    assert root.parameters["model"] == leaf


def test_v2_parameters_reject_legacy_keyword_paths_and_raw_call_records():
    leaf = ConcreteDefinition._from_persisted_record(
        objects.TestClass1,
        identity_version=V2_IDENTITY_VERSION,
        parameters=BoundArguments((("value", 10), ("test", "leaf"))),
    )
    semantic = ConcreteDefinition._from_persisted_record(
        objects.TestNest3,
        identity_version=V2_IDENTITY_VERSION,
        parameters=BoundArguments((("model", leaf),)),
    )

    assert semantic.graph_path(DefinitionPath((Parameter("model"),))) == leaf
    with pytest.raises(QueryPathError):
        semantic.graph_path(DefinitionPath((Kwarg("model"),)))
    with pytest.raises(TypeError):
        ConcreteDefinition._from_persisted_record(objects.TestNest3, (), {"model": leaf})


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

    from dryml.core.query.fingerprint import target_local_fingerprint

    assert any(
        token.kind == "CDEF_EDGE_AT_PATH" and str(token.path) == '$[@param("sources")][1]'
        for token in target_local_fingerprint(root).counts
    )


def test_invalid_paths_report_errors():
    obj = objects.TestNest3(child=1)
    from dryml.core.query.path import get_subtree

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
    from dryml.core.query.path import get_subtree, replace_subtree
    obj = objects.TestNest3(items=("a", "b"), mapping={"k": [1, 2]})
    replaced = replace_subtree(obj.definition, '$[@param("kwargs")]["items"][1]', "c")
    replaced = replace_subtree(replaced, '$[@param("kwargs")]["mapping"]["k"][0]', 9)

    assert isinstance(get_subtree(replaced, '$[@param("kwargs")]["items"]'), tuple)
    assert get_subtree(replaced, '$[@param("kwargs")]["items"][1]') == "c"
    assert get_subtree(replaced, '$[@param("kwargs")]["mapping"]["k"][0]') == 9


def test_u4_soft_cdef_ancestor_uses_stored_parameters_without_resolution(monkeypatch):
    """Replacing beneath an unloaded CDef preserves its symbolic authority."""
    from dryml.core.utils.graph.value import replace_subtree
    from dryml.core.symbol import ImportRef

    source = ConcreteDefinition._from_persisted_record(
        ImportRef("missing.u4", "Parent"),
        identity_version=V2_IDENTITY_VERSION,
        parameters=BoundArguments((("child", 1), ("label", "source"))),
    )
    monkeypatch.setattr(
        ImportRef,
        "resolve",
        lambda self: pytest.fail("soft CDef replacement must not resolve classes"),
    )

    result = replace_subtree(
        source,
        DefinitionPath((Parameter("child"),)),
        Definition(SKIP_ARGS, selected=True),
    )

    assert result.cls == source.cls
    assert result.args is None
    assert result.parameters["label"] == "source"
    assert result.parameters["child"].parameters == {"selected": True}


def test_u4_soft_cdef_ancestor_preserves_finalized_ref_links_and_siblings(monkeypatch):
    """Replacing through one finalized link does not resolve or rewrite its alias."""
    from dryml.core.utils.graph.value import get_subtree, replace_subtree

    child = ConcreteDefinition._from_persisted_record(
        ImportRef("missing.u4", "Child"),
        identity_version=V2_IDENTITY_VERSION,
        parameters=BoundArguments((("value", 1),)),
    )
    source = ConcreteDefinition._from_persisted_record(
        ImportRef("missing.u4", "Parent"),
        identity_version=V2_IDENTITY_VERSION,
        parameters=BoundArguments(((
            "left", DefLink.finalized(EdgeKind.REF, child)),
            ("right", DefLink.finalized(EdgeKind.REF, child)),
        )),
    )
    monkeypatch.setattr(
        ImportRef,
        "resolve",
        lambda self: pytest.fail("soft CDef replacement must not resolve classes"),
    )

    result = replace_subtree(
        source,
        DefinitionPath((Parameter("left"), Parameter("value"))),
        Definition(SKIP_ARGS, selected=True),
    )

    assert get_subtree(result, DefinitionPath((Kwarg("left"), Kwarg("value")))).parameters == {"selected": True}
    assert get_subtree(result, DefinitionPath((Kwarg("right"), Parameter("value")))) == 1


def test_query_projection_does_not_mutate_source_or_drop_ordinary_controls():
    repo = Repo()
    child = objects.TestClass4(1, discriminator="child", repo=repo)
    parent = objects.TestNest3(child=child, repo=repo)
    original = parent.definition

    projected = repo.query(original).categorical(
        path='$[@param("kwargs")]["child"]',
        recursive=True,
    )

    assert parent.definition == original
    assert original.parameters["kwargs"]["child"].parameters["discriminator"] == "child"
    assert projected.selector.kwargs["kwargs"]["child"].kwargs["discriminator"] == "child"


def test_chained_query_methods_return_independent_queries():
    repo = Repo()
    source = objects.TestNest3(
        child=objects.TestClass4(1, discriminator="child", repo=repo), repo=repo,
    ).definition
    q1 = repo.query(source)
    q2 = q1.categorical(path='$[@param("kwargs")]["child"]', recursive=True)
    q3 = q2.restore()

    assert q1 is not q2
    assert q2 is not q3
    assert q1.selector.parameters["kwargs"]["child"].parameters["discriminator"] == "child"
    assert q2.selector.kwargs["kwargs"]["child"].kwargs["discriminator"] == "child"
    assert q3.selector == source


@pytest.mark.parametrize("operation", ("exact", "restore"))
def test_u4_semantic_query_projection_translates_nested_cdef_paths(operation):
    """Public projection composes exact and restore across CDef boundaries."""
    source = Definition(
        objects.TestNest2,
        Definition(
            objects.TestNest2,
            Definition(objects.TestClass1, 10, test="leaf"),
        ),
    ).concretize()
    query = Repo().query(source).categorical(recursive=True)

    transformed = getattr(query, operation)(path="A.A")

    from dryml.core.query.path import get_subtree

    assert get_subtree(transformed.selector, "A.A") == get_subtree(
        source,
        '$[@param("A")][@param("A")]',
    )


@pytest.mark.parametrize("operation", ("exact", "restore"))
def test_u4_semantic_query_projection_maps_changed_set_member_paths(operation):
    """Set-member paths retain their original occurrence after projection."""
    from dryml.core.utils.graph.value import iter_set_members

    first = Definition(U4SetLeaf, 1, seed=10).concretize()
    second = Definition(U4SetLeaf, 2, seed=20).concretize()
    source = Definition(U4SetParent, {first, second}).concretize()
    query = Repo().query(source).categorical(
        recursive=True,
        drop=("seed",),
    )
    segment, projected_child = iter_set_members(query.selector.parameters["members"])[0]
    path = DefinitionPath((Kwarg("members"), segment))
    expected = first if projected_child.parameters["value"] == 1 else second

    transformed = getattr(query, operation)(path=path)

    assert expected in transformed.selector.parameters["members"]


@pytest.mark.parametrize("operation", ("restore", "exact"))
@pytest.mark.parametrize("parameter", ("first", "width", "items", "options"))
def test_u4_semantic_projection_restores_authored_semantic_buckets(operation, parameter):
    """Chained projection retains original values for authored parameter buckets."""
    from dryml.core.query.path import get_subtree

    leaf = Definition(U4SetLeaf, 1, seed=10).concretize()
    source = Definition(
        U4VariadicParent,
        1,
        4,
        leaf,
        seed=9,
        selected=leaf,
    )
    query = Repo().query(source).categorical().categorical()
    path = DefinitionPath((Kwarg(parameter),))

    if operation == "exact" and parameter in {"first", "width", "items", "options"}:
        if parameter not in {"items", "options"}:
            with pytest.raises(TypeError, match="ConcreteDefinition"):
                query.exact(path=path)
            return
        if parameter == "items":
            path = path.child(Index(0))
        else:
            path = path.child(Key("selected"))

    transformed = getattr(query, operation)(path=path)
    expected = source.parameters[parameter]
    if operation == "exact":
        expected = expected[0] if parameter == "items" else expected["selected"]
    assert get_subtree(transformed.selector, path) == expected


@pytest.mark.parametrize("operation", ("exact", "restore"))
def test_u4_semantic_projection_set_correspondence_is_occurrence_specific(operation, monkeypatch):
    """Overlapping projected set members retain their own original CDef anchors."""
    from dryml.core.utils.graph.value import iter_set_members

    leaf_ref = ImportRef("missing.u4", "Leaf")
    first = ConcreteDefinition._from_persisted_record(
        leaf_ref,
        identity_version=V2_IDENTITY_VERSION,
        parameters=BoundArguments((("value", 1), ("seed", 10))),
    )
    second = ConcreteDefinition._from_persisted_record(
        ImportRef("missing.u4", "OtherLeaf"),
        identity_version=V2_IDENTITY_VERSION,
        parameters=BoundArguments((("value", 1), ("marker", 2), ("seed", 20))),
    )
    source = ConcreteDefinition._from_persisted_record(
        ImportRef("missing.u4", "Parent"),
        identity_version=V2_IDENTITY_VERSION,
        parameters=BoundArguments((("members", FrozenSet({first, second})),)),
    )
    resolved = []
    monkeypatch.setattr(ImportRef, "resolve", lambda self: resolved.append(self))
    query = Repo().query(source).categorical(recursive=True, drop=("seed",))
    member_segment, member = next(
        (segment, value)
        for segment, value in iter_set_members(query.selector.parameters["members"])
        if "marker" not in value.parameters
    )
    path = DefinitionPath((Kwarg("members"), member_segment))

    transformed = getattr(query, operation)(path=path)

    assert first in transformed.selector.parameters["members"]
    assert resolved == []


@pytest.mark.parametrize("operation", ("exact", "restore"))
def test_u4_repeated_root_projection_tracks_changed_set_member_occurrences(operation):
    """Repeated set projection restores the CDef before either projection."""
    from dryml.core.utils.graph.value import iter_set_members

    original_child = Definition(U4SetLeaf, 1, seed=10, another=20).concretize()
    source = Definition(U4SetParent, {original_child}).concretize()
    query = (
        Repo().query(source)
        .categorical(recursive=True, drop=("seed",))
        .categorical(recursive=True, drop=("another",))
    )
    segment, projected_child = iter_set_members(query.selector.kwargs["members"])[0]
    assert projected_child.parameters == {"value": 1}

    result = getattr(query, operation)(
        path=DefinitionPath((Kwarg("members"), segment)),
    )

    assert original_child in result.selector.kwargs["members"]


@pytest.mark.parametrize("operation", ("exact", "restore"))
def test_u4_nested_projection_composes_current_prepared_paths(operation):
    """Nested CDef edits retain semantic buckets and do not rewrite sibling aliases."""
    leaf = Definition(U4SetLeaf, 1, seed=10, another=20).concretize()
    child = Definition(
        U4VariadicParent,
        1,
        4,
        leaf,
        seed=9,
        selected=leaf,
    ).concretize()
    source = Definition(U4AliasParent, child, child).concretize()
    first = Repo().query(source).categorical(
        path=DefinitionPath((Parameter("child"),)),
        recursive=True,
        drop=("seed",),
    )
    assert first.selector.kwargs["sibling"] is child
    query = first.categorical(
        path=DefinitionPath((Kwarg("child"),)),
        recursive=True,
        drop=("another",),
    )
    path = DefinitionPath((Kwarg("child"),))

    result = getattr(query, operation)(path=path)

    assert result.selector.kwargs["child"] is child
    assert result.selector.kwargs["sibling"] is child


def test_u4_restore_none_uses_synthesized_named_positional_origin():
    """A retained original ``None`` is not confused with a missing path cache entry."""
    source = Definition(U4NoneLeaf, None)
    query = Repo().query(source).categorical()

    result = query.restore(path=DefinitionPath((Kwarg("value"),)))

    assert result.selector.kwargs["value"] is None


@pytest.mark.parametrize("operation", ("exact", "restore"))
def test_u4_projection_keeps_equal_independent_occurrence_authority(operation):
    """Equal independent CDefs retain their addressed original occurrence."""
    first = Definition(U4SetLeaf, 1, seed=10, another=20).concretize()
    second = first.copy_graph()
    source = Definition(U4ListParent, [first, second]).concretize()
    query = Repo().query(source).categorical(
        recursive=True,
        drop=("seed",),
    )
    path = DefinitionPath((Kwarg("values"), Index(1)))

    result = getattr(query, operation)(path=path)

    assert result.selector.kwargs["values"][0] is not first
    assert result.selector.kwargs["values"][1] is second


def test_u4_projection_treats_existing_prepared_selector_as_its_own_source():
    """Projecting a prepared form does not guess an earlier authored authority."""
    from dryml.core import categorical_definition

    prepared = categorical_definition(
        Definition(U4SetLeaf, 1, seed=10, another=20),
    )
    query = Repo().query(prepared).categorical(drop=("another",))

    assert query.restore().selector == prepared


@pytest.mark.parametrize("operation", ("exact", "restore"))
def test_original_path_translates_each_nested_v2_cdef_boundary(operation):
    """Query edits translate categorical paths through every V2 CDef boundary."""

    from dryml.core.query.path import get_subtree

    source = Definition(
        objects.TestNest2,
        Definition(
            objects.TestNest2,
            Definition(objects.TestClass1, 10, test="leaf"),
        ),
    ).concretize()
    query = Repo().query(source).categorical(recursive=True)

    transformed = getattr(query, operation)(path="A.A")

    assert get_subtree(transformed.selector, "A.A") == get_subtree(
        source,
        '$[@param("A")][@param("A")]',
    )


def test_restore_frozen_list_branch_preserves_query_soundness():
    repo = Repo()
    source = Definition(objects.TestNest3, items=[1, 2]).concretize(repo=repo)
    match = objects.TestNest3(items=[1, 2], repo=repo)
    other = objects.TestNest3(items=[1, 3], repo=repo)
    repo.add_objects(match, other)

    query = repo.query(source).categorical(recursive=True).restore(path="kwargs.items")

    assert isinstance(query.selector.kwargs["kwargs"]["items"], FrozenList)
    assert list(query.known(refresh=False).defs()) == [match.definition]
