import pytest
from typing import Annotated

import dryml
from dryml.core2 import (
    CDefEdge,
    CDefNode,
    ConcreteDefinitionGraph,
    ConcreteDefinitionGraphError,
    Definition,
    EdgeKind,
    FrozenConcreteDefinition,
    FrozenDefinition,
    Object,
    Repo,
    Serializable,
    UniqueID,
)
from dryml.core2.arg_roles import FrozenCDefArg, FrozenDefArg, resolve_arg_roles
from dryml.core2.definition import ConcreteDefinition
from dryml.core2.freeze import FrozenDict, FrozenTuple
from dryml.core2.materialization import build_materialization_plan
from dryml.core2.policies import RepoLoadOptions
from dryml.core2.query.sqlite.schema import DDL, compatibility_decision, expected_semantic_version
from dryml.core2.query.sqlite import SQLiteQueryIndexConfig
from dryml.core2.query.path import Arg, GraphPath
from dryml.core2.query.selector_graph import compile_selector_graph
from dryml.core2.store.dir import DirStore


class FrozenLeaf(Serializable):
    constructed = 0

    def __init__(self, name):
        super().__init__()
        self.name = name
        type(self).constructed += 1


class FrozenOwner(Object):
    def __init__(self, child: dryml.FrozenCDef, *, label="owner"):
        super().__init__()
        self.child = child
        self.label = label


class FrozenOwnerByMap(Object):
    __dryml_arg_roles__ = {"child": "frozen_cdef"}

    def __init__(self, child):
        super().__init__()
        self.child = child


class FrozenOwnerByAnnotated(Object):
    def __init__(self, child: Annotated[ConcreteDefinition, FrozenCDefArg()] = None):
        super().__init__()
        self.child = child


class FrozenOwnerConflict(Object):
    __dryml_arg_roles__ = {"child": "materialize"}

    def __init__(self, child: dryml.FrozenCDef):
        super().__init__()
        self.child = child


class FrozenOwnerInvalidName(Object):
    __dryml_arg_roles__ = {"missing": "frozen_cdef"}

    def __init__(self, child):
        super().__init__()
        self.child = child


class FrozenOwnerInvalidRole(Object):
    __dryml_arg_roles__ = {"child": "not_a_role"}

    def __init__(self, child):
        super().__init__()
        self.child = child


class FrozenSelectorOwner(Object):
    def __init__(self, selector: dryml.FrozenDef):
        super().__init__()
        self.selector = selector


class FrozenSelectorOwnerByAnnotated(Object):
    def __init__(self, selector: Annotated[Definition, FrozenDefArg()]):
        super().__init__()
        self.selector = selector


class MaterializingOwner(Object):
    def __init__(self, child):
        super().__init__()
        self.child = child


class FrozenContainerOwner(Object):
    def __init__(self, values):
        super().__init__()
        self.values = values


class PreparedFrozenOwner(Object):
    @classmethod
    def __prepare_args__(cls, *args, **kwargs):
        if "child" not in kwargs:
            kwargs["child"] = args[0]
            args = ()
        return args, kwargs

    def __init__(self, child: dryml.FrozenCDef):
        super().__init__()
        self.child = child


class UidFrozen(UniqueID):
    pass


def test_concrete_definition_freeze_roundtrip():
    cdef = Definition(FrozenLeaf, "x").concretize()

    frozen = cdef.freeze()

    assert isinstance(frozen, FrozenConcreteDefinition)
    assert frozen.thaw() == cdef
    assert hash(frozen) == hash(cdef.freeze())
    assert frozen != cdef
    assert "FrozenConcreteDefinition" in repr(frozen)


def test_definition_freeze_roundtrip_independent_and_uid_safe():
    definition = Definition(UidFrozen)
    frozen = definition.freeze()

    assert isinstance(frozen, FrozenDefinition)
    assert "uid" not in frozen.kwargs
    thawed = frozen.thaw()
    assert isinstance(thawed, Definition)
    thawed.kwargs["extra"] = 1
    assert "extra" not in frozen.kwargs
    assert "uid" in thawed.concretize().kwargs


def test_dryml_freeze_helper_inputs():
    repo = Repo()
    obj = FrozenLeaf("x", repo=repo)
    cdef = obj.definition
    definition = Definition(FrozenLeaf, "y")

    assert dryml.freeze(obj).thaw() == cdef
    assert dryml.freeze(cdef).thaw() == cdef
    assert isinstance(dryml.freeze(definition), FrozenDefinition)
    frozen = cdef.freeze()
    assert dryml.freeze(frozen) is frozen
    with pytest.raises(TypeError):
        dryml.freeze(1)


def test_annotation_frozen_cdef_role_from_object_and_runtime_presentation():
    repo = Repo()
    child = FrozenLeaf("child", repo=repo)
    owner = FrozenOwner(child, repo=repo)

    assert isinstance(owner.definition.args[0], FrozenConcreteDefinition)
    assert owner.definition.args[0].thaw() == child.definition
    assert owner.child == child.definition


def test_annotation_frozen_cdef_role_from_cdef_and_default():
    child_cdef = Definition(FrozenLeaf, "child").concretize()

    owner_cdef = Definition(FrozenOwnerByAnnotated, child_cdef).concretize()
    default_cdef = Definition(FrozenOwnerByAnnotated).concretize()

    assert isinstance(owner_cdef.args[0], FrozenConcreteDefinition)
    assert owner_cdef.args[0].thaw() == child_cdef
    assert default_cdef.args == FrozenTuple(())


def test_annotation_inherited_constructor_role():
    class InheritedFrozenOwner(FrozenOwner):
        pass

    child_cdef = Definition(FrozenLeaf, "child").concretize()

    owner_cdef = Definition(InheritedFrozenOwner, child_cdef).concretize()

    assert isinstance(owner_cdef.args[0], FrozenConcreteDefinition)
    assert owner_cdef.args[0].thaw() == child_cdef


def test_class_arg_roles_fallback_and_cdef_input():
    child_cdef = Definition(FrozenLeaf, "child").concretize()
    owner_cdef = Definition(FrozenOwnerByMap, child_cdef).concretize()

    assert isinstance(owner_cdef.args[0], FrozenConcreteDefinition)
    assert owner_cdef.args[0].thaw() == child_cdef


def test_role_resolution_priority_and_validation():
    roles = resolve_arg_roles(FrozenOwnerConflict)

    assert isinstance(roles["child"], FrozenCDefArg)
    with pytest.raises(ValueError, match="Unknown DRYML argument role"):
        resolve_arg_roles(FrozenOwnerInvalidName)
    with pytest.raises(TypeError, match="Invalid DRYML argument role"):
        resolve_arg_roles(FrozenOwnerInvalidRole)


def test_annotation_frozen_def_role_from_definition():
    selector = Definition(FrozenLeaf, "model")
    owner = FrozenSelectorOwner(selector)

    assert isinstance(owner.definition.args[0], FrozenDefinition)
    assert isinstance(owner.selector, Definition)
    assert owner.selector is not selector


def test_annotation_frozen_def_role_from_annotated_definition():
    selector = Definition(FrozenLeaf, "model")

    owner = FrozenSelectorOwnerByAnnotated(selector)

    assert isinstance(owner.definition.args[0], FrozenDefinition)
    assert isinstance(owner.selector, Definition)
    assert owner.selector is not selector


def test_frozen_roles_preserve_explicit_wrappers_and_reject_invalid_values():
    child_cdef = Definition(FrozenLeaf, "child").concretize()
    frozen_cdef = child_cdef.freeze()
    frozen_def = Definition(FrozenLeaf, "selector").freeze()

    owner_cdef = Definition(FrozenOwner, frozen_cdef).concretize()
    selector_owner_cdef = Definition(FrozenSelectorOwner, frozen_def).concretize()

    assert owner_cdef.args[0] is frozen_cdef
    assert selector_owner_cdef.args[0] is frozen_def
    with pytest.raises(TypeError, match="FrozenCDef argument expects"):
        Definition(FrozenOwner, "not a cdef").concretize()
    with pytest.raises(TypeError, match="FrozenDef argument expects"):
        Definition(FrozenSelectorOwner, child_cdef).concretize()


def test_unmarked_cdef_arg_still_materializes():
    repo = Repo()
    child = FrozenLeaf("child", repo=repo)
    owner = MaterializingOwner(child, repo=repo)

    assert owner.definition.args[0] == child.definition
    assert not isinstance(owner.definition.args[0], FrozenConcreteDefinition)
    assert owner.child is child


def test_no_name_or_artifact_heuristic_freezing():
    class ModelNamedOwner(Object):
        def __init__(self, model):
            super().__init__()
            self.model = model

    class ArtifactLikeOwner(Object):
        def __init__(self, child):
            super().__init__()
            self.child = child

    child_cdef = Definition(FrozenLeaf, "child").concretize()

    assert Definition(ModelNamedOwner, child_cdef).concretize().args[0] == child_cdef
    assert Definition(ArtifactLikeOwner, child_cdef).concretize().args[0] == child_cdef


def test_role_canonicalization_after_prepare_preserves_explicit_wrapper():
    child_cdef = Definition(FrozenLeaf, "child").concretize()
    explicit = child_cdef.freeze()

    prepared = Definition(PreparedFrozenOwner, child_cdef).concretize()
    preserved = Definition(PreparedFrozenOwner, explicit).concretize()

    assert isinstance(prepared.args[0], FrozenConcreteDefinition)
    assert prepared.args[0].thaw() == child_cdef
    assert preserved.args[0] is explicit


def test_frozen_wrappers_affect_hash_and_containers_are_preserved():
    child_cdef = Definition(FrozenLeaf, "child").concretize()
    materialized = Definition(FrozenContainerOwner, [child_cdef]).concretize()
    frozen = Definition(FrozenContainerOwner, [child_cdef.freeze()]).concretize()

    assert materialized.stable_hash() != frozen.stable_hash()
    assert isinstance(frozen.args[0][0], FrozenConcreteDefinition)


def test_container_of_frozen_refs_runtime_presented():
    repo = Repo()
    child = FrozenLeaf("child", repo=repo)
    owner = FrozenContainerOwner([child.definition.freeze()], repo=repo)

    assert owner.values == [child.definition]
    assert isinstance(owner.definition.args[0][0], FrozenConcreteDefinition)


def test_frozen_cdef_creates_frozen_edge_without_expanding_target_children():
    repo = Repo()
    inner = FrozenLeaf("inner", repo=repo)
    materialized_child = MaterializingOwner(inner, repo=repo)
    frozen_owner = FrozenOwner(materialized_child, repo=repo)

    graph = ConcreteDefinitionGraph.from_root(frozen_owner.definition)
    edges = graph.edges()

    assert len(edges) == 1
    assert edges[0].kind is EdgeKind.FROZEN
    assert edges[0].child == materialized_child.definition
    assert inner.definition not in {node.definition for node in graph.nodes()}
    assert tuple(graph.iter_occurrences(target=materialized_child.definition)) == ()


def test_raw_cdef_creates_materialize_edge_and_occurrence_kind():
    repo = Repo()
    child = FrozenLeaf("child", repo=repo)
    owner = MaterializingOwner(child, repo=repo)

    graph = ConcreteDefinitionGraph.from_root(owner.definition)
    edge = graph.edges()[0]
    occurrence = tuple(graph.iter_occurrences(target=child.definition))[0]

    assert edge.kind is EdgeKind.MATERIALIZE
    assert occurrence.kind is EdgeKind.MATERIALIZE
    assert graph.primary_path(owner.definition, child.definition) == edge.path


def test_materialize_and_frozen_edges_can_share_path_and_child():
    child_cdef = Definition(FrozenLeaf, "child").concretize()
    materialized = Definition(MaterializingOwner, child_cdef).concretize()
    frozen = Definition(FrozenOwner, child_cdef).concretize()

    graph = ConcreteDefinitionGraph.from_roots((materialized, frozen))
    edges = {(edge.parent, edge.path, edge.child, edge.kind) for edge in graph.edges()}

    assert (materialized, GraphPath((Arg(0),)), child_cdef, EdgeKind.MATERIALIZE) in edges
    assert (frozen, GraphPath((Arg(0),)), child_cdef, EdgeKind.FROZEN) in edges


def test_graph_validation_checks_frozen_edge_path():
    child_cdef = Definition(FrozenLeaf, "child").concretize()
    parent = Definition(FrozenOwner, child_cdef).concretize()

    with pytest.raises(ConcreteDefinitionGraphError, match="frozen reference"):
        ConcreteDefinitionGraph(
            (parent,),
            (CDefNode(parent, parent.stable_hash()), CDefNode(child_cdef, child_cdef.stable_hash())),
            (CDefEdge(parent, GraphPath((Arg(0),)), child_cdef, EdgeKind.MATERIALIZE),),
        )
    material_parent = Definition(MaterializingOwner, child_cdef).concretize()
    with pytest.raises(ConcreteDefinitionGraphError, match="does not resolve to a FrozenConcreteDefinition"):
        ConcreteDefinitionGraph(
            (material_parent,),
            (CDefNode(material_parent, material_parent.stable_hash()), CDefNode(child_cdef, child_cdef.stable_hash())),
            (CDefEdge(material_parent, GraphPath((Arg(0),)), child_cdef, EdgeKind.FROZEN),),
        )


def test_materialization_plan_excludes_frozen_target_and_load_receives_cdef():
    repo = Repo()
    child = FrozenLeaf("child", repo=repo)
    owner_cdef = Definition(FrozenOwner, child).concretize(repo=repo)
    repo.clear_cache(strong=True, weak=True)

    plan = build_materialization_plan(
        repo,
        owner_cdef,
        RepoLoadOptions(build_missing=True, restore_state=False),
        memo={},
    )

    assert child.definition not in plan.actions
    owner = repo.load_object(owner_cdef, build_missing=True, restore_state=False)
    assert owner.child == child.definition


def test_frozen_target_can_be_loaded_explicitly_later(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    child = FrozenLeaf("child", repo=repo)
    owner = FrozenOwner(child, repo=repo)
    repo.save_object(child)
    repo.save_object(owner)
    repo.clear_cache(strong=True, weak=True)

    loaded_owner = repo.load_object(owner.definition, restore_state=False)
    loaded_child = repo.load_object(loaded_owner.child, restore_state=False)

    assert loaded_owner.child == child.definition
    assert loaded_child.definition == child.definition


def test_missing_frozen_ref_does_not_break_load(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    child = FrozenLeaf("child", repo=repo)
    owner = FrozenOwner(child, repo=repo)
    repo.save_object(owner)
    repo.clear_cache(strong=True, weak=True)

    loaded = repo.load_object(owner.definition, restore_state=False)

    assert loaded.child == child.definition


def test_save_does_not_save_frozen_target_by_default(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    child = FrozenLeaf("child", repo=repo)
    owner = FrozenOwner(child, repo=repo)

    repo.save_object(owner)

    assert store.has(owner.definition)
    assert not store.has(child.definition)


@pytest.mark.parametrize("query_index", ["memory", SQLiteQueryIndexConfig(journal_mode="delete")])
def test_query_matches_frozen_ref_but_not_materialize_edge(tmp_path, query_index):
    store = DirStore(tmp_path / "store", query_index=query_index)
    repo = Repo(stores=store)
    child = FrozenLeaf("child", repo=repo)
    frozen_owner = FrozenOwner(child, repo=repo)
    material_owner = MaterializingOwner(child, repo=repo)
    repo.save_object(frozen_owner)
    repo.save_object(material_owner)

    frozen_selector = Definition(FrozenOwner, child)
    material_selector = Definition(MaterializingOwner, child)

    assert tuple(repo.query(frozen_selector).stored().defs()) == (frozen_owner.definition,)
    assert tuple(repo.query(material_selector).stored().defs()) == (material_owner.definition,)


def test_nested_default_uses_materialize_edges_only(tmp_path):
    store = DirStore(tmp_path / "store", query_index="memory")
    repo = Repo(stores=store)
    child = FrozenLeaf("child", repo=repo)
    frozen_owner = FrozenOwner(child, repo=repo)
    material_owner = MaterializingOwner(child, repo=repo)
    repo.save_object(frozen_owner)
    repo.save_object(material_owner)

    owners = tuple(repo.query(child).nested().owners().defs())

    assert owners == (material_owner.definition,)


def test_selector_graph_shows_edge_kind_constraint():
    child_cdef = Definition(FrozenLeaf, "child").concretize()

    frozen_graph = compile_selector_graph(Definition(FrozenOwner, child_cdef))
    material_graph = compile_selector_graph(Definition(MaterializingOwner, child_cdef))

    assert frozen_graph.edges[0].edge_kind is EdgeKind.FROZEN
    assert material_graph.edges[0].edge_kind is EdgeKind.MATERIALIZE


def test_sqlite_schema_and_edge_kind_rows(tmp_path):
    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=store)
    child = FrozenLeaf("child", repo=repo)
    frozen_owner = FrozenOwner(child, repo=repo)
    material_owner = MaterializingOwner(child, repo=repo)
    repo.save_object(frozen_owner)
    repo.save_object(material_owner)

    assert tuple(repo.query(Definition(FrozenOwner, child)).stored().defs()) == (frozen_owner.definition,)
    index = store.open_query_index()
    with index.read_view():
        con = index._connections.connection(readonly=True)
        columns = {row[1] for row in con.execute("PRAGMA table_info(definition_edges)")}
        kinds = {row[0] for row in con.execute("SELECT edge_kind FROM definition_edges")}

    assert "edge_kind" in columns
    assert kinds == {"materialize", "frozen"}


def test_sqlite_old_schema_requests_migrate_or_rebuild():
    expected = expected_semantic_version(store_key="store")
    old_schema = expected.catalog_state() | {"schema_version": expected.schema_version - 1}
    old_graph = expected.catalog_state() | {"graph_schema_version": expected.graph_schema_version - 1}

    assert compatibility_decision(old_schema, expected=expected) == "migrate"
    assert compatibility_decision(old_graph, expected=expected) == "rebuild"
    assert any("edge_kind" in statement for statement in DDL)


def test_sqlite_parent_child_lookups_filter_edge_kind(tmp_path):
    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=store)
    child = FrozenLeaf("child", repo=repo)
    frozen_owner = FrozenOwner(child, repo=repo)
    material_owner = MaterializingOwner(child, repo=repo)
    repo.save_object(frozen_owner)
    repo.save_object(material_owner)
    tuple(repo.query(Definition(FrozenOwner, child)).stored().defs())

    index = store.open_query_index()
    with index.read_view() as view:
        child_id = view.cdef_id(child.definition)
        frozen_parent_id = view.cdef_id(frozen_owner.definition)
        material_parent_id = view.cdef_id(material_owner.definition)
        frozen_parent_ids = view.parents({child_id}, GraphPath((Arg(0),)), unordered=False, edge_kind=EdgeKind.FROZEN)
        material_parent_ids = view.parents({child_id}, GraphPath((Arg(0),)), unordered=False, edge_kind=EdgeKind.MATERIALIZE)
        frozen_child_ids = view.children({frozen_parent_id}, GraphPath((Arg(0),)), unordered=False, edge_kind=EdgeKind.FROZEN)
        material_child_ids = view.children({frozen_parent_id}, GraphPath((Arg(0),)), unordered=False, edge_kind=EdgeKind.MATERIALIZE)

    assert set(frozen_parent_ids) == {frozen_parent_id}
    assert set(material_parent_ids) == {material_parent_id}
    assert set(frozen_child_ids) == {child_id}
    assert set(material_child_ids) == set()


def test_accuracy_and_plot_artifact_style_examples(tmp_path):
    class Accuracy(Object):
        def __init__(self, data: dryml.FrozenCDef, model: dryml.FrozenCDef, value=None):
            super().__init__()
            self.data = data
            self.model = model
            self.value = value

        def compute(self, repo):
            data = repo.load_object(self.data, restore_state=False)
            model = repo.load_object(self.model, restore_state=False)
            self.value = (data.name, model.name)
            return self.value

    class PlotA(Object):
        def __init__(self, models: dryml.FrozenDef):
            super().__init__()
            self.models = models

        def compute(self, repo):
            return tuple(repo.query(self.models).stored().defs())

    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=store)
    data = FrozenLeaf("data", repo=repo)
    model = FrozenLeaf("model", repo=repo)
    repo.save_object(data)
    repo.save_object(model)
    accuracy = Accuracy(data, model, value=1.0, repo=repo)
    plot = PlotA(Definition(FrozenLeaf, "model"), repo=repo)
    repo.save_object(accuracy)
    repo.save_object(plot)
    repo.clear_cache(strong=True, weak=True)

    loaded_accuracy_def = repo.query(Definition(Accuracy, data, model, value=1.0)).stored().defs().one()
    loaded_accuracy = repo.load_object(loaded_accuracy_def, restore_state=False)
    loaded_plot = repo.load_object(plot.definition, restore_state=False)

    assert loaded_accuracy.value == 1.0
    assert loaded_accuracy.compute(repo) == ("data", "model")
    assert isinstance(loaded_plot.models, Definition)
    assert loaded_plot.compute(repo) == (model.definition,)
