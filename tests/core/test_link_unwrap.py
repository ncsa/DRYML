"""Explicit link unwrapping exposes retained metadata without selecting authority."""

import pytest

from dryml.core import Definition, Mat, Node, Object, ObjectId, Par, Ref, Selector
from dryml.core.cdef_graph import EdgeKind
from dryml.core.links import DefLink
from dryml.core.quoted import QuotedDef, SelectorSpec


class UnwrapLeaf(Object):
    """Provide inert recipe metadata without allowing query-time construction."""

    def __init__(self, value):
        raise AssertionError("unwrapping constructed an Object")


@pytest.mark.parametrize("role", (Ref, Mat))
def test_assertion_unwrap_preserves_original_live_target_and_role(role):
    value = object()
    link = role(value)
    assert link.unwrap() is value
    assert link.target is value and not link.is_finalized
    assert link.kind is (EdgeKind.REF if role is Ref else EdgeKind.MATERIALIZE)


@pytest.mark.parametrize("kind", (EdgeKind.REF, EdgeKind.MATERIALIZE))
def test_structural_unwrap_retains_exact_node_facets_and_aliases(kind):
    child = Definition(UnwrapLeaf, 7).concretize().with_identity(ObjectId(("unwrap",)))
    parent = Definition(UnwrapLeaf, (child, child)).concretize()
    link = DefLink.finalized(kind, parent)
    value = link.unwrap()
    assert value is parent
    assert value.parameters["value"][0] is child
    assert value.parameters["value"][1] is child
    assert value.parameters["value"][0].object_id is child.object_id
    assert link.is_finalized and link.target is parent


@pytest.mark.parametrize("finalized", (False, True))
def test_unwrap_definition_data_reveals_recipe_without_binding_its_expressions(finalized):
    recipe = Definition(UnwrapLeaf, Par("this.value"))
    quoted = QuotedDef(recipe)
    link = DefLink.finalized(EdgeKind.REF, quoted) if finalized else Ref(quoted)
    assert link.unwrap() is quoted.value
    assert link.unwrap().names == ("this",)
    assert link.target is quoted
    assert link.unwrap().sub(this=Definition(UnwrapLeaf, 7)).parameters["value"] == 7


@pytest.mark.parametrize("finalized", (False, True))
def test_unwrap_selector_data_preserves_its_policy(finalized):
    selector = Selector(Definition(UnwrapLeaf, 7), strict=True, cls_policy="exact")
    quoted = SelectorSpec(selector)
    link = DefLink.finalized(EdgeKind.REF, quoted) if finalized else Ref(quoted)
    assert link.unwrap() is quoted.selector
    assert link.unwrap().strict is True and link.unwrap().cls_policy == "exact"
    assert link.target is quoted


def test_unwrap_removes_only_the_outer_quotation_carrier():
    inner = QuotedDef(Definition(UnwrapLeaf, Par("value")))
    outer = QuotedDef(inner)
    link = DefLink.finalized(EdgeKind.REF, outer)
    assert link.unwrap() is outer.value
    assert isinstance(link.unwrap(), QuotedDef)
    nested_link = Ref(Definition(UnwrapLeaf, 7))
    assert Ref(nested_link).unwrap() is nested_link


def test_unwrap_does_not_resolve_bind_normalize_or_access_repo(monkeypatch):
    from dryml.core import Repo, object as object_module, symbol

    node = Definition(UnwrapLeaf, Par("value"))
    link = Ref(node.quote())
    def forbidden(*args, **kwargs):
        pytest.fail("unwrap crossed a metadata-only boundary")
    monkeypatch.setattr(symbol, "resolve_symbol", forbidden)
    monkeypatch.setattr(object_module, "_normalize_definition_arguments", forbidden)
    monkeypatch.setattr(Definition, "sub", forbidden)
    monkeypatch.setattr(Node, "build", forbidden)
    monkeypatch.setattr(Repo, "describe", forbidden)
    monkeypatch.setattr(Repo, "materialize_boundary", forbidden)
    assert link.unwrap() is link.target.value


def test_unwrap_experiment_metric_recipe_can_build_a_query_pattern(tmp_path):
    import numpy as np

    from dryml.core import Repo
    from dryml.core.store.dir import DirStore
    from dryml.data import ArrayDataset
    from dryml.metrics import regressor_mse
    from dryml.models import Experiment
    from tests.models.test_experiment_artifacts import CounterModel, TerminalOnly

    recipe = regressor_mse(Par("this.test_data"), Par("this.model"), mode="global")
    template = Definition(Experiment, Definition(CounterModel, Par("width")), Definition(TerminalOnly),
                          test_data=Definition(ArrayDataset, np.asarray([1.0, 2.0])), artifacts={"mse": recipe})
    exp_sel = template.loose_selector()
    art = exp_sel.artifacts["mse"].unwrap()
    assert isinstance(art, Definition)
    pattern = art.sub(this=exp_sel.root, traverse_refs=True)
    repo = Repo(DirStore(tmp_path / "store", query_index="sqlite"))
    try:
        for width in (32, 64):
            checkpoint = template.sub(width=width).concretize()
            saved_metric = recipe.sub(this=checkpoint, traverse_refs=True).concretize()
            repo.default_store.publish_node(saved_metric)
        assert repo.query().sel(pattern).count() == 2
    finally:
        repo.close(flush=False)
