import pytest

from dryml.core import Definition, ObjectId, Par, Ref, Repo, Selector
from dryml.core.freeze import FrozenDict
from dryml.core.object import Object


class SelectorFixture(Object):
    def __init__(self, value, optional="default"):
        self.value = value
        self.optional = optional


class SelectorContainer(Object):
    """Expose a model component without constructing it during navigation."""

    def __init__(self, model, peer=None):
        raise AssertionError("selector access constructed an Object")


class DerivedSelectorFixture(SelectorFixture):
    """Distinguish inherited from exact-class selector matching."""


def test_selector_exposes_only_supplied_semantic_parameters():
    selector = Selector(Definition(SelectorFixture, 4))

    assert selector.parameters == FrozenDict({"value": 4})
    assert selector.value == 4
    assert not hasattr(selector, "optional")
    with pytest.raises(AttributeError):
        selector.missing


@pytest.mark.parametrize("strict,cls_policy", ((True, "exact"), (False, "selector")))
def test_definition_attributes_return_policy_preserving_nested_selectors(strict, cls_policy):
    leaf = Definition(SelectorFixture, 4)
    branch = Definition(SelectorContainer, leaf)
    root = Definition(SelectorContainer, branch)
    selector = Selector(root, strict=strict, cls_policy=cls_policy)

    assert isinstance(selector.model, Selector)
    assert selector.model.root is branch
    assert selector.model.model.root is leaf
    assert selector.model.model.value == 4
    assert selector.model.strict is strict and selector.model.model.strict is strict
    assert selector.model.cls_policy == cls_policy and selector.model.model.cls_policy == cls_policy
    assert selector.parameters["model"] is branch
    assert selector.model.parameters["model"] is leaf
    assert selector.model.exact_root is None
    assert not hasattr(selector.model.model, "optional")


def test_concrete_child_selector_retains_exact_handle_facets_and_aliases():
    child = Definition(SelectorFixture, 4).concretize().with_identity(ObjectId(("child",)))
    selector = Selector(Definition(SelectorContainer, child, peer=child))

    assert isinstance(selector.model, Selector)
    assert selector.model.root is child and selector.model.exact_root is child
    assert selector.peer.root is selector.model.root
    assert selector.model.root.object_id is child.object_id
    assert selector.model.matches(child)
    assert not selector.model.matches(Definition(SelectorFixture, 5).concretize())


def test_child_selector_rebases_retained_exact_authority_for_restore():
    exact = Definition(SelectorContainer, Definition(SelectorFixture, 4)).concretize()
    parent = Selector(exact, strict=True, cls_policy="exact").categorical(recursive=True, drop=("optional",))
    child = parent.model

    assert isinstance(child, Selector)
    assert child.exact_root is exact.parameters["model"]
    assert "optional" not in child.parameters
    assert child.restore().root.graph_equal(exact.parameters["model"])
    assert child.exact().root.graph_equal(exact.parameters["model"])
    assert "optional" not in parent.model.parameters
    assert child.restore().strict and child.restore().cls_policy == "exact"


def test_child_attribute_preserves_scalars_containers_and_quotation_boundaries():
    child = Definition(SelectorFixture, 4)
    link = Ref(child.quote())
    selector = Selector(Definition(SelectorFixture, {"child": child}, optional=link))

    assert isinstance(selector.value, FrozenDict)
    assert selector.value["child"] is child
    assert selector.optional is link
    assert selector.optional.unwrap() is link.target.value
    assert isinstance(selector.optional.unwrap(), Definition)
    assert selector.root is not child


def test_child_attribute_requires_no_resolution_defaults_construction_or_repo(monkeypatch):
    from dryml.core import object as object_module, symbol

    child = Definition(SelectorFixture, 4)
    exact = Definition(SelectorContainer, child).concretize()
    selector = Selector(exact).categorical(recursive=True, drop=("optional",))
    def forbidden(*args, **kwargs):
        pytest.fail("selector component access crossed a metadata-only boundary")
    monkeypatch.setattr(symbol, "resolve_symbol", forbidden)
    monkeypatch.setattr(object_module, "_normalize_definition_arguments", forbidden)
    monkeypatch.setattr(Definition, "concretize", forbidden)
    monkeypatch.setattr(Repo, "describe", forbidden)
    assert selector.model.exact_root is exact.parameters["model"]
    assert "optional" not in selector.model.parameters


@pytest.mark.parametrize("query_index", ("memory", "sqlite"))
def test_loose_model_component_is_queryable_with_inherited_exact_policy(tmp_path, query_index):
    from dryml.core.store.dir import DirStore

    template = Definition(SelectorContainer, Definition(SelectorFixture, Par("width")))
    exp_sel = template.loose_selector()
    model_sel = exp_sel.model
    assert isinstance(model_sel, Selector)
    assert model_sel.cls_policy == "exact"
    repo = Repo(DirStore(tmp_path / "store", query_index=query_index))
    try:
        for width in (32, 64):
            repo.default_store.publish_node(template.sub(width=width).concretize())
        repo.default_store.publish_node(Definition(DerivedSelectorFixture, 32).concretize())
        assert repo.query().sel(model_sel).count() == 2
        assert repo.query().sel(exp_sel).nested(model_sel).targets().count() == 2
        assert model_sel.matches(Definition(SelectorFixture, 32).concretize())
        assert not model_sel.matches(Definition(DerivedSelectorFixture, 32).concretize())
    finally:
        repo.close(flush=False)
