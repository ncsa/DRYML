import dryml.annotations as ann
from dryml.core2.arg_roles import RefCDef


@ann.require(namespace="environment", fragment={"requirements": ["module-owner"]})
class _ModuleOwner:
    @ann.require(namespace="world", fragment={"roles": {"main": {"resources": {}}}})
    def run(self):
        return None


def test_collection_filters_and_provider_fragments():
    provider = ann.AnnotationFragment("environment", "requirement", {"requirements": ["provider"]}, ann.SourceTrace("provider"))

    @ann.require(namespace="environment", fragment={"requirements": ["fn"]})
    @ann.default(namespace="runtime", fragment={"frameworks": {"plain": {}}})
    def fn(model: RefCDef):
        return model

    assert [f.kind for f in ann.fragments_for(fn)] == ["default", "requirement"]
    assert [f.namespace for f in ann.fragments_for(fn, namespace="environment")] == ["environment"]
    assert ann.collect_fragments((fn,), provider_fragments=(provider,), namespace="environment")[-1] is provider


def test_class_method_and_multiple_inheritance_order():
    @ann.require(namespace="environment", fragment={"requirements": ["a"]})
    class A:
        pass

    @ann.require(namespace="environment", fragment={"requirements": ["b"]})
    class B:
        pass

    class C(A, B):
        @ann.require(namespace="world", fragment={"roles": {"main": {"resources": {}}}})
        def run(self):
            return None

    assert [f.fragment["requirements"][0] for f in ann.fragments_for_class(C, namespace="environment")] == ["a", "b"]
    assert [f.namespace for f in ann.fragments_for_callable(C().run)] == ["environment", "environment", "world"]


def test_deep_inheritance_order_is_base_to_subclass():
    @ann.require(namespace="environment", fragment={"requirements": ["base"]})
    class Base:
        pass

    @ann.require(namespace="environment", fragment={"requirements": ["mid"]})
    class Mid(Base):
        pass

    @ann.require(namespace="environment", fragment={"requirements": ["leaf"]})
    class Leaf(Mid):
        pass

    assert [f.fragment["requirements"][0] for f in ann.fragments_for_class(Leaf, namespace="environment")] == ["base", "mid", "leaf"]


def test_module_level_unbound_method_includes_owner_class_fragments():
    assert [f.namespace for f in ann.fragments_for_callable(_ModuleOwner.run)] == ["environment", "world"]
