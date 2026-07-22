import inspect

import dryml.annotations as ann


def test_function_decorator_preserves_callable_metadata():
    @ann.require(namespace="environment", fragment={"requirements": ["dryml"]})
    def fn(a: int, b: str = "x") -> str:
        return f"{a}{b}"

    assert fn(1, "y") == "1y"
    assert str(inspect.signature(fn)) == "(a: int, b: str = 'x') -> str"
    assert fn.__name__ == "fn"
    assert fn.__qualname__.endswith("fn")
    assert fn.__annotations__["a"] is int


def test_class_method_stacking_and_subclass_isolation():
    @ann.require(namespace="environment", fragment={"requirements": ["base"]})
    class Base:
        @ann.default(namespace="runtime", fragment={"frameworks": {"plain": {}}})
        def method(self):
            return "ok"

    @ann.require(namespace="world", fragment={"roles": {"main": {"resources": {}}}})
    @ann.default(namespace="runtime", fragment={"frameworks": {"torch": {"num_threads": 1}}})
    class Child(Base):
        pass

    assert Base().method() == "ok"
    assert len(ann.fragments_for_class(Base)) == 1
    assert len(ann.fragments_for_class(Child)) == 3
    assert len(Base.__dict__[ann.FRAGMENT_ATTR]) == 1


def test_string_source_becomes_source_trace_label():
    @ann.require(namespace="environment", fragment={"requirements": ["dryml"]}, source="custom-source")
    def fn():
        pass

    assert ann.fragments_for(fn)[0].source.label == "custom-source"
