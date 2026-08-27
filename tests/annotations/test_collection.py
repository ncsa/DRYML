"""C3, override, descriptor, and decorator-order collection coverage."""

from dryml.annotations import collect_fragments, fragments_for_method, require


def _label(label):
    return require(namespace="runtime", fragment={"labels": [label]}, source=None)


def test_decorator_order_c3_diamond_and_override_order_are_deterministic():
    @_label("root")
    class Root:
        @_label("root-method")
        def method(self):
            return "root"
    @_label("left")
    class Left(Root):
        pass
    @_label("right")
    class Right(Root):
        pass
    @_label("leaf")
    class Leaf(Left, Right):
        @_label("leaf-method")
        def method(self):
            return "leaf"
    labels = [item.fragment["labels"][0] for item in fragments_for_method(Leaf, "method")]
    assert labels == ["root", "right", "left", "leaf", "leaf-method"]
    assert "root-method" not in labels


def test_descriptor_precedes_underlying_function_and_direct_stack_is_inside_out():
    @_label("outer")
    @_label("inner")
    def function():
        return None
    descriptor = staticmethod(function)
    _label("descriptor")(descriptor)
    class Subject:
        method = descriptor
    labels = [item.fragment["labels"][0] for item in fragments_for_method(Subject, "method")]
    assert labels == ["descriptor", "inner", "outer"]
    assert [item.fragment["labels"][0] for item in collect_fragments(function)] == ["inner", "outer"]
