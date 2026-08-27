"""Static and class method collection coverage."""

from dryml.annotations import fragments_for_method, require


def test_static_and_class_method_descriptors_collect_without_binding_or_calling():
    def static_body():
        return "static"
    def class_body(cls):
        return cls.__name__
    static_descriptor = staticmethod(static_body)
    class_descriptor = classmethod(class_body)
    require(namespace="runtime", fragment={"limits": {"static": 1}})(static_descriptor)
    require(namespace="runtime", fragment={"limits": {"class": 1}})(class_descriptor)

    class Subject:
        static = static_descriptor
        method = class_descriptor

    assert len(fragments_for_method(Subject, "static")) == 1
    assert len(fragments_for_method(Subject, "method")) == 1
