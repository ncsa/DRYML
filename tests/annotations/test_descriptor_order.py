"""Descriptor storage ordering and identity-deduplication coverage."""

from dryml.annotations import fragments_for_method, require


def test_same_fragment_on_descriptor_and_function_is_emitted_once():
    def function():
        return "ok"
    descriptor = staticmethod(function)
    decorator = require(namespace="runtime", fragment={"frameworks": {"plain": {}}})
    decorator(descriptor)
    fragment = descriptor.__dryml_annotation_fragments__[0]
    function.__dryml_annotation_fragments__ = (fragment,)

    class Subject:
        method = descriptor

    assert fragments_for_method(Subject, "method") == (fragment,)
