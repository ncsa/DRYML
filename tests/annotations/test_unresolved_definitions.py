"""Definition/CDef method collection never resolves a symbolic class."""

from dryml.annotations import UnresolvedAnnotationResult, collect_fragments, fragments_for_definition_method, require
from dryml.core import Definition, SKIP_ARGS
from dryml.core.symbol import ImportRef


def test_unresolved_definition_requires_live_class_without_importing_it():
    definition = Definition(ImportRef("optional_backend.never_import", "Missing"), SKIP_ARGS)
    result = fragments_for_definition_method(definition, "method")
    assert isinstance(result, UnresolvedAnnotationResult)

    class Live:
        @require(namespace="runtime", fragment={"limits": {"threads": 1}})
        def method(self):
            return None
    assert len(fragments_for_definition_method(definition, "method", live_cls=Live)) == 1


def test_collecting_a_definition_never_uses_its_live_class_implicitly():
    """Definition collection requires the caller to select a live target."""

    @require(namespace="runtime", fragment={"limits": {"threads": 1}})
    class Live:
        pass

    result = collect_fragments(Definition(Live))
    assert isinstance(result, UnresolvedAnnotationResult)
