import core2_objects as objects

from dryml.core2.query import Arg, DefinitionPath, Index, Key, Kwarg, normalize_path


def test_query_path_parses_root_and_segments():
    assert normalize_path("$") == DefinitionPath()
    assert normalize_path("model.encoder") == DefinitionPath((Kwarg("model"), Kwarg("encoder")))
    assert normalize_path("args[0]") == DefinitionPath((Arg(0),))
    assert normalize_path("model.layer_defs[1]") == DefinitionPath((Kwarg("model"), Kwarg("layer_defs"), Index(1)))
    assert normalize_path('metadata["x.y"]') == DefinitionPath((Kwarg("metadata"), Key("x.y")))


def test_query_path_resolves_concrete_definition_subtrees():
    leaf = objects.TestClass1(10, test="leaf")
    root = objects.TestNest3(leaf, metadata={"x.y": [leaf]})

    from dryml.core2.query.path import get_subtree

    assert get_subtree(root.definition, "args[0]") == leaf.definition
    assert get_subtree(root.definition, 'metadata["x.y"][0]') == leaf.definition
