import pytest

from dryml.core2.query.path import (
    Arg,
    DefinitionPath,
    Index,
    Key,
    Kwarg,
    QueryPathError,
    SetMember,
    get_subtree,
    iter_set_members,
    normalize_path,
    parse_path,
)
from dryml.core2.utils.graph import GraphCtx, GraphPath


def test_query_path_types_are_shared_graph_types():
    assert DefinitionPath is GraphPath
    assert normalize_path("model.encoder") == GraphPath((Kwarg("model"), Kwarg("encoder")))


def test_graph_path_equality_and_hash_contract():
    a = GraphPath((Kwarg("model"),))
    b = GraphPath((Kwarg("model"),))

    assert a == b
    assert hash(a) == hash(b)
    assert a != ("model",)
    assert a.to_legacy_tuple() == ("model",)


def test_graph_ctx_accepts_graph_path_and_legacy_children():
    ctx = GraphCtx(path=GraphPath((Kwarg("root"),)))
    child = ctx.child("items").child(2)

    assert child.path.to_legacy_tuple() == ("root", "items", 2)
    assert child.path_str() == "root/items/2"


def test_graph_path_join_parent_and_relative_to():
    path = GraphPath((Kwarg("model"),)).join(GraphPath((Kwarg("encoder"), Index(0))))

    assert str(path) == "$.model.encoder[0]"
    assert path.parent == GraphPath((Kwarg("model"), Kwarg("encoder")))
    assert path.relative_to("model") == GraphPath((Kwarg("encoder"), Index(0)))


def test_arg_and_index_are_not_equal():
    assert Arg(0) != Index(0)
    assert str(GraphPath((Arg(0),))) == "$.args[0]"
    assert str(GraphPath((Index(0),))) == "$[0]"


def test_kwarg_and_key_are_not_equal():
    assert Kwarg("x") != Key("x")
    assert str(GraphPath((Kwarg("x"),))) == "$.x"
    assert str(GraphPath((Key("x"),))) == "$['x']"


def test_set_member_path_string_round_trips():
    path = GraphPath((Kwarg("items"), SetMember("abc", 2)))

    assert parse_path(str(path)) == path


def test_set_member_path_resolves_stably():
    member = ("needle", 1)
    value = {member, ("other", 2)}
    path = GraphPath((SetMember("", 0),))
    for seg, child in iter_set_members(value):
        if child == member:
            path = GraphPath((seg,))
            break

    assert get_subtree(value, path) == member


@pytest.mark.parametrize(
    "path",
    [
        GraphPath((Arg(0),)),
        GraphPath((Kwarg("model"),)),
        GraphPath((Index(5),)),
        GraphPath((Key("name"),)),
        GraphPath((Key(5),)),
        GraphPath((SetMember("abc", 2),)),
    ],
)
def test_graph_path_canonical_roundtrip(path):
    assert GraphPath.from_data(path.to_data()) == path


def test_integer_mapping_key_string_roundtrip_does_not_become_index():
    path = GraphPath((Key(5),))

    assert str(path) == "$[@key(5)]"
    assert parse_path(str(path)) == path


def test_set_member_collision_order_is_rejected(monkeypatch):
    from dryml.core2.utils.graph import value as value_mod

    class Ambiguous:
        def __repr__(self):
            return "Ambiguous()"

    left = Ambiguous()
    right = Ambiguous()

    monkeypatch.setattr(value_mod, "stable_hash_function", lambda value: "collision")

    with pytest.raises(QueryPathError):
        iter_set_members({left, right})
