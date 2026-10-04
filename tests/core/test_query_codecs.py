import pytest

from dryml.core import Definition, F, Object, ObjectId, ObjectRef
from dryml.core.bound_args import BoundArguments
from dryml.core.cdef_identity import V2_IDENTITY_VERSION
from dryml.core.definition import ConcreteDefinition
from dryml.core.query.codecs import (
    CDEF_CODEC_VERSION,
    FEATURE_CODEC_VERSION,
    QUERY_INDEX_CODEC_VERSION,
    QueryCodecError,
    QueryIndexCodec,
    decode_cdef,
    decode_feature_token,
    decode_graph_path,
    decode_metadata_predicate,
    decode_reference,
    digest_blob,
    encode_cdef,
    encode_feature_token,
    encode_graph_path,
    encode_metadata_predicate,
    encode_reference,
)
from dryml.core.query.metadata import MetadataField, MetadataPredicate, field
from dryml.core.query.model import FeatureToken
from dryml.core.query.path import Arg, GraphPath, Index, Key, Kwarg, Parameter, SetMember
from dryml.core.query.utils import chunked, stable_hash_from_blob, stable_hash_to_blob
from dryml.core.utils.general import pickler


class CodecLeaf(Object):
    def __init__(self, value="leaf"):
        super().__init__()
        self.value = value


def test_cdef_codec_roundtrip():
    cdef = CodecLeaf("roundtrip").definition

    assert decode_cdef(encode_cdef(cdef)) == cdef


def test_cdef_codec_roundtrips_inert_concrete_factory_values():
    """Derived query payloads retain concrete FactorySpecs without resolution."""

    cdef = CodecLeaf(F("unresolved_factory_target", 7, flag=True)).definition

    decoded = decode_cdef(encode_cdef(cdef))

    assert decoded.parameters["value"] == F("unresolved_factory_target", 7, flag=True)


def test_reference_query_promotion_increments_its_index_codec_markers():
    """Reference index rows use a new semantic and feature codec boundary."""

    assert CDEF_CODEC_VERSION == 4
    assert FEATURE_CODEC_VERSION == 4
    assert QUERY_INDEX_CODEC_VERSION == 6


def test_reference_codec_round_trips_complete_values():
    """Derived reference rows retain complete ObjectId and ObjectRef values."""

    definition = Definition(CodecLeaf, "reference").concretize()
    for value in (ObjectId(("test",)), ObjectRef(definition, {})):
        assert decode_reference(encode_reference(value)) == value


def test_cdef_codec_decodes_current_v2_record():
    cdef = ConcreteDefinition._from_persisted_record(
        CodecLeaf,
        identity_version=V2_IDENTITY_VERSION,
        parameters=BoundArguments((("value", "legacy"),)),
    )
    decoded = decode_cdef(encode_cdef(cdef))

    assert decoded.identity_version == V2_IDENTITY_VERSION
    assert decoded == cdef


def test_query_index_codec_facade_roundtrips_all_payloads():
    codec = QueryIndexCodec()
    cdef = CodecLeaf("facade").definition
    token = FeatureToken("SCALAR", GraphPath((Kwarg("items"), Index(1))), "value")
    path = GraphPath((Arg(0), Kwarg("items"), Index(1), Key("name"), SetMember("abc", 2)))

    assert codec.version == QUERY_INDEX_CODEC_VERSION
    assert codec.decode_cdef(codec.encode_cdef(cdef)) == cdef
    assert codec.decode_feature_token(codec.encode_feature_token(token)) == token
    assert codec.decode_graph_path(codec.encode_graph_path(path)) == path
    assert codec.digest_blob(b"payload") == digest_blob(b"payload")


def test_feature_codec_roundtrip():
    token = FeatureToken(
        "SCALAR",
        GraphPath((Kwarg("items"), Index(2), Key("name"), SetMember("abc", 1))),
        ("payload", 7),
    )

    assert decode_feature_token(encode_feature_token(token)) == token


def test_feature_codec_roundtrip_without_path():
    token = FeatureToken("ROOT", None, "payload")

    assert decode_feature_token(encode_feature_token(token)) == token


def test_path_codec_roundtrip():
    path = GraphPath((Kwarg("model"), Key("encoder"), SetMember("def", 0)))

    assert decode_graph_path(encode_graph_path(path)) == path


def test_graph_path_codec_roundtrip_all_segments():
    path = GraphPath((Arg(0), Kwarg("model"), Index(3), Key(("tuple", 1)), SetMember("def", 2)))

    assert decode_graph_path(encode_graph_path(path)) == path


@pytest.mark.parametrize(
    "segment",
    [
        Parameter("model"),
        Kwarg("model"),
        Arg(0),
        Index(3),
        Key(("tuple", 1)),
        SetMember("def", 2),
    ],
)
def test_typed_segment_codec_closure(segment):
    path = GraphPath((segment,))

    assert decode_graph_path(encode_graph_path(path)) == path


def test_path_codec_keeps_v1_and_v2_path_kinds_distinct():
    legacy = GraphPath((Kwarg("model"),))
    semantic = GraphPath((Parameter("model"),))

    assert decode_graph_path(encode_graph_path(legacy)) == legacy
    assert decode_graph_path(encode_graph_path(semantic)) == semantic
    assert legacy != semantic


def test_codec_rejects_corruption():
    with pytest.raises(QueryCodecError):
        decode_cdef(b"not a pickle envelope")


def test_codec_rejects_wrong_type():
    with pytest.raises(QueryCodecError):
        decode_cdef(pickler({"kind": "cdef", "version": 1, "payload": "not a cdef"}))


def test_codec_rejects_version_mismatch():
    with pytest.raises(QueryCodecError):
        decode_graph_path(pickler({"kind": "graph-path", "version": 999, "payload": GraphPath().to_data()}))


def test_metadata_fields_and_predicates_validate_and_round_trip():
    """Metadata predicates remain inert, bounded, and canonically serializable."""

    selector = field("object", "nested", 0)
    predicate = (
        selector.exists()
        & field("object", "missing").missing()
        & selector.eq({"names": ("vision",), "enabled": True})
        & field("object", "number").lt(8)
        & field("object", "number").le(7)
        & field("object", "number").gt(3)
        & field("object", "number").ge(4)
        & field("object", "text").contains("is")
    ) | ~field("object", "tags").contains("other")

    decoded = decode_metadata_predicate(encode_metadata_predicate(predicate))

    assert decoded.to_data() == predicate.to_data()
    assert decoded.fingerprint() == predicate.fingerprint()
    assert encode_metadata_predicate(decoded) == encode_metadata_predicate(predicate)
    with pytest.raises(ValueError):
        field("captured", "value")
    with pytest.raises(ValueError):
        field("object", True)
    with pytest.raises(ValueError):
        field("object", -1)
    with pytest.raises(ValueError):
        field("object", *("x",) * 33)
    with pytest.raises(ValueError, match="4096"):
        field("object", "x" * 4097)
    with pytest.raises(ValueError, match="4096"):
        MetadataField("object", ("x" * 4097,))
    with pytest.raises(TypeError, match="tuple"):
        MetadataField("object", ["value"])
    with pytest.raises(QueryCodecError, match="field data"):
        MetadataPredicate.from_data({
            "kind": "exists",
            "field": {"scope": "object", "path": ["x" * 4097]},
        })
    with pytest.raises(TypeError):
        bool(predicate)


def test_metadata_predicate_bounds_reject_noncanonical_input():
    """Predicate trees and values fail before exceeding public codec bounds."""

    leaf = field("object", "value").exists()
    leaves = [leaf for _ in range(256)]
    while len(leaves) > 1:
        leaves = [left | right for left, right in zip(leaves[::2], leaves[1::2])]
    predicate = leaves[0]
    decoded = decode_metadata_predicate(encode_metadata_predicate(predicate))
    assert decoded.to_data() == predicate.to_data()
    assert decoded.fingerprint() == predicate.fingerprint()
    with pytest.raises(ValueError, match="256"):
        predicate | leaf

    predicate = leaf
    for _ in range(32):
        predicate = ~predicate
    with pytest.raises(ValueError, match="32"):
        ~predicate

    nested = 0
    for _ in range(9):
        nested = [nested]
    with pytest.raises(ValueError, match="nesting"):
        field("object", "value").eq(nested)
    blob = encode_metadata_predicate(leaf)
    with pytest.raises(QueryCodecError, match="noncanonical"):
        decode_metadata_predicate(blob + b" ")
    with pytest.raises(QueryCodecError):
        decode_metadata_predicate(b'{"kind":"metadata-predicate"}')
    deeply_nested = leaf.to_data()
    for _ in range(33):
        deeply_nested = {"kind": "not", "operand": deeply_nested}
    with pytest.raises(QueryCodecError, match="bounds"):
        MetadataPredicate.from_data(deeply_nested)


def test_hash_blob_roundtrip():
    digest = "00ff10"

    assert stable_hash_from_blob(stable_hash_to_blob(digest)) == digest


def test_digest_blob_is_stable_sha256_bytes():
    assert digest_blob(b"abc") == digest_blob(b"abc")
    assert digest_blob(b"abc") != digest_blob(b"abcd")
    assert len(digest_blob(b"abc")) == 32


def test_chunked():
    assert list(chunked(range(5), 2)) == [(0, 1), (2, 3), (4,)]
    with pytest.raises(ValueError):
        list(chunked(range(1), 0))
