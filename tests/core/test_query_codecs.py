import pytest

from dryml.core2 import Object
from dryml.core2.bound_args import BoundArguments
from dryml.core2.cdef_identity import V1_IDENTITY_VERSION, V2_IDENTITY_VERSION
from dryml.core2.definition import ConcreteDefinition
from dryml.core2.query.codecs import (
    QUERY_INDEX_CODEC_VERSION,
    QueryCodecError,
    QueryIndexCodec,
    decode_cdef,
    decode_feature_token,
    decode_graph_path,
    digest_blob,
    encode_cdef,
    encode_feature_token,
    encode_graph_path,
)
from dryml.core2.query.model import FeatureToken
from dryml.core2.query.path import Arg, GraphPath, Index, Key, Kwarg, Parameter, SetMember
from dryml.core2.query.utils import chunked, stable_hash_from_blob, stable_hash_to_blob
from dryml.core2.utils.general import pickler


class CodecLeaf(Object):
    def __init__(self, value="leaf"):
        super().__init__()
        self.value = value


def test_cdef_codec_roundtrip():
    cdef = CodecLeaf("roundtrip").definition

    assert decode_cdef(encode_cdef(cdef)) == cdef


def test_cdef_codec_retains_v1_identity_when_pickle_state_has_no_version():
    cdef = ConcreteDefinition._from_persisted_record(CodecLeaf, ("legacy",), {})
    decoded = decode_cdef(encode_cdef(cdef))

    assert isinstance(cdef.__getstate__(), list)
    assert decoded.identity_version == V1_IDENTITY_VERSION
    assert decoded == cdef
    assert decoded.stable_hash() == cdef.stable_hash()


def test_cdef_codec_decodes_private_v2_record_in_a_distinct_hash_domain():
    cdef = ConcreteDefinition._from_persisted_record(
        CodecLeaf,
        identity_version=V2_IDENTITY_VERSION,
        parameters=BoundArguments((("value", "legacy"),)),
    )
    decoded = decode_cdef(encode_cdef(cdef))
    v1 = ConcreteDefinition._from_persisted_record(CodecLeaf, ("legacy",), {})

    assert decoded.identity_version == V2_IDENTITY_VERSION
    assert decoded == cdef
    assert decoded != v1
    assert decoded.stable_hash() != v1.stable_hash()


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
