import pytest

from dryml.core2 import Object
from dryml.core2.query.codecs import (
    QueryCodecError,
    decode_cdef,
    decode_feature_token,
    decode_graph_path,
    digest_blob,
    encode_cdef,
    encode_feature_token,
    encode_graph_path,
)
from dryml.core2.query.model import FeatureToken
from dryml.core2.query.path import GraphPath, Index, Key, Kwarg, SetMember
from dryml.core2.query.utils import chunked, stable_hash_from_blob, stable_hash_to_blob
from dryml.core2.utils.general import pickler


class CodecLeaf(Object):
    def __init__(self, value="leaf"):
        super().__init__()
        self.value = value


def test_cdef_codec_roundtrip():
    cdef = CodecLeaf("roundtrip").definition

    assert decode_cdef(encode_cdef(cdef)) == cdef


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
