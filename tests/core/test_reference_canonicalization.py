"""Retired reference tags and lossy definition exports fail closed."""

import pytest

from dryml.core import Definition, ObjectId
from dryml.core.cdef_codec import CDefGraphCodecError, decode_cdef_graph, encode_cdef_graph
from dryml.core.object import Object, Serializable


class CanonicalLeaf(Serializable):
    pass


class CanonicalOwner(Object):
    def __init__(self, value):
        self.value = value


@pytest.mark.parametrize("kind", ("object_ref", "state_ref", "state_selector_ref"))
def test_cdef_graph_codec_rejects_retired_reference_tags(kind):
    cdef = Definition(CanonicalOwner, "value").concretize()
    authority = encode_cdef_graph(cdef)
    authority["nodes"][0]["parameters"]["items"][0][1] = {"kind": kind}

    with pytest.raises(CDefGraphCodecError, match="Unknown CDef graph value kind"):
        decode_cdef_graph(authority)


def test_legacy_definition_surface_export_refuses_node_facet_loss():
    """A call-surface projection cannot silently erase identity-3 constraints."""
    from dryml.core.canonical import thaw_definition_surface_value

    pinned = Definition(CanonicalLeaf).concretize().with_identity(ObjectId())
    with pytest.raises(CDefGraphCodecError, match="identity-3"):
        thaw_definition_surface_value(pinned)
