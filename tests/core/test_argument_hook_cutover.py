"""Public cutover coverage for categorical projection and constructor binding."""

import pytest

from dryml.core import Definition, Object, Repo
from dryml.core.bound_args import BoundArguments
from dryml.core.cdef_identity import V2_IDENTITY_VERSION
from dryml.core.definition import ConcreteDefinition
from dryml.core.symbol import ImportRef


class CutoverFixture(Object):
    """Retain ordinary constructor controls without framework injection."""

    def __init__(self, value, /, optional=2, *items, uid=None, metadata=None, **options):
        self.value = value
        self.optional = optional
        self.items = items
        self.uid = uid
        self.metadata = metadata
        self.options = options


def test_public_categorical_controls_are_named_and_do_not_construct():
    """All public categorical surfaces share the named projection engine."""

    source = Definition(CutoverFixture, 1, 4, "tail", uid="kept", metadata={"note": "kept"})

    from dryml.core import categorical_definition

    projected = source.categorical(drop=("uid",), recursive=False)
    helper = categorical_definition(source, drop=("uid",))
    queried = Repo().query(source).categorical(drop=("uid",), recursive=False)

    assert projected.args is None
    assert "uid" not in projected.parameters
    assert source.categorical().parameters["uid"] == "kept"
    assert source.categorical().parameters["metadata"] == {"note": "kept"}
    assert queried.selector.parameters["metadata"] == {"note": "kept"}
    assert helper == projected
    assert queried.selector == projected
    assert queried.restore().selector == source


def test_constructor_binding_preserves_ordinary_controls_without_hooks():
    """Direct binding retains supplied controls and does not expose retired hooks."""

    cdef = Definition(
        CutoverFixture, 1, 4, "tail", uid="caller-id", metadata={"source": "caller"}, enabled=True,
    ).concretize()

    assert cdef.parameters["uid"] == "caller-id"
    assert cdef.parameters["metadata"] == {"source": "caller"}
    assert cdef.parameters["options"] == {"enabled": True}
    assert not hasattr(Object, "__prepare_args__")
    assert not hasattr(Object, "__strip_unique_args__")
    with pytest.raises(ImportError):
        exec("from dryml.core import UniqueID", {})
    with pytest.raises(ImportError):
        exec("from dryml.core import Metadata", {})

    import dryml

    assert dryml.categorical_definition is __import__("dryml.core", fromlist=["categorical_definition"]).categorical_definition


def test_retired_mixin_authority_fails_without_rewriting_persisted_parameters():
    """Retired ImportRefs fail contextually only at construction boundaries."""

    cdef = ConcreteDefinition._from_persisted_record(
        ImportRef("dryml.core.object", "UniqueID"),
        identity_version=V2_IDENTITY_VERSION,
        parameters=BoundArguments((("uid", "kept"),)),
    )

    from dryml.core.materialization import project_cdef_call

    with pytest.raises(TypeError, match="retired UniqueID mixin authority"):
        project_cdef_call(cdef)
    assert cdef.parameters["uid"] == "kept"
