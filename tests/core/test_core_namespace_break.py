"""Exercise the intentional persisted-identity break at the core namespace."""

import importlib
import pickle

import pytest

from dryml.core.definition import ConcreteDefinition
from dryml.core.freeze import FrozenDict, FrozenTuple
from dryml.core.symbol import ImportRef


def _retired_module() -> str:
    """Build the unsupported former namespace without retaining it as source text."""

    return "dryml.core" + "2"


def test_retired_package_has_no_import_or_root_alias():
    """The retired package is absent rather than retained as a compatibility shim."""

    import dryml

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(_retired_module())
    assert not hasattr(dryml, _retired_module().rpartition(".")[2])


def test_new_pickle_globals_and_symbolic_identity_use_promoted_namespace():
    """New core values emit the promoted module path in pickle and semantic identity."""

    cdef = ConcreteDefinition._from_persisted_record(
        ImportRef("dryml.core.object", "Object"),
        FrozenTuple(()),
        FrozenDict({}),
    )

    assert b"dryml.core" in pickle.dumps(ConcreteDefinition)
    assert _retired_module().encode() not in pickle.dumps(cdef)
    assert cdef.cls.module == "dryml.core.object"


def test_retired_symbolic_reference_fails_without_translation():
    """Historical symbolic references fail through normal module resolution."""

    retired = ImportRef(_retired_module() + ".object", "Object")

    with pytest.raises(ModuleNotFoundError):
        retired.resolve()
