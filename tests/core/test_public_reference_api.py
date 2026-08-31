"""Lock the public CDef V2 reference and persistence surface."""

import inspect

import dryml
from dryml import core
from dryml.core.object import Object
from dryml.core.repo import Repo


def test_root_and_core_export_exact_reference_values_and_repo_apis():
    """Reference values and exact-load APIs are public while generic alias load is absent."""

    required = {
        "ObjectId", "ObjectRef", "StateRef", "StateSelectorRef",
        "object_namespace", "Repo", "load_object", "load_state_ref",
        "save_object",
    }
    assert required <= set(dryml.__all__)
    assert required <= set(core.__all__)
    assert "load_alias" not in core.__all__
    assert not hasattr(core, "load_alias")
    assert not hasattr(Repo, "load_alias")


def test_public_save_and_load_signatures_exclude_retired_controls():
    """Public structural and exact APIs do not expose mutable-state switches."""

    assert tuple(inspect.signature(Object.save).parameters) == (
        "self", "repo", "main", "store", "alias", "deep_capture",
        "federated", "report_stores",
    )
    assert tuple(inspect.signature(Repo.load_object).parameters) == ("self", "x", "cache")
    assert tuple(inspect.signature(Repo.load).parameters) == ("self", "cdef", "cache")
    assert tuple(inspect.signature(Repo.load_or_build).parameters) == ("self", "x", "cache")
    assert not hasattr(Object, "load")
