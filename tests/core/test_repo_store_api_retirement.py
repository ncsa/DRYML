"""The retired compound-reference API has no import or capability shims."""

import importlib

import pytest

from dryml.core import Repo, RepoDefinition, RepoDefinitionError
from dryml.core.store import DirStore, Store, ZipStore


def test_retired_reference_and_record_imports_fail():
    names = ("ObjectRef", "StateRef", "StateSelectorRef", "ObjectSelector")
    retired = {
        "dryml": names,
        "dryml.core": (*names, "ObjectSelectorSet", "LineageMetadata", "SnapshotMetadata"),
        "dryml.core.reference_values": names[:3],
        "dryml.core.selector": ("ObjectSelector",),
        "dryml.core.query.identity": ("ObjectSelectorSet",),
        "dryml.core.query.model": ("ReferenceOccurrence",),
        "dryml.core.repo": ("ReferenceEvidence", "ReferenceDeclarationEvidence", "ReferenceStateEvidence"),
        "dryml.core.store.records": (
            "DefinitionRecord", "StoredRootRecord", "LocalStateManifest",
            "StateRefRecord", "DeclarationRecord", "ClaimRecord", "MainRefRecord",
            "ObjectAliasRecord", "StateAliasRecord",
        ),
    }
    for module, symbols in retired.items():
        for symbol in symbols:
            with pytest.raises(ImportError, match=symbol):
                exec(f"from {module} import {symbol}", {})


def test_retired_reference_codec_module_is_absent():
    with pytest.raises(ModuleNotFoundError, match="dryml.core.reference_json"):
        importlib.import_module("dryml.core.reference_json")


def test_retired_repo_and_store_capabilities_are_absent():
    for name in (
        "declare_object", "build_object_ref", "abandon_object_ref",
        "fork_object_ref", "fork_state_ref", "get_or_declare_object_ref",
        "reference_evidence", "find_object_refs", "lookup_object_ref", "find_object_ref",
        "resolve_state_selector", "set_alias", "get_alias", "resolve_object_alias",
        "delete_alias", "aliases", "set_state_alias", "resolve_state_alias",
        "save_object_if_current", "get_snapshot_directory", "has_cdef_light", "set_main_def",
    ):
        assert not hasattr(Repo, name), name
    for store_type in (Store, DirStore, ZipStore):
        for name in (
            "read_definition_record", "write_definition_record", "iter_definition_records",
            "read_stored_root_record", "iter_stored_root_records",
            "read_state_ref_record", "iter_state_ref_records", "publish_snapshot",
            "get_snapshot_directory", "read_snapshot_metadata",
            "read_declaration_record", "write_declaration_record", "iter_declaration_records",
            "read_claim_record", "write_claim_record", "read_main_ref", "write_main_ref",
            "read_object_alias", "write_object_alias", "read_state_alias", "write_state_alias",
            "compare_and_set_state_alias", "open_local_state", "prepare_local_state",
        ):
            assert not hasattr(store_type, name), (store_type, name)


def test_claim_configuration_is_retired_and_old_repo_definitions_are_refused():
    for argument, value in (("clock", lambda: 0), ("lease_duration", 30),
                            ("owner_token_factory", lambda: "owner")):
        with pytest.raises(TypeError, match=argument):
            Repo(**{argument: value})
    current = Repo().to_definition().to_data()
    assert current["version"] == 2
    assert "lease_duration" not in current["settings"]
    current["version"] = 1
    current["settings"]["lease_duration"] = 30
    with pytest.raises(RepoDefinitionError, match="version"):
        RepoDefinition.from_data(current)
