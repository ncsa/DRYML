"""Integrity helpers for the closed qualification reader fixture set."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import pickletools
from typing import Any


_MANIFEST_SCHEMA = "dryml-qualification-reader-fixtures"
_MANIFEST_VERSION = 1
_FIXTURE_FIELDS = {
    "artifact_value/value.pkl": {"format", "version", "sha256"},
    "experiment_data.json": {"format", "version", "sha256"},
    "template_bundle.json": {"format", "kind", "version", "sha256"},
    "train_state.json": {"fixture", "fixture_version", "sha256"},
}
_UNSAFE_PICKLE_OPCODES = frozenset({
    "BUILD", "EXT1", "EXT2", "EXT4", "GLOBAL", "INST", "NEWOBJ",
    "NEWOBJ_EX", "OBJ", "PERSID", "BINPERSID", "REDUCE", "STACK_GLOBAL",
})


def verify_qualification_fixture_manifest(root: Path) -> dict[str, Any]:
    """Validate the closed manifest and file hashes before a fixture is decoded.

    Args:
        root: Directory containing the committed or freshly generated fixture set.

    Returns:
        The validated manifest mapping.

    Raises:
        AssertionError: If the manifest schema, file list, or a listed file hash
            is not the expected closed v1 fixture contract.
    """

    manifest = json.loads((root / "manifest.json").read_bytes().decode("ascii"))
    assert type(manifest) is dict
    assert set(manifest) == {"schema", "version", "generator", "provenance", "fixtures"}
    assert manifest["schema"] == _MANIFEST_SCHEMA
    assert manifest["version"] == _MANIFEST_VERSION
    assert manifest["generator"] == "generate.py"
    assert type(manifest["provenance"]) is str and manifest["provenance"]
    fixtures = manifest["fixtures"]
    assert type(fixtures) is dict and set(fixtures) == set(_FIXTURE_FIELDS)

    for relative, required_fields in _FIXTURE_FIELDS.items():
        path = PurePosixPath(relative)
        assert not path.is_absolute() and ".." not in path.parts
        entry = fixtures[relative]
        assert type(entry) is dict and set(entry) == required_fields
        digest = entry["sha256"]
        assert type(digest) is str and len(digest) == 64
        assert all(character in "0123456789abcdef" for character in digest)
        assert sha256((root / path).read_bytes()).hexdigest() == digest

    return manifest


def assert_value_fixture_has_no_executable_pickle_opcodes(path: Path) -> None:
    """Reject executable pickle instructions before the Value fixture is unpickled.

    Args:
        path: Hash-verified ``value.pkl`` fixture path.

    Raises:
        AssertionError: If the static pickle stream contains an opcode capable of
            resolving globals, persistent identities, or object reconstruction.
    """

    opcodes = {opcode.name for opcode, _, _ in pickletools.genops(path.read_bytes())}
    assert not opcodes & _UNSAFE_PICKLE_OPCODES
