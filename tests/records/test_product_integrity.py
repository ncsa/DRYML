from __future__ import annotations

import hashlib

import pytest

from dryml.core2.store.dir import DirStore
from dryml.formats.refs import format_cdef_id
from dryml.records import (
    ProductManifest,
    ProductManifestEntry,
    ProductWriteSession,
    RecordValidationError,
    StorageRef,
    StoredStateRecord,
    default_object_state_representation_spec,
    require_product_integrity,
    validate_product_availability,
)


def _record(manifest):
    return StoredStateRecord(
        format_cdef_id("a" * 64),
        default_object_state_representation_spec()["id"],
        (StorageRef.self_product(),),
        manifest=manifest.to_json(),
    ).to_envelope()


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        (lambda root: (root / "extra.bin").write_bytes(b"extra"), "unexpected_product_path"),
        (lambda root: (root / "data.bin").write_bytes(b"short"), "product_size_mismatch"),
        (lambda root: (root / "data.bin").write_bytes(b"xxxxxxxx"), "product_digest_mismatch"),
    ],
)
def test_product_integrity_checks_exact_file_set_size_and_digest(tmp_path, mutation, code):
    store = DirStore(tmp_path / code)
    with ProductWriteSession(store.records) as session:
        session.write_bytes("data.bin", b"payload!")
        result = session.commit_record(_record(session.manifest()))
    record = store.records.read_record(result.located.record_id)

    mutation(result.product_root)
    issues = validate_product_availability(store.records, record)

    assert code in {issue.code for issue in issues}
    with pytest.raises(RecordValidationError, match="integrity"):
        require_product_integrity(store.records, record)


def test_product_manifest_rejects_duplicate_paths():
    digest = hashlib.sha256(b"x").hexdigest()
    with pytest.raises(RecordValidationError, match="duplicate"):
        ProductManifest(
            (
                ProductManifestEntry("x.bin", 1, digest),
                ProductManifestEntry("x.bin", 1, digest),
            )
        )
