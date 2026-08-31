from pathlib import Path

import dill
import pytest

from dryml.core.store.dir import DirStore
from dryml.core.store.store import StoreAuthorityError


def test_new_store_has_only_the_current_format_gate(tmp_path):
    store = DirStore(tmp_path / "store")

    assert Path(store.store_format_path).is_file()
    assert not (Path(store.base_dir) / "objects").exists()


def test_nonempty_ungated_store_fails_without_mutation(tmp_path):
    root = tmp_path / "old"
    root.mkdir()
    retired = root / "objects"
    retired.mkdir()

    with pytest.raises(StoreAuthorityError, match="store-format"):
        DirStore(root)

    assert retired.is_dir()


def test_previous_store_format_is_rejected_without_mutation(tmp_path):
    root = tmp_path / "previous"
    root.mkdir()
    gate = root / "store-format.record"
    payload = {
        "schema": "store-format",
        "version": 1,
        "format_version": 1,
    }
    original = (
        b"DRYML-STORE-RECORD/store-format/1\n"
        + dill.dumps(payload, protocol=5)
    )
    gate.write_bytes(original)

    with pytest.raises(StoreAuthorityError, match="Malformed Store record"):
        DirStore(root)

    assert gate.read_bytes() == original
