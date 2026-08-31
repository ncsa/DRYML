from pathlib import Path

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
