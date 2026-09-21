from pathlib import Path

import pytest

from dryml.core.store.dir import DirStore
from dryml.core.store.store import StoreAuthorityError


def test_v2_store_is_rejected_without_mutating_existing_authority(tmp_path):
    root = tmp_path / "store"
    store = DirStore(root)
    format_path = Path(store.store_format_path)
    original = format_path.read_bytes()
    v2 = original.replace(b"K\x03", b"K\x02")
    format_path.write_bytes(v2)

    with pytest.raises(StoreAuthorityError):
        DirStore.open_existing(root)

    assert format_path.read_bytes() == v2
