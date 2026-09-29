"""Verify obsolete Store roots reject without creating current authority."""

from pathlib import Path

import pytest

from dryml.core.store.dir import DirStore
from dryml.core.store.store import StoreAuthorityError


def test_old_store_layout_rejects_before_creating_format_or_catalog(tmp_path):
    """A pre-port object root is never treated as an empty current Store."""

    root = tmp_path / "old-store"
    old = root / "objects" / "legacy"
    old.mkdir(parents=True)
    payload = old / "definition.pkl"
    payload.write_bytes(b"old authority")
    before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}

    with pytest.raises(StoreAuthorityError, match="store-format|old or mixed"):
        DirStore(root)

    after = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    assert after == before
    assert not (root / "store-format.record").exists()
