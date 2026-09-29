from io import BytesIO

import pytest

from dryml.core.store.store import StoreCapabilityError, StorePublicationCapabilities
from dryml.core.store.zip import ZipStore


def test_capability_preflight_fails_closed_before_local_state_hooks(tmp_path):
    caps = StorePublicationCapabilities(False, False, False, False, False)
    with pytest.raises(StoreCapabilityError, match="writable"):
        caps.require_writable("save", local_state=True)

    store = ZipStore(BytesIO())
    try:
        with pytest.raises(StoreCapabilityError, match="writable"):
            store.preflight_publication("save", local_state=True)
    finally:
        store.close()
