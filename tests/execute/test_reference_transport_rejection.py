import pytest

from dryml.core import Repo, Serializable
from dryml.core.store.dir import DirStore
from dryml.execute.transfer import prepare_call
from dryml.execute.protocol import UnsupportedReferenceTransportError


class TransportValue(Serializable):
    pass


@pytest.mark.parametrize("reference", [lambda state: state, lambda state: state.object], ids=["state", "object"])
def test_exact_reference_transport_rejects_before_opening_transfer_store(tmp_path, reference):
    repo = Repo(DirStore(tmp_path / "source"))
    state = repo.save_object(TransportValue(repo=repo))
    transfer = tmp_path / "transfer"

    with pytest.raises(UnsupportedReferenceTransportError):
        prepare_call((reference(state),), {}, repo=repo, transfer_store=transfer)

    assert not transfer.exists()
