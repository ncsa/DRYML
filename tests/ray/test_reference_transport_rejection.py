import pytest

from dryml.core import Repo, Serializable
from dryml.core.store.dir import DirStore
from dryml.execute.protocol import UnsupportedReferenceTransportError
from dryml.execute.transfer import prepare_call


class RayTransportValue(Serializable):
    pass


def test_ray_install_path_keeps_exact_transport_rejection_backend_free(tmp_path):
    import dryml.ray

    repo = Repo(DirStore(tmp_path / "source"))
    state = repo.save_object(RayTransportValue(repo=repo))

    with pytest.raises(UnsupportedReferenceTransportError):
        prepare_call((state.object,), {}, repo=repo, transfer_store=tmp_path / "transfer")
