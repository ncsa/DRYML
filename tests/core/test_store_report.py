from dryml.core import Repo, Serializable, StoreReport
from dryml.core.store.dir import DirStore


class ReportState(Serializable):
    def save_state_to_dir_imp(self, dest_dir, *, codec):
        pass


def test_store_report_is_ephemeral_complete_and_not_state_identity(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    state, report = ReportState(repo=repo).save(repo=repo, report_stores=True)

    assert isinstance(report, StoreReport)
    assert report.target_store is store
    assert tuple(report.state_stores) == tuple(state.states)
    assert report.required_stores == (store,)
    assert "StoreReport" not in repr(state)
