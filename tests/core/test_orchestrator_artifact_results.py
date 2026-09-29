"""Result restoration under process-isolated orchestration controls."""

import pytest

from dryml import session
from dryml.core import AutoRef, Object, Ref, Repo
from dryml.core.store.dir import DirStore
from dryml.runtime import materialization_scope
from dryml.runtime.errors import RuntimeTransitionError
from tests.core.test_execute_managed_integration import WorkerDependent


class ScopedResult(Object):
    """Small result whose reference-valued input stays inert during restoration."""

    def __init__(self, source: Ref[AutoRef], value: int) -> None:
        """Retain a non-materializing source reference and one scalar result."""
        self.source = source
        self.value = value


@pytest.mark.usefixtures("fixed_snapshot_environment")
def test_orchestrator_result_materialization_scope_restores_after_success_and_error(tmp_path):
    """A scoped result read restores strict orchestration without loading its input."""

    marker = tmp_path / "source-built"
    repo = Repo(DirStore(tmp_path / "state", query_index="none"))
    source = WorkerDependent(str(marker), repo=repo)
    marker.unlink()
    result = ScopedResult(source, 7, repo=repo)
    state = repo.save_object(result, deep_capture=True)
    session.set_mode("orchestrator")
    try:
        with pytest.warns(RuntimeWarning, match="explicit warn scope"):
            with materialization_scope("warn"):
                restored = repo.load_state_ref(state, reuse_live="never")
        assert restored.value == 7
        assert not marker.exists()
        with pytest.raises(RuntimeTransitionError, match="prohibits Object materialization"):
            repo.load_state_ref(state, reuse_live="never")

        with pytest.warns(RuntimeWarning, match="explicit warn scope"):
            with materialization_scope("warn"):
                with pytest.raises(RuntimeError, match="reader failure"):
                    repo.load_state_ref(state, reuse_live="never")
                    raise RuntimeError("reader failure")
        with pytest.raises(RuntimeTransitionError, match="prohibits Object materialization"):
            repo.load_state_ref(state, reuse_live="never")
        assert not marker.exists()
    finally:
        session.reset()
