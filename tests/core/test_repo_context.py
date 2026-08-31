from dryml.core import Object, Repo
from dryml.core.repo import get_default_repo, manage_repo


class ContextAware(Object):
    def __init__(self):
        super().__init__()
        self.repo = get_default_repo()


def test_manage_repo_none_is_context_local_and_temporary(monkeypatch):
    assert get_default_repo() is None
    closed = []
    original_close = Repo.close
    monkeypatch.setattr(Repo, "close", lambda self, *args, **kwargs: closed.append(self))

    with manage_repo(None) as repo:
        assert get_default_repo() is repo
        assert ContextAware().repo is repo

    assert closed == [repo]
    assert get_default_repo() is None
    monkeypatch.setattr(Repo, "close", original_close)
