"""Concurrent derived completion must not replay authoritative publication."""

from contextlib import ExitStack, closing

import pytest

from dryml.core import Definition, Object, Repo
from dryml.core.store import authority
from dryml.core.store.dir import DirStore
from tests.core.test_store_report import ReportState


pytestmark = pytest.mark.usefixtures("fixed_snapshot_environment")


def test_save_retries_projection_cut_drift_without_recapture_or_republication(tmp_path, monkeypatch):
    from dryml.core.query import authority as projection

    with ExitStack() as stack:
        store = stack.enter_context(closing(DirStore(tmp_path / "store")))
        peer = stack.enter_context(closing(DirStore.open_existing(store.base_dir)))
        repo = Repo(store)
        stack.callback(repo.close, flush=False)
        store.publish_node(Definition(Object).concretize())
        capture = projection.capture_store_projection
        save = ReportState.save_state_to_dir_imp
        apply = authority.apply
        captures, applications, moved = [], [], []

        def capture_and_advance(owner, **kwargs):
            prepared = capture(owner, **kwargs)
            if owner is store and prepared.receipts and not moved:
                moved.append(True)
                peer.publish_node(Definition(ReportState, 123).concretize())
            return prepared

        def counted_save(obj, *args, **kwargs):
            captures.append(obj)
            return save(obj, *args, **kwargs)

        def counted_apply(owner, batch, **kwargs):
            result = apply(owner, batch, **kwargs)
            if owner is store and result[0] is not None:
                applications.append(result[0])
            return result

        monkeypatch.setattr(projection, "capture_store_projection", capture_and_advance)
        monkeypatch.setattr(ReportState, "save_state_to_dir_imp", counted_save)
        monkeypatch.setattr(authority, "apply", counted_apply)
        obj = ReportState(9, repo=repo)
        saved, report = repo.save(obj, report_stores=True)

        assert moved == [True] and captures == [obj]
        assert len(applications) == 1
        assert report.publications[0].mutation_id == applications[0]
        assert not store.authority_cut().pending
        assert store.open_query_index().status().state == "ready"
        assert repo.load_state_ref(saved, reuse_live="never").value == 9


@pytest.mark.parametrize("failure, attempts", [
    (authority.StoreCutChanged, 8), (OSError, 1), (KeyboardInterrupt, 1),
])
def test_failed_projection_keeps_one_publication_and_repairs_without_recapture(
        tmp_path, monkeypatch, failure, attempts):
    from dryml.core.query import authority as projection
    from dryml.core.repo import RepoSaveError

    with closing(DirStore(tmp_path / "store")) as store:
        repo = Repo(store)
        try:
            store.publish_node(Definition(Object).concretize())
            calls, captures = [], []
            save = ReportState.save_state_to_dir_imp

            def fail(*args, **kwargs):
                calls.append(True)
                raise failure("injected projection interruption")

            def counted_save(obj, *args, **kwargs):
                captures.append(obj)
                return save(obj, *args, **kwargs)

            with monkeypatch.context() as patch:
                patch.setattr(projection, "capture_store_projection", fail)
                patch.setattr(ReportState, "save_state_to_dir_imp", counted_save)
                patch.setattr(authority, "_pause_cut_retry", lambda attempt: None)
                with pytest.raises(KeyboardInterrupt if failure is KeyboardInterrupt else RepoSaveError) as raised:
                    repo.save(ReportState(13, repo=repo))
            assert len(calls) == attempts and len(captures) == 1
            assert len(store.authority_cut().pending) == 1
            assert raised.value.report.publications
            store.open_query_index().maintain_store()
            assert not store.authority_cut().pending
            saved = repo.query().saved().one()
            assert repo.load_state_ref(saved, reuse_live="never").value == 13
            assert len(captures) == 1
        finally:
            repo.close(flush=False)


@pytest.mark.parametrize("boundary", ["retention", "adoption", "previous-capture", "recovery-probe"])
def test_cooperating_writer_drift_retries_only_retained_metadata(tmp_path, monkeypatch, boundary):
    with ExitStack() as stack:
        store = stack.enter_context(closing(DirStore(tmp_path / "store")))
        peer = stack.enter_context(closing(DirStore.open_existing(store.base_dir)))
        repo = Repo(store)
        stack.callback(repo.close, flush=False)
        obj = ReportState(1, repo=repo)
        saved, report = repo.save(obj, report_stores=True)
        token = report.publications[0].retention_token
        moves, captures = [], []
        save = ReportState.save_state_to_dir_imp

        def advance():
            if not moves:
                moves.append(True)
                peer.publish_node(Definition(ReportState, 100).concretize())

        def counted_save(obj, *args, **kwargs):
            captures.append(obj)
            return save(obj, *args, **kwargs)

        monkeypatch.setattr(ReportState, "save_state_to_dir_imp", counted_save)
        if boundary == "retention":
            path = store.retention._path(token)
            reader = store._read_file

            def read(path_arg, record_type):
                result = reader(path_arg, record_type)
                if path_arg == path:
                    advance()
                return result

            monkeypatch.setattr(store, "_read_file", read)
            assert store.retention.read(token).token == token
            assert not captures
        elif boundary == "adoption":
            from dryml.core.store.retention import RetentionGrantRecord
            reader = store._read_file

            def read(path_arg, record_type):
                result = reader(path_arg, record_type)
                if record_type is RetentionGrantRecord:
                    advance()
                return result

            monkeypatch.setattr(store, "_read_file", read)
            assert store.retention.adopt(token, (store,)).disposition == "adopted"
            assert not captures
        elif boundary == "previous-capture":
            reader = store._read_node_metadata

            def read(*args, **kwargs):
                result = reader(*args, **kwargs)
                advance()
                return result

            monkeypatch.setattr(store, "_read_node_metadata", read)
            obj.value = 2
            current = repo.save(obj)
            assert current.state.key != saved.state.key
            assert captures == [obj]
        else:
            probe = store.save_batch_needs_recovery

            def check(batch):
                if not moves:
                    advance()
                    raise authority.StoreCutChanged("recovery metadata changed")
                return probe(batch)

            monkeypatch.setattr(store, "save_batch_needs_recovery", check)
            obj.value = 3
            current = repo.save(obj)
            assert repo.load_state_ref(current, reuse_live="never").value == 3
            assert captures == [obj]
        assert moves == [True]


def test_selected_capture_reuses_decode_but_rereads_membership(tmp_path, monkeypatch):
    from dryml.core import node_codec
    from dryml.core._content_facts import _content_facts_scope
    from dryml.core.node_prepare import NodePreparation

    repo = Repo(DirStore(tmp_path / "store"))
    try:
        store = repo.default_store
        saved = repo.save(ReportState(5, repo=repo))
        prepared = NodePreparation((saved,))
        digest = prepared.digest(saved, domain="dryml-saved-root/v1")
        decode = node_codec._decode_node_graph_views
        calls = []

        def counted(*args, **kwargs):
            if kwargs.get("with_owner"):
                calls.append(True)
            return decode(*args, **kwargs)

        monkeypatch.setattr(node_codec, "_decode_node_graph_views", counted)
        with _content_facts_scope():
            first = store._read_node_capture_for_selection(saved, prepared)
            second = store._read_node_capture_for_selection(saved, prepared)
            assert first[0] == second[0]
            assert calls == [True]
            store.withdraw_node(digest, kind="saved")
            assert store._read_node_capture_for_selection(saved, prepared) is None
    finally:
        repo.close(flush=False)


def test_single_state_reader_recaptures_metadata_before_payload_and_body(tmp_path, monkeypatch):
    with closing(DirStore(tmp_path / "store")) as store, closing(DirStore.open_existing(tmp_path / "store")) as peer:
        repo = Repo(store)
        try:
            saved = repo.save(ReportState(7, repo=repo))
            read, authenticate = store.read_creation, store._authenticate_state_data
            moved, authenticated, delivered = [], [], []

            def read_and_move(*args, **kwargs):
                result = read(*args, **kwargs)
                if not moved:
                    moved.append(True)
                    peer.publish_node(Definition(ReportState, 100).concretize())
                return result

            def counted_authentication(*args, **kwargs):
                authenticated.append(True)
                return authenticate(*args, **kwargs)

            monkeypatch.setattr(store, "read_creation", read_and_move)
            monkeypatch.setattr(store, "_authenticate_state_data", counted_authentication)
            with store.read_state(saved.state.key) as source:
                delivered.append(source.state.key)
            assert delivered == [saved.state.key]
            assert moved == authenticated == [True]
            with peer.state_leases((saved.state.key,), exclusive=True, blocking=False) as acquired:
                assert acquired
        finally:
            repo.close(flush=False)


@pytest.mark.parametrize("failure, attempts", [
    (authority.StoreCutChanged, 8), (OSError, 1), (KeyboardInterrupt, 1),
])
def test_single_state_reader_preflight_failure_releases_lease_without_payload(
        tmp_path, monkeypatch, failure, attempts):
    with closing(DirStore(tmp_path / "store")) as store:
        repo = Repo(store)
        try:
            saved = repo.save(ReportState(7, repo=repo))
            calls = []
            error = failure("injected state metadata failure")

            def fail(*args, **kwargs):
                calls.append(True)
                raise error

            monkeypatch.setattr(store, "read_creation", fail)
            monkeypatch.setattr(authority, "_pause_cut_retry", lambda attempt: None)
            monkeypatch.setattr(store, "_authenticate_state_data", lambda *a: pytest.fail("failed metadata opened payload"))
            with pytest.raises(failure) as raised:
                with store.read_state(saved.state.key):
                    pytest.fail("failed metadata delivered a source")
            assert raised.value is error and len(calls) == attempts
            with store.state_leases((saved.state.key,), exclusive=True, blocking=False) as acquired:
                assert acquired
        finally:
            repo.close(flush=False)


def test_single_state_reader_does_not_retry_borrower_failure(tmp_path, monkeypatch):
    with closing(DirStore(tmp_path / "store")) as store:
        repo = Repo(store)
        try:
            saved = repo.save(ReportState(7, repo=repo))
            error = authority.StoreCutChanged("borrower changed its own cut")
            monkeypatch.setattr(authority, "_pause_cut_retry", lambda attempt: pytest.fail("borrower failure was retried"))
            with pytest.raises(authority.StoreCutChanged) as raised:
                with store.read_state(saved.state.key):
                    raise error
            assert raised.value is error
            with store.state_leases((saved.state.key,), exclusive=True, blocking=False) as acquired:
                assert acquired
        finally:
            repo.close(flush=False)


@pytest.mark.parametrize("failure, attempts", [
    (authority.StoreCutChanged, 8), (OSError, 1), (KeyboardInterrupt, 1),
])
def test_retention_adoption_read_failure_does_not_write_owner_intent(tmp_path, monkeypatch, failure, attempts):
    from dryml.core.store.retention import RetentionGrantRecord

    with closing(DirStore(tmp_path / "store")) as store:
        repo = Repo(store)
        try:
            _, report = repo.save(ReportState(7, repo=repo), report_stores=True)
            token = report.publications[0].retention_token
            read = store._read_file
            calls = []
            error = failure("injected grant metadata failure")

            def fail_grant(path, record_type):
                if record_type is RetentionGrantRecord:
                    calls.append(True)
                    raise error
                return read(path, record_type)

            monkeypatch.setattr(store, "_read_file", fail_grant)
            monkeypatch.setattr(authority, "_pause_cut_retry", lambda attempt: None)
            monkeypatch.setattr(store.retention, "_write", lambda *a: pytest.fail("failed adoption wrote owner intent"))
            with pytest.raises(failure) as raised:
                store.retention.adopt(token, (store,))
            assert raised.value is error and len(calls) == attempts
        finally:
            repo.close(flush=False)
