"""Independent writers retain exact read-set CAS without whole-Store starvation."""

from contextlib import closing

import pytest

from dryml.core import Definition, Object, Repo
from dryml.core.store.dir import DirStore
from dryml.core.store import authority
from dryml.core.store.records import MutationIntentRecord, NodeNameRecord


class IndependentValue(Object):
    """Stateless graph used to exercise metadata publication only."""

    def __init__(self, value):
        self.value = value


def test_private_preparation_survives_independent_writes_at_each_boundary(tmp_path, monkeypatch):
    with closing(DirStore(tmp_path / "store")) as store, closing(DirStore.open_existing(tmp_path / "store")) as peer:
        root = store.publish_node(Definition(IndependentValue, "target").concretize())
        target = NodeNameRecord("target", "recipe", root.digest)
        path = store._name_path("target")
        prepare_calls, apply_calls = [], []
        apply = authority.apply

        def independent_write(index):
            peer.compare_and_set_node_name(NodeNameRecord("peer-" + str(index), "recipe", root.digest), expected=None)

        def prepare():
            prepare_calls.append(True)
            change = store._prepared_file(path, target)
            independent_write("prepare-" + str(len(prepare_calls)))
            return [change], ("name",), target

        def apply_after_peer(selected, facts, **kwargs):
            if selected is store:
                apply_calls.append(True)
                independent_write("apply-" + str(len(apply_calls)))
                before = store.authority_cut()
                result = apply(selected, facts, **kwargs)
                record = store._read_file(authority._path(store, result[0]), MutationIntentRecord)
                assert record.intent["base_epoch"] == before.epoch > facts.cut.epoch
                assert result[1].epoch == before.epoch + 1
                return result
            return apply(selected, facts, **kwargs)

        monkeypatch.setattr(authority, "apply", apply_after_peer)
        monkeypatch.setattr(authority, "_pause_cut_retry", lambda attempt: None)
        assert store._prepare_and_apply(prepare) == target
        assert prepare_calls == apply_calls == [True]
        assert store.read_node_name("target") == target
        assert not store.authority_cut().pending
        assert store.query().cdefs().alias("target").require_indexed().count() == 1


def test_rebased_read_rejects_changed_dependency_and_strict_reads_keep_cut(tmp_path):
    with closing(DirStore(tmp_path / "store")) as store, closing(DirStore.open_existing(tmp_path / "store")) as peer:
        first = store.publish_node(Definition(IndependentValue, "first").concretize())
        second = store.publish_node(Definition(IndependentValue, "second").concretize())
        original = NodeNameRecord("target", "recipe", first.digest)
        replacement = NodeNameRecord("target", "recipe", second.digest)
        store.compare_and_set_node_name(original, expected=None)
        with pytest.raises(authority.StoreCutChanged):
            with store._metadata_read(rebase=True):
                assert store.read_node_name("target") == original
                peer.compare_and_set_node_name(replacement, expected=original)
        assert store.read_node_name("target") == replacement
        with pytest.raises(authority.StoreCutChanged):
            with store._metadata_read():
                assert store.read_node_name("target") == replacement
                peer.compare_and_set_node_name(NodeNameRecord("other", "recipe", first.digest), expected=None)


def test_rebased_read_cannot_forget_explicit_epoch_or_directory_dependencies(tmp_path):
    with closing(DirStore(tmp_path / "store")) as store, closing(DirStore.open_existing(tmp_path / "store")) as peer:
        root = store.publish_node(Definition(IndependentValue, "target").concretize())
        for dependency in ("epoch", "directory"):
            with pytest.raises(authority.StoreCutChanged):
                with store._metadata_read(rebase=True) as read:
                    if dependency == "epoch":
                        store.authority_cut()
                    else:
                        read.directory("names")
                    peer.compare_and_set_node_name(NodeNameRecord(dependency, "recipe", root.digest), expected=None)


def test_public_prepared_mutation_still_rejects_independent_epoch_drift(tmp_path):
    with closing(DirStore(tmp_path / "store")) as store, closing(DirStore.open_existing(tmp_path / "store")) as peer:
        root = store.publish_node(Definition(IndependentValue, "target").concretize())
        record = NodeNameRecord("target", "recipe", root.digest)
        with store._metadata_read() as read:
            facts = read.batch([store._prepared_file(store._name_path("target"), record)], ("name",))
        peer.compare_and_set_node_name(NodeNameRecord("other", "recipe", root.digest), expected=None)
        with pytest.raises(authority.StoreCutChanged):
            store.apply_prepared_mutation(facts)
        assert store.read_node_name("target") is None


def test_checkpoint_metadata_validation_keeps_progress_with_independent_writers(tmp_path, monkeypatch, fixed_snapshot_environment):
    from dryml.core.materialization import build_exact_state_load_plan
    from tests.core.test_store_report import ReportState

    repo = Repo(DirStore(tmp_path / "store"))
    peer = DirStore.open_existing(repo.default_store.base_dir)
    try:
        store = repo.default_store
        saved = repo.save(ReportState(7, repo=repo))
        root = peer.publish_node(Definition(IndependentValue, "peer").concretize())
        captures, lineages = [], []
        capture, metadata = store._read_node_capture_for_selection, store._read_node_metadata

        def advance(label):
            peer.compare_and_set_node_name(NodeNameRecord(label, "recipe", root.digest), expected=None)

        def capture_and_move(*args, **kwargs):
            result = capture(*args, **kwargs)
            captures.append(True)
            advance("capture-" + str(len(captures)))
            return result

        def lineage_and_move(*args, **kwargs):
            result = metadata(*args, **kwargs)
            lineages.append(True)
            advance("lineage-" + str(len(lineages)))
            return result

        monkeypatch.setattr(store, "_read_node_capture_for_selection", capture_and_move)
        monkeypatch.setattr(store, "_read_node_metadata", lineage_and_move)
        monkeypatch.setattr(authority, "_pause_cut_retry", lambda attempt: pytest.fail("independent metadata write retried validation"))
        plan = build_exact_state_load_plan(repo, saved)
        assert plan is not None and captures == lineages == [True]
    finally:
        peer.close()
        repo.close(flush=False)


@pytest.mark.parametrize("interference", ("independent", "dependency-once", "dependency-repeated"))
def test_prior_checkpoint_preflight_validates_dependencies_without_recapturing(
        tmp_path, monkeypatch, fixed_snapshot_environment, interference):
    from tests.core.test_store_report import ReportState

    repo = Repo(DirStore(tmp_path / "store"))
    peer = DirStore.open_existing(repo.default_store.base_dir)
    try:
        store = repo.default_store
        obj = ReportState(7, repo=repo)
        prior = repo.save(obj)
        root = peer.publish_node(Definition(IndependentValue, "peer").concretize())
        metadata, capture, publish = store._read_node_metadata, ReportState.save_state_to_dir_imp, store.publish_save_batch
        reads, captures, publications, pauses = [], [], [], []

        def metadata_and_move(target, scope=None):
            result = metadata(target, scope)
            if scope == "capture" and target.graph_equal(prior):
                reads.append(True)
                if interference == "independent":
                    peer.compare_and_set_node_name(NodeNameRecord("peer-" + str(len(reads)), "recipe", root.digest), expected=None)
                elif interference == "dependency-repeated" or len(reads) == 1:
                    peer.publish_node(Definition(IndependentValue, (prior, len(reads))).concretize())
            return result

        def counted_capture(self, *args, **kwargs):
            captures.append(self)
            return capture(self, *args, **kwargs)

        def counted_publish(batch):
            publications.append(batch)
            return publish(batch)

        monkeypatch.setattr(store, "_read_node_metadata", metadata_and_move)
        monkeypatch.setattr(ReportState, "save_state_to_dir_imp", counted_capture)
        monkeypatch.setattr(store, "publish_save_batch", counted_publish)
        monkeypatch.setattr(authority, "_pause_cut_retry", pauses.append)
        obj.value = 8
        if interference == "dependency-repeated":
            with pytest.raises(authority.StoreCutChanged, match="byte witness changed"):
                repo.save(obj)
            assert len(reads) == 8 and pauses == list(range(7))
            assert not captures and not publications and obj.last_state_ref is prior
        else:
            current = repo.save(obj)
            assert current is obj.last_state_ref and current.state.key != prior.state.key
            assert captures == [obj] and len(publications) == 1
            assert len(reads) == (1 if interference == "independent" else 2)
            assert pauses == ([] if interference == "independent" else [0])
        monkeypatch.setattr(store, "_read_node_metadata", metadata)
        assert repo.load_state_ref(prior, reuse_live="never").value == 7
    finally:
        peer.close()
        repo.close(flush=False)


@pytest.mark.parametrize("boundary", ("retention-owner", "recovery-probe"))
def test_save_preflight_keeps_progress_with_independent_writers(
        tmp_path, monkeypatch, fixed_snapshot_environment, boundary):
    from dryml.core.store.retention import RetentionIntentRecord
    from tests.core.test_store_report import ReportState

    repo = Repo(DirStore(tmp_path / "store"))
    peer = DirStore.open_existing(repo.default_store.base_dir)
    try:
        store = repo.default_store
        root = peer.publish_node(Definition(IndependentValue, "peer").concretize())
        reads, captures, publications = [], [], []
        capture, publish = ReportState.save_state_to_dir_imp, store.publish_save_batch

        def advance():
            reads.append(True)
            peer.compare_and_set_node_name(NodeNameRecord("peer-" + str(len(reads)), "recipe", root.digest), expected=None)

        if boundary == "retention-owner":
            reader = store._read_file

            def read(path, record_type):
                result = reader(path, record_type)
                if record_type is RetentionIntentRecord:
                    advance()
                return result

            monkeypatch.setattr(store, "_read_file", read)
        else:
            probe = store.save_batch_needs_recovery

            def check(batch):
                result = probe(batch)
                advance()
                return result

            monkeypatch.setattr(store, "save_batch_needs_recovery", check)

        def counted_capture(self, *args, **kwargs):
            captures.append(self)
            return capture(self, *args, **kwargs)

        def counted_publish(batch):
            publications.append(batch)
            return publish(batch)

        monkeypatch.setattr(ReportState, "save_state_to_dir_imp", counted_capture)
        monkeypatch.setattr(store, "publish_save_batch", counted_publish)
        monkeypatch.setattr(authority, "_pause_cut_retry", lambda attempt: pytest.fail("independent save read retried"))
        obj = ReportState(9, repo=repo)
        saved, report = repo.save(obj, report_stores=True)
        assert reads and captures == [obj] and len(publications) == 1
        assert saved is obj.last_state_ref and report.required_stores == (store,)
    finally:
        peer.close()
        repo.close(flush=False)


def test_internal_publication_does_not_overwrite_a_changed_cas_dependency(tmp_path, monkeypatch):
    with closing(DirStore(tmp_path / "store")) as store, closing(DirStore.open_existing(tmp_path / "store")) as peer:
        root = store.publish_node(Definition(IndependentValue, "target").concretize())
        winner = NodeNameRecord("target", "recipe", root.digest)
        calls = []

        def prepare():
            calls.append(True)
            if store.read_node_name("target") is not None:
                raise ValueError("name CAS lost")
            change = store._prepared_file(store._name_path("target"), winner)
            peer.compare_and_set_node_name(winner, expected=None)
            return [change], ("names",), None

        monkeypatch.setattr(authority, "_pause_cut_retry", lambda attempt: None)
        with pytest.raises(ValueError, match="name CAS lost"):
            store._prepare_and_apply(prepare)
        assert calls == [True, True] and store.read_node_name("target") == winner


def test_reused_facts_keep_all_dependencies_and_nested_reads_cannot_relax_cut(tmp_path):
    with closing(DirStore(tmp_path / "store")) as store, closing(DirStore.open_existing(tmp_path / "store")) as peer:
        roots = [store.publish_node(Definition(IndependentValue, value).concretize()) for value in (1, 2)]
        first, second = [NodeNameRecord("target", "recipe", root.digest) for root in roots]
        store.compare_and_set_node_name(first, expected=None)
        with store._metadata_read() as source:
            assert store.read_node_name("target") == first
        with pytest.raises(authority.StoreCutChanged):
            with store._metadata_read(rebase=True) as capture:
                with store.authority_read_fence():
                    source.validate()
                capture.include_witnesses(source)
                peer.compare_and_set_node_name(second, expected=first)
        with pytest.raises(authority.StoreCutChanged):
            with store._metadata_read():
                with store._metadata_read(rebase=True):
                    peer.compare_and_set_node_name(NodeNameRecord("other", "recipe", roots[0].digest), expected=None)


@pytest.mark.parametrize("failure", (OSError, KeyboardInterrupt))
def test_rebased_publication_preserves_partial_receipts_without_replaying_work(tmp_path, monkeypatch, failure):
    import dryml.filesystem as filesystem

    with closing(DirStore(tmp_path / "store")) as store, closing(DirStore.open_existing(tmp_path / "store")) as peer:
        root = store.publish_node(Definition(IndependentValue, "target").concretize())
        record = NodeNameRecord("target", "recipe", root.digest)
        calls, base_epochs = [], []
        publish, apply = filesystem.publish_file, authority.apply

        def prepare():
            calls.append(True)
            return [store._prepared_file(store._name_path("target"), record)], ("names",), None

        def moved_apply(selected, facts, **kwargs):
            if selected is store:
                peer.compare_and_set_node_name(NodeNameRecord("peer", "recipe", root.digest), expected=None)
                base_epochs.append(store.authority_cut().epoch)
            return apply(selected, facts, **kwargs)

        def failed_publish(source, target, **kwargs):
            result = publish(source, target, **kwargs)
            if str(target) == store._name_path("target"):
                raise failure("after target became visible")
            return result

        with monkeypatch.context() as patch:
            patch.setattr(authority, "apply", moved_apply)
            patch.setattr(filesystem, "publish_file", failed_publish)
            with pytest.raises(authority.StoreMutationError if failure is OSError else failure) as raised:
                store._prepare_and_apply(prepare)
        assert calls == [True] and len(base_epochs) == 1
        receipt = store._read_file(authority._path(store, raised.value.mutation_id), MutationIntentRecord)
        assert receipt.intent["base_epoch"] == base_epochs[0]
        assert "names/target.record" in raised.value.confirmed_refs
        store.recover_authority()
        store.open_query_index().maintain_store()
        assert store.read_node_name("target") == record and not store.authority_cut().pending
