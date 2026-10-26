"""U5 direct-record save, structural-load, and future exact-restore contracts."""

import inspect
import gc
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.usefixtures("fixed_snapshot_environment")

from dryml.core import Object, Repo, Serializable
from dryml.core.repo import make_store
from dryml.core.repo import RepoSaveError
from dryml.core.store.dir import DirStore
from dryml.core.store.zip import ZipStore
from dryml.core.node import Node
from dryml.core.node_prepare import NodePreparation


def saved_digest(node):
    return NodePreparation((node,)).digest(node, domain="dryml-saved-root/v1")


class SaveLoadValue(Serializable):
    saves = 0

    def __init__(self, value):
        self.value = value

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        type(self).saves += 1
        Path(dest_dir, "value.txt").write_text(str(self.value), encoding="ascii")


class SaveLoadNode(Object):
    def __init__(self, children):
        self.children = children


class RestoreValue(Serializable):
    def __init__(self, value):
        self.value = value

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        Path(dest_dir, "value.txt").write_text(str(self.value), encoding="ascii")

    def restore_state_from_dir_imp(self, src_dir, *, codec):
        self.value = int(Path(src_dir, "value.txt").read_text(encoding="ascii"))


class StatefulSaveLoadNode(RestoreValue):
    """Stateful parent whose child graph can include inert quoted checkpoints."""

    def __init__(self, children):
        self.children = children
        self.value = 30


class FailingSave(Serializable):
    def save_state_to_dir_imp(self, dest_dir, *, codec):
        Path(dest_dir, "partial.txt").write_text("partial", encoding="ascii")
        raise RuntimeError("serializer failed")


def _run_fresh_process(code: str) -> subprocess.CompletedProcess[str]:
    """Run ``code`` with this checkout's source tree first on ``PYTHONPATH``."""

    src = Path(__file__).resolve().parents[2] / "src"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(src), env.get("PYTHONPATH", "")))
    return subprocess.run(
        (sys.executable, "-c", code),
        text=True,
        capture_output=True,
        env=env,
    )


def _run_main_script(tmp_path: Path, code: str) -> subprocess.CompletedProcess[str]:
    """Run file-backed source as ``__main__`` so class source inspection works."""

    script = tmp_path / "main_namespace.py"
    script.write_text(textwrap.dedent(code), encoding="utf-8")
    return _run_fresh_process(
        f"import runpy; runpy.run_path({str(script)!r}, run_name='__main__')"
    )


@pytest.mark.parametrize("store_name", ("store",), ids=("dir",))
def test_main_module_object_save_load_preserves_definition(tmp_path, store_name):
    """Issue #25: a main-defined Object reloads with its class, recipe, and data."""

    completed = _run_main_script(tmp_path, f"""
        from contextlib import closing
        from dryml.core import Object, Repo
        from dryml.core.store.dir import DirStore
        from dryml.core.symbol import SourceSpec

        class Data(Object):
            def __init__(self, data):
                self.data = data

        assert Data.__module__ == "__main__"
        with closing(Repo(DirStore({str(tmp_path / store_name)!r}))) as repo:
            saved = Data([1, 2, 3, 4, 5], repo=repo)
            definition = saved.definition
            assert isinstance(definition.cls, SourceSpec)
            repo.save_object(saved, main=True)

        with closing(Repo(DirStore({str(tmp_path / store_name)!r}))) as reopened:
            assert reopened.main_def.graph_equal(definition)
            loaded = reopened.load_object(reopened.main_def)
            assert loaded is not saved
            assert type(loaded) is Data
            assert loaded.definition.graph_equal(definition)
            assert loaded.data == [1, 2, 3, 4, 5]
    """)

    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("store_name", ("store",), ids=("dir",))
def test_main_module_state_load_without_live_class_preserves_identity(tmp_path, store_name):
    """A fresh reader restores a main-defined class, exact ObjectId, and payload."""

    saved = _run_main_script(tmp_path, f"""
        from contextlib import closing
        from dryml.core import Repo
        from dryml.core.store.dir import DirStore
        from dryml.core.object import Pickleable
        from tests.core.test_repo_save_load import saved_digest

        class Data(Pickleable):
            def __init__(self, data):
                self.data = data

        assert Data.__module__ == "__main__"
        with closing(Repo(DirStore({str(tmp_path / store_name)!r}))) as repo:
            obj = Data([1, 2, 3, 4, 5], repo=repo)
            obj.result = {{"answer": 42}}
            state = repo.save_object(obj, main=True)
            assert repo.describe(obj).object_projection_equal(state)
            print(saved_digest(state))
    """)
    assert saved.returncode == 0, saved.stderr

    loaded = _run_fresh_process(f"""
import __main__
from contextlib import closing
from dryml.core import Repo
from dryml.core.symbol import SourceSpec
from dryml.core.store.dir import DirStore

assert "Data" not in vars(__main__)
with closing(Repo(DirStore({str(tmp_path / store_name)!r}))) as repo:
    state = repo.default_store.read_node({saved.stdout.strip()!r}, kind="saved")
    assert isinstance(state.cls, SourceSpec)
    assert repo.main_def.graph_equal(state.creation_view())
    restored = repo.load_state_ref(state, reuse_live="never")
    assert type(restored).__name__ == "Data"
    assert restored.definition.graph_equal(state.creation_view())
    assert repo.describe(restored).object_projection_equal(state)
    assert restored.last_state_ref == state
    assert restored.data == [1, 2, 3, 4, 5]
    assert restored.result == {{"answer": 42}}
""")

    assert loaded.returncode == 0, loaded.stderr


def test_public_save_returns_saved_node_and_publishes_local_state(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    obj = SaveLoadValue(10, repo=repo)

    state = obj.save(repo=repo, main=False)

    assert isinstance(state, Node)
    assert dict(state.identity_spec()) == dict(repo.describe(obj).identity_spec())
    assert store.read_node(saved_digest(state), kind="saved").graph_equal(state)
    with store.read_state(state.state.key) as source:
        assert Path(source.data_directory, "value.txt").read_text() == "10"


def test_save_publishes_state_ref_before_main_and_object_alias(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    obj = SaveLoadValue(10, repo=repo)
    calls = []
    original = store._atomic_write
    def record_writes(path, payload):
        if "/graphs/" in str(path).replace("\\", "/"):
            calls.append("state")
        if str(path).replace("\\", "/").endswith("/names/main.record"):
            calls.append("main")
        if str(path).replace("\\", "/").endswith("/names/latest.record"):
            calls.append("alias")
        return original(path, payload)
    monkeypatch.setattr(store, "_atomic_write", record_writes)

    state = repo.save_object(obj, main=True, alias="latest")

    assert calls[0] == "state" and set(calls) == {"state", "main", "alias"}
    assert repo.main_def.graph_equal(obj.definition)
    main = store.read_node_name("main")
    assert store.read_node(main.root_digest, kind="recipe").graph_equal(obj.definition)
    alias = store.read_node_name("latest")
    assert alias.root_kind == "saved" and alias.root_digest == saved_digest(state)


def test_shared_nodes_save_once_and_equal_independent_nodes_stay_distinct(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    SaveLoadValue.saves = 0
    shared = SaveLoadValue("same", repo=repo)

    shared_state = SaveLoadNode([shared, shared], repo=repo).save(repo=repo, main=False)

    assert SaveLoadValue.saves == 1
    assert len(shared_state.state_spec()) == 1

    SaveLoadValue.saves = 0
    first = SaveLoadValue("same", repo=repo)
    second = SaveLoadValue("same", repo=repo)
    independent_state = SaveLoadNode([first, second], repo=repo).save(repo=repo, main=False)

    assert SaveLoadValue.saves == 2
    assert len(independent_state.state_spec()) == 2


def test_explicit_store_save_copies_reusable_state_into_its_closure(tmp_path):
    source = DirStore(tmp_path / "source")
    copied = DirStore(tmp_path / "copied")
    repo = Repo([source, copied])
    child = SaveLoadValue("child", repo=repo)
    child_state = repo.save_object(child, store=source)
    root = SaveLoadNode(child, repo=repo)

    child_path = next(path for path, object_id in repo.describe(root).identity_spec().items() if object_id == child.object_id)
    copied_state, copied_report = repo.save_object(root, states={child_path: child_state.state}, store=copied, report_stores=True)
    copied_path = next(path for path, object_id in copied_state.identity_spec().items() if object_id == child.object_id)
    assert copied_report.required_stores == (copied,)
    with copied.read_state(copied_state.at(copied_path).state.key) as borrowed:
        assert Path(borrowed.data_directory, "value.txt").read_text() == "child"


@pytest.mark.parametrize("query_index", ("sqlite", "none"))
@pytest.mark.parametrize("owned", (False, True), ids=("quote-only", "owned-and-quoted"))
def test_save_quoted_checkpoint_preserves_exact_data_and_canonical_declarations(
    tmp_path, query_index, owned,
):
    """Quoted state is inert; owned declarations agree across Repo and direct Store."""
    source = DirStore(tmp_path / "source", query_index="none")
    destination = DirStore(tmp_path / "destination", query_index=query_index)
    direct = DirStore(tmp_path / "direct", query_index=query_index)
    writer = Repo([source, destination])
    donor = RestoreValue(10, repo=writer)
    quoted = donor.save(repo=writer, store=source)
    donor.value = 20
    parent = StatefulSaveLoadNode({"owned": donor if owned else None, "quoted": quoted.quote()}, repo=writer)

    saved = parent.save(repo=writer, store=destination)

    saved_quote = saved.parameters["children"]["quoted"].value
    assert saved_quote.graph_equal(quoted)
    assert saved_quote.object_id == quoted.object_id
    assert saved_quote.state == quoted.state
    assert saved.object_view().parameters["children"]["quoted"].value.graph_equal(quoted)
    assert len(saved.state_spec()) == 1 + int(owned)
    assert not Path(destination.state_data_path(quoted.state.key)).exists()
    if owned:
        selected = saved.parameters["children"]["owned"]
        assert selected.object_id == quoted.object_id
        assert selected.state.key != quoted.state.key
    else:
        assert destination.read_declaration_anchor(quoted.object_id) is None

    direct_record = direct.publish_node(saved, kind="saved")
    assert direct.read_node(direct_record.digest, kind="saved").graph_equal(saved)
    identities = (saved.object_id, quoted.object_id) if owned else (saved.object_id,)
    for identity in identities:
        batch_anchor = destination.read_declaration_anchor(identity)
        direct_anchor = direct.read_declaration_anchor(identity)
        batch_projection = destination._expand_node_root(batch_anchor.projection)
        direct_projection = direct._expand_node_root(direct_anchor.projection)
        assert batch_projection.graph_equal(batch_projection.object_view())
        assert direct_projection.graph_equal(batch_projection)
        assert batch_projection.state is None
    if not owned:
        assert direct.read_declaration_anchor(quoted.object_id) is None
    writer.close(flush=False)
    direct.close()

    reopened = DirStore.open_existing(tmp_path / "destination", query_index="sqlite")
    reader = Repo(reopened)
    try:
        index = reopened.open_query_index()
        index.rebuild()
        assert index.status().state == "ready"
        restored = reopened.read_node(saved_digest(saved), kind="saved")
        assert restored.graph_equal(saved)
        loaded = reader.load_state_ref(restored, reuse_live="never")
        assert loaded.value == 30
        assert loaded.children["quoted"].value.graph_equal(quoted)
        if owned:
            assert loaded.children["owned"].value == 20
        else:
            assert loaded.children["owned"] is None
    finally:
        reader.close(flush=False)


def test_failed_local_state_save_never_publishes_a_state_ref(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    obj = FailingSave(repo=repo)

    with pytest.raises(RepoSaveError, match="snapshot-local state preparation"):
        repo.save_object(obj)

    assert obj._last_state_hash is None
    assert not (tmp_path / "store" / "snapshots").exists()
    staging = tmp_path / "store" / ".staging"
    assert not staging.exists() or {p.name for p in staging.iterdir()} <= {".leases"}
    assert not store._staging_leases


def test_orchestrator_guard_rejects_live_save_before_publication(tmp_path):
    store_path = tmp_path / "store"
    completed = _run_fresh_process(
        f"""
from pathlib import Path

from dryml import session
from dryml.core import Repo
from dryml.core.store.dir import DirStore
from dryml.runtime.errors import RuntimeTransitionError
from tests.core.test_repo_save_load import SaveLoadValue


store_path = Path({str(store_path)!r})
repo = Repo(DirStore(store_path))
obj = SaveLoadValue(10, repo=repo)
session.set_mode("orchestrator")
try:
    try:
        repo.save_object(obj)
    except RuntimeTransitionError as error:
        assert "prohibits Object materialization" in str(error)
    else:
        raise AssertionError("orchestrator save unexpectedly materialized an Object")
finally:
    session.reset()

assert not (store_path / "graphs").exists()
"""
    )

    assert completed.returncode == 0, completed.stderr


def test_changed_save_surface_rejects_retired_revision_options_and_generation_keywords():
    for callable_ in (Repo.save_object, Repo.save, SaveLoadValue.save):
        names = set(inspect.signature(callable_).parameters)
        assert not {"revision", "options", "generation", "ephemeral_depth"} & names

    repo = Repo()
    for keyword, value in (("revision", "retired"), ("options", {}), ("generation", 1)):
        with pytest.raises(TypeError):
            repo.save_object(SaveLoadValue(1), **{keyword: value})


def test_deletion_save_uses_strong_cache_snapshot_and_unraisable_cleanup(tmp_path, monkeypatch):
    """Deletion saves stop at the first failure but still perform non-flushing cleanup."""

    repo = Repo(tmp_path / "store.zip")
    first = SaveLoadValue("first", repo=repo)
    failing = SaveLoadValue("failing", repo=repo)
    later = SaveLoadValue("later", repo=repo)
    for obj in (first, failing, later):
        repo.cache_strong(obj)
    saved = []
    closes = []
    unraisable = []

    def save(obj, **kwargs):
        saved.append(obj.value)
        if obj.value == "failing":
            raise RuntimeError("implicit save failed")

    original_close = repo.close

    def close(*, flush=True, _close=original_close):
        closes.append(flush)
        return _close(flush=flush)

    repo.save = save
    repo.close = close
    monkeypatch.setattr(sys, "unraisablehook", lambda args: unraisable.append(args))

    repo.save_objs_on_deletion = True
    del first, failing, later, original_close, close
    del repo
    gc.collect()

    assert saved == ["first", "failing"]
    assert closes == [False]
    assert len(unraisable) == 1
    assert unraisable[0].exc_type is RuntimeError
    assert str(unraisable[0].exc_value) == "Repo deletion cleanup failed."


def test_deletion_save_reports_sanitized_failure_to_default_unraisable_hook():
    """The interpreter's real unraisable hook receives no save exception details."""

    completed = _run_fresh_process(
        """
import gc

from dryml.core import Repo, Serializable


repo = Repo()
repo.save_objs_on_deletion = True
repo.cache_strong(Serializable(repo=repo))
def save_failure(obj):
    raise RuntimeError("save secret: top-secret")


repo.save = save_failure
del repo
gc.collect()
"""
    )

    assert completed.returncode == 0
    assert "Repo deletion cleanup failed" in completed.stderr
    assert "top-secret" not in completed.stderr
    assert "save_failure" not in completed.stderr
    assert "Repo object at" not in completed.stderr


def test_partially_constructed_repo_deletion_is_inert():
    """Destructor cleanup tolerates constructor failure before Repo state exists."""

    partial = Repo.__new__(Repo)
    partial.__del__()


def test_nonflushing_close_never_runs_deletion_saves(tmp_path, monkeypatch):
    """Worker-style non-flushing cleanup cannot publish a deletion-save snapshot."""

    repo = Repo(tmp_path / "store.zip")
    obj = SaveLoadValue("cached", repo=repo)
    repo.cache_strong(obj)
    repo.save_objs_on_deletion = True
    monkeypatch.setattr(repo, "save", lambda *args, **kwargs: pytest.fail("unexpected save"))

    repo.close(flush=False)


def test_make_store_accepts_a_delegating_binary_file_wrapper():
    with tempfile.NamedTemporaryFile() as wrapped:
        store = make_store(wrapped)
        try:
            assert isinstance(store, ZipStore)
        finally:
            store.close()


def test_direct_layout_has_no_retired_object_or_generation_paths(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    state = repo.save_object(SaveLoadValue(10, repo=repo))

    paths = {path.name for path in Path(store.base_dir).rglob("*")}

    assert store.read_node(saved_digest(state), kind="saved").graph_equal(state)
    assert "objects" not in paths
    assert ".state-generations" not in paths
    assert ".state-current.pkl" not in paths
    assert "generation" not in " ".join(str(path) for path in Path(store.base_dir).rglob("*"))


def test_explicit_build_uses_construction_defaults_instead_of_saved_state(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    saved = RestoreValue(10, repo=repo)
    saved.value = 20
    repo.save_object(saved)

    loaded = saved.definition.build(repo=Repo(DirStore(store.base_dir)), cache="none")

    assert isinstance(loaded, RestoreValue)
    assert loaded.value == 10


def test_reopen_v3_query_discovers_definition_authority_without_materializing(tmp_path):
    store = DirStore(tmp_path / "store", query_index="memory")
    repo = Repo(store)
    first = SaveLoadValue("first", repo=repo)
    second = SaveLoadValue("second", repo=repo)
    repo.save_object(first)
    repo.save_object(second)

    reopened = Repo(DirStore(store.base_dir, query_index="memory"))

    found = tuple(reopened.query().cdefs().stored().collect())
    assert len(found) == 2
    assert all(any(node.graph_equal(expected) for node in found) for expected in (first.definition, second.definition))
    assert reopened._num_constructions == 0
    assert reopened.default_store.query().cdefs().stored().count() == 2


def test_reopening_main_definition_is_read_only_hydration(tmp_path):
    source = DirStore(tmp_path / "source")
    target = DirStore(tmp_path / "target")
    from dryml.core.store.records import NodeNameRecord
    definition = SaveLoadValue(3).definition
    record = source.publish_node(definition)
    source.compare_and_set_node_name(NodeNameRecord("main", "recipe", record.digest), expected=None)
    before = {path.name: path.read_bytes() for path in (Path(target.base_dir) / "store-format.record", Path(target.base_dir) / "authority.record")}

    repo = Repo([target, source])

    assert repo.main_def.graph_equal(definition)
    assert target.read_node_name("main") is None
    assert target.read_node(record.digest) is None
    assert {name: (Path(target.base_dir) / name).read_bytes() for name in before} == before
    assert not (Path(target.base_dir) / "membership").exists()


def test_u7_exact_state_ref_restores_published_local_state(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    saved = RestoreValue(10, repo=repo)
    saved.value = 20
    state = repo.save_object(saved)

    loaded = Repo(DirStore(store.base_dir)).load_state_ref(state, reuse_live="never")

    assert loaded.value == 20


def test_u7_exact_state_ref_restores_nested_independent_local_states(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    first = RestoreValue(1, repo=repo)
    second = RestoreValue(2, repo=repo)
    first.value = 10
    second.value = 20
    state = repo.save_object(SaveLoadNode([SaveLoadNode(first, repo=repo), second], repo=repo))

    loaded = Repo(DirStore(store.base_dir)).load_state_ref(state, reuse_live="never")

    assert loaded.children[0].children.value == 10
    assert loaded.children[1].value == 20


def test_u7_exact_state_ref_restores_shared_nodes_and_reuses_the_exact_graph(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    child = RestoreValue(10, repo=repo)
    child.value = 20
    state = repo.save_object(SaveLoadNode([child, child], repo=repo))
    reopened = Repo(DirStore(store.base_dir))

    first = reopened.load_state_ref(state)
    first.children[0].value = 99
    second = reopened.load_state_ref(state)

    assert first.children[0] is first.children[1]
    assert first.children[0].value == 20
    assert second is first
    assert second.children[0] is first.children[0]


def test_real_borrowed_reads_share_budget_but_not_authority_or_public_values(tmp_path):
    """Node readers share the owner's budget, never its authority or public values.

    Snapshot-JSON retention was removed with Store 3. Current Node readers do not
    admit those facts: repeated/borrowed reads authenticate fresh full-key bytes,
    and no record, payload or resource leaks into that retired cache facility.
    """
    from dryml.core import SaveAnnotations
    from dryml.core.materialization import build_exact_state_load_plan
    from tests.core.test_exact_load_metadata_evidence import exact_work_counts

    store = DirStore(tmp_path / "store", query_index="none")
    writer = Repo(store)
    state = writer.save_object(RestoreValue(3, repo=writer), annotations=SaveAnnotations(
        state={"nested": [{"values": [1, 2]}]},
    ))
    owner = Repo(store)
    facility = owner._validation_evidence
    facility.max_entries = 6
    borrower = Repo._for_state_io((store,), _content_owner=owner)
    try:
        assert borrower._validation_evidence is facility
        with exact_work_counts() as cold:
            build_exact_state_load_plan(owner, state)
        with exact_work_counts() as nested:
            build_exact_state_load_plan(borrower, state)
        assert nested["validate_payload"] == cold["validate_payload"] == 2  # wrapper and inventory authenticator
        first = owner.get_snapshot_metadata(state)
        second = borrower.get_snapshot_metadata(state)
        assert first is not second
        second.captured_state_annotations["nested"][0]["values"].append(99)
        assert borrower.get_snapshot_metadata(state).captured_state_annotations["nested"] == [{"values": [1, 2]}]
        assert not facility._pending and not facility._resident and facility.accounted_bytes == 0
        owner.clear_cache(strong=False, weak=False)
        assert facility.accounted_bytes == 0
        with exact_work_counts() as cleared:
            build_exact_state_load_plan(borrower, state)
        assert cleared["validate_payload"] == 2
        independent = Repo(store)
        try:
            with exact_work_counts() as isolated:
                build_exact_state_load_plan(independent, state)
            assert isolated["validate_payload"] == 2
            assert independent._validation_evidence is not facility
        finally:
            independent.close(flush=False)
        retained = tuple(facility._resident.values())
        borrower.close(flush=False)
        assert tuple(facility._resident.values()) == retained
        assert facility.high_water_entries == 0
        assert facility.high_water_bytes <= facility.max_bytes
        assert owner._topology_leases == 0
    finally:
        borrower.close(flush=False)
        owner.close(flush=False)
        writer.close(flush=False)
    assert not facility._resident and facility.accounted_bytes == 0
