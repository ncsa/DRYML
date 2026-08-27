import os
from pathlib import Path
import subprocess
import sys
import tempfile

import pytest


def _module_importable(module_name: str) -> bool:
    result = subprocess.run(
        [sys.executable, "-c", f"import {module_name}"],
        text=True,
        capture_output=True,
    )
    return result.returncode == 0


def _run_import_probe(code: str):
    src_dir = Path(__file__).resolve().parents[2] / "src"
    env = os.environ.copy()
    pythonpath = [str(src_dir)]
    pythonpath.extend(str(p) for p in sys.path if p)
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    return subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )


def test_dryml_tf_import_installs_adapters_without_importing_tensorflow():
    _run_import_probe(
        """
import sys

assert "tensorflow" not in sys.modules
import dryml.tf
from dryml.core.tensor_spec import TensorSpec

spec = TensorSpec("float32", shape=(32,))
assert hasattr(spec, "tf")
assert "tensorflow" not in sys.modules
        """
    )


def test_dryml_torch_import_installs_adapters_without_importing_torch():
    _run_import_probe(
        """
import sys

assert "torch" not in sys.modules
import dryml.torch
from dryml.core.tensor_spec import TensorSpec
from dryml.torch import TorchTensorSpec

spec = TensorSpec("float32", shape=(32,))
assert hasattr(spec, "torch")
torch_spec = TorchTensorSpec(shape=(32,), dtype="torch.float32")
assert torch_spec.layout == "torch.strided"
assert "torch" not in sys.modules
        """
    )


def test_v2_semantic_access_does_not_import_or_execute_symbolic_classes():
    _run_import_probe(
        """
import sys

assert "tensorflow" not in sys.modules
assert "torch" not in sys.modules

from dryml.core.bound_args import BoundArguments
from dryml.core.definition import ConcreteDefinition, Definition, SKIP_ARGS
from dryml.core.freeze import FrozenTuple
from dryml.core.symbol import ImportRef, SourceSpec
from dryml.core.utils.general import get_unique_concrete_definitions

tf_cdef = ConcreteDefinition._from_bound_record(
    ImportRef("dryml.models.tf.keras.base", "Sequential"),
    BoundArguments((("layer_defs", FrozenTuple(())),)),
)
source_cdef = ConcreteDefinition._from_bound_record(
    SourceSpec.from_source(
        "raise AssertionError('semantic inspection executed source')",
        kind="class",
        name="Danger",
    ),
    BoundArguments((("value", "safe"),)),
)
selector = Definition(ImportRef("dryml.models.torch.base", "Model"), SKIP_ARGS, name="safe")
ImportRef.resolve = lambda self: (_ for _ in ()).throw(AssertionError("graph inspection resolved class"))

assert tf_cdef.layer_defs == FrozenTuple(())
assert tf_cdef.parameters["layer_defs"] == FrozenTuple(())
assert get_unique_concrete_definitions(tf_cdef) == {tf_cdef}
assert source_cdef.value == "safe"
assert source_cdef.parameters["value"] == "safe"
assert selector.name == "safe"
assert selector.parameters["name"] == "safe"
assert "tensorflow" not in sys.modules
assert "torch" not in sys.modules
        """
    )


def test_symbolic_store_hydration_does_not_import_tensorflow():
    _run_import_probe(
        """
import sys
import tempfile
from pathlib import Path

assert "tensorflow" not in sys.modules
from dryml.core.definition import ConcreteDefinition
from dryml.core.freeze import FrozenDict, FrozenTuple
from dryml.core.store.dir import DirStore
from dryml.core.symbol import ImportRef
from dryml.core.utils.general import pickle_save

with tempfile.TemporaryDirectory() as tmp:
    store = DirStore(Path(tmp) / "store", query_index="memory")
    cdef = ConcreteDefinition._from_persisted_record(
        ImportRef("dryml.models.tf.keras.base", "Sequential"),
        FrozenTuple(()),
        FrozenDict({"name": "symbolic"}),
    )
    path = Path(store.object_dir(cdef)) / "def.pkl"
    path.parent.mkdir(parents=True)
    pickle_save(cdef, path)
    assert tuple(store.hydrate_index()) == (cdef,)
    assert store.read_definition(cdef) == cdef

assert "tensorflow" not in sys.modules
        """
    )


def test_query_indexing_canonical_refs_does_not_import_backends():
    _run_import_probe(
        """
import sys

assert "tensorflow" not in sys.modules
assert "torch" not in sys.modules

from dryml.core import Definition, Repo
from dryml.core.definition import ConcreteDefinition
from dryml.core.freeze import FrozenDict, FrozenTuple
from dryml.core.query.fingerprint import legacy_target_fingerprint
from dryml.core.symbol import ImportRef

cdef = ConcreteDefinition._from_persisted_record(
    ImportRef("dryml.models.tf.keras.base", "Sequential"),
    FrozenTuple(()),
    FrozenDict({"layer_defs": FrozenTuple(())}),
)

repo = Repo()
repo._query_catalog.register_cached(cdef)
legacy_target_fingerprint(cdef)
assert repo._query_catalog.cdef_id(cdef) is not None
assert repo.query(cdef).cached().count() == 0
repo.query(cdef).cached().explain()

assert "tensorflow" not in sys.modules
assert "torch" not in sys.modules
        """
    )


def test_exact_query_with_candidate_does_not_import_tensorflow():
    _run_import_probe(
        """
import sys

assert "tensorflow" not in sys.modules
from dryml.core import Definition, Repo, SKIP_ARGS
from dryml.core.definition import ConcreteDefinition
from dryml.core.freeze import FrozenDict, FrozenTuple
from dryml.core.symbol import ImportRef

class FakeStore:
    def catalog_key(self):
        return "fake-tf-store"

cdef = ConcreteDefinition._from_persisted_record(
    ImportRef("dryml.models.tf.keras.base", "Sequential"),
    FrozenTuple(()),
    FrozenDict({"layer_defs": FrozenTuple(())}),
)
repo = Repo()
repo._query_catalog.register_stored(cdef, FakeStore())

assert repo.query(cdef).class_match("exact").stored(refresh=False).count() == 1
selector = Definition(ImportRef("dryml.models.tf.keras.base", "Sequential"), SKIP_ARGS)
assert repo.query(selector).class_match("exact").stored(refresh=False).count() == 1
repo.query(selector).class_match("exact").stored(refresh=False).explain()

assert "tensorflow" not in sys.modules
assert "torch" not in sys.modules
        """
    )


def test_exact_query_with_candidate_does_not_import_torch():
    _run_import_probe(
        """
import sys

assert "torch" not in sys.modules
from dryml.core import Definition, Repo, SKIP_ARGS
from dryml.core.definition import ConcreteDefinition
from dryml.core.freeze import FrozenDict, FrozenTuple
from dryml.core.symbol import ImportRef

class FakeStore:
    def catalog_key(self):
        return "fake-torch-store"

cdef = ConcreteDefinition._from_persisted_record(
    ImportRef("dryml.models.torch.base", "Sequential"),
    FrozenTuple(()),
    FrozenDict({"layer_defs": FrozenTuple(())}),
)
repo = Repo()
repo._query_catalog.register_stored(cdef, FakeStore())

selector = Definition(ImportRef("dryml.models.torch.base", "Sequential"), SKIP_ARGS)
assert repo.query(cdef).class_match("exact").stored(refresh=False).count() == 1
assert repo.query(selector).class_match("exact").stored(refresh=False).count() == 1

assert "tensorflow" not in sys.modules
assert "torch" not in sys.modules
        """
    )


def test_unresolved_semantic_selector_query_is_import_free_and_rejects_positional_binding():
    _run_import_probe(
        """
import sys

assert "tensorflow" not in sys.modules
from dryml.core import Definition, Repo, SKIP_ARGS
from dryml.core.bound_args import BoundArguments
from dryml.core.definition import ConcreteDefinition
from dryml.core.symbol import ImportRef

class FakeStore:
    def catalog_key(self):
        return "unresolved-v2-store"

cls = ImportRef("dryml.models.tf.keras.base", "Sequential")
cdef = ConcreteDefinition._from_bound_record(cls, BoundArguments((("name", "safe"),)))
repo = Repo()
repo._query_catalog.register_stored(cdef, FakeStore())

selector = Definition(cls, SKIP_ARGS, name="safe")
assert tuple(repo.query(selector).stored(refresh=False).defs()) == (cdef,)
with __import__("pytest").raises(TypeError, match="keyword spelling or SKIP_ARGS"):
    repo.query(Definition(cls, "safe")).stored(refresh=False).count()

assert "tensorflow" not in sys.modules
        """
    )


def test_graph_building_and_planning_canonical_refs_does_not_import_backends():
    _run_import_probe(
        """
import sys

assert "tensorflow" not in sys.modules
assert "torch" not in sys.modules

from dryml.core import Definition, Repo, SKIP_ARGS
from dryml.core.cdef_graph import ConcreteDefinitionGraph
from dryml.core.definition import ConcreteDefinition
from dryml.core.freeze import FrozenDict, FrozenTuple
from dryml.core.query.fingerprint import target_local_fingerprint
from dryml.core.query.selector_graph import compile_selector_graph
from dryml.core.symbol import ImportRef

child = ConcreteDefinition._from_persisted_record(
    ImportRef("dryml.models.tf.keras.base", "Sequential"),
    FrozenTuple(()),
    FrozenDict({"layer_defs": FrozenTuple(())}),
)
root = ConcreteDefinition._from_persisted_record(
    ImportRef("dryml.models.torch.base", "Model"),
    FrozenTuple(()),
    FrozenDict({"child": child}),
)

class FakeStore:
    def catalog_key(self):
        return "fake-graph-store"

graph = ConcreteDefinitionGraph.from_root(root)
assert len(graph.nodes()) == 2
target_local_fingerprint(root)
compile_selector_graph(Definition(ImportRef("dryml.models.torch.base", "Model"), SKIP_ARGS, child=child))

repo = Repo()
repo._query_catalog.register_stored(root, FakeStore())
assert repo.query(Definition(ImportRef("dryml.models.torch.base", "Model"), SKIP_ARGS, child=child)).stored(refresh=False).count() == 1
assert repo.query(child).nested(refresh=False).owners().defs().count() == 1
repo.query(child).nested(refresh=False).owners().defs().explanation.format()

assert "tensorflow" not in sys.modules
assert "torch" not in sys.modules
        """
    )


def test_canonical_ref_edges_do_not_import_backends_for_selector_planning():
    _run_import_probe(
        """
import sys

assert "tensorflow" not in sys.modules
assert "torch" not in sys.modules

from dryml.core import Definition, Repo
from dryml.core.cdef_graph import ConcreteDefinitionGraph, EdgeKind
from dryml.core.definition import ConcreteDefinition
from dryml.core.freeze import FrozenDict, FrozenTuple
from dryml.core.query.fingerprint import target_local_fingerprint
from dryml.core.query.selector_graph import compile_selector_graph
from dryml.core.symbol import ImportRef

child = ConcreteDefinition._from_persisted_record(
    ImportRef("dryml.models.tf.keras.base", "Sequential"),
    FrozenTuple(()),
    FrozenDict({"layer_defs": FrozenTuple(())}),
)
root = ConcreteDefinition._from_persisted_record(
    ImportRef("dryml.models.torch.base", "Model"),
    FrozenTuple(()),
    FrozenDict({"child": child.freeze()}),
)

class FakeStore:
    def catalog_key(self):
        return "fake-ref-edge-store"

graph = ConcreteDefinitionGraph.from_root(root)
assert graph.edges()[0].kind is EdgeKind.REF
target_local_fingerprint(root)
selector_graph = compile_selector_graph(root)
assert selector_graph.edges == ()
selector = Definition(ImportRef("dryml.models.torch.base", "Model"), child=child.freeze())
selector_graph = compile_selector_graph(selector)
assert selector_graph.edges[0].edge_kind is EdgeKind.REF

repo = Repo()
repo._query_catalog.register_stored(root, FakeStore())
assert repo.query(root).stored(refresh=False).count() == 1

assert "tensorflow" not in sys.modules
assert "torch" not in sys.modules
        """
    )


def test_sqlite_tf_import_refs_do_not_import_tensorflow_for_query_terminals():
    _run_import_probe(
        """
import sys
import tempfile
from pathlib import Path

assert "tensorflow" not in sys.modules
assert "torch" not in sys.modules

from dryml.core import Definition, Repo, SKIP_ARGS
from dryml.core.cdef_graph import ConcreteDefinitionGraph
from dryml.core.definition import ConcreteDefinition
from dryml.core.freeze import FrozenDict, FrozenTuple
from dryml.core.query.sqlite import SQLiteQueryIndexConfig
from dryml.core.query.sqlite.index import SQLiteStoreQueryIndex
from dryml.core.store.dir import DirStore
from dryml.core.symbol import ImportRef

with tempfile.TemporaryDirectory() as tmp:
    store = DirStore(Path(tmp) / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    child_cls = ImportRef("dryml.models.tf.keras.base", "Sequential")
    root_cls = ImportRef("dryml.models.tf.base", "Model")
    child = ConcreteDefinition._from_persisted_record(child_cls, FrozenTuple(()), FrozenDict({"name": "tf-child"}))
    root = ConcreteDefinition._from_persisted_record(root_cls, FrozenTuple(()), FrozenDict({"child": child, "name": "tf-root"}))
    index = SQLiteStoreQueryIndex(
        source_key=store.catalog_key(),
        path=store.query_index_path,
        config=SQLiteQueryIndexConfig(path=store.query_index_path, journal_mode="delete"),
    )
    index.register_stored_roots(ConcreteDefinitionGraph.from_root(root), [root])
    index.close()

    repo = Repo(stores=store)
    assert repo.index_status(store=store)[0].state == "ready"
    assert repo.query(root).stored().count() == 1
    assert list(repo.query(root).stored().defs()) == [root]
    selector = Definition(child_cls, SKIP_ARGS, name="tf-child")
    assert list(repo.query(selector).nested().definitions().defs()) == [child]
    assert list(repo.query(selector).nested().owners().defs()) == [root]
    occurrences = tuple(repo.query(selector).nested().max_occurrences(2).execute())
    assert len(occurrences) == 1
    assert occurrences[0].owner == root
    assert occurrences[0].definition == child

assert "tensorflow" not in sys.modules
assert "torch" not in sys.modules
        """
    )


def test_sqlite_v2_semantic_selector_is_import_free():
    _run_import_probe(
        """
import sys
import tempfile
from pathlib import Path

assert "tensorflow" not in sys.modules
from dryml.core import Definition, Repo, SKIP_ARGS
from dryml.core.bound_args import BoundArguments
from dryml.core.cdef_graph import ConcreteDefinitionGraph
from dryml.core.definition import ConcreteDefinition
from dryml.core.query.sqlite import SQLiteQueryIndexConfig
from dryml.core.query.sqlite.index import SQLiteStoreQueryIndex
from dryml.core.store.dir import DirStore
from dryml.core.symbol import ImportRef

with tempfile.TemporaryDirectory() as tmp:
    store = DirStore(Path(tmp) / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    leaf_cls = ImportRef("dryml.models.tf.keras.base", "Sequential")
    root_cls = ImportRef("dryml.models.tf.base", "Model")
    child = ConcreteDefinition._from_bound_record(leaf_cls, BoundArguments((("name", "v2-child"),)))
    root = ConcreteDefinition._from_bound_record(root_cls, BoundArguments((("child", child),)))
    index = SQLiteStoreQueryIndex(
        source_key=store.catalog_key(),
        path=store.query_index_path,
        config=SQLiteQueryIndexConfig(path=store.query_index_path, journal_mode="delete"),
    )
    index.register_stored_roots(ConcreteDefinitionGraph.from_root(root), [root])
    index.close()

    repo = Repo(stores=store)
    selector = Definition(leaf_cls, SKIP_ARGS, name="v2-child")
    assert list(repo.query(selector).nested().definitions().defs()) == [child]
    assert list(repo.query(selector).nested().owners().defs()) == [root]
    nested_selector = Definition(root_cls, child=selector)
    assert list(repo.query(nested_selector).stored().defs()) == [root]

assert "tensorflow" not in sys.modules
        """
    )


def test_sqlite_rebuild_and_reconcile_hydrate_symbolic_v1_and_v2_roots_without_activation():
    _run_import_probe(
        """
import sys
import tempfile
from pathlib import Path

assert "tensorflow" not in sys.modules
assert "torch" not in sys.modules

from dryml.core.bound_args import BoundArguments
from dryml.core.definition import ConcreteDefinition
from dryml.core.freeze import FrozenDict, FrozenTuple
from dryml.core.query.sqlite import SQLiteQueryIndexConfig
from dryml.core.store.dir import DirStore
from dryml.core.symbol import ImportRef, SourceSpec
from dryml.core.utils.general import pickle_save

with tempfile.TemporaryDirectory() as tmp:
    store = DirStore(Path(tmp) / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    v1 = ConcreteDefinition._from_persisted_record(
        ImportRef("dryml.models.tf.keras.base", "Sequential"),
        FrozenTuple(()),
        FrozenDict({
            "source": SourceSpec.from_source(
                "raise AssertionError('V1 source must not execute')",
                kind="function",
            ),
        }),
    )
    v2 = ConcreteDefinition._from_bound_record(
        SourceSpec.from_source(
            "raise AssertionError('V2 source must not execute')",
            kind="class",
            name="UnavailableSourceClass",
        ),
        BoundArguments((("backend", ImportRef("dryml.models.torch.base", "Model")),)),
    )
    for root in (v1, v2):
        path = Path(store.object_dir(root)) / "def.pkl"
        path.parent.mkdir(parents=True)
        pickle_save(root, path)

    index = store.open_query_index()
    index.initialize_empty()
    previous_sidecar = Path(store.query_index_path).read_bytes()

    def fail_activation(self):
        raise AssertionError("indexing must not resolve symbolic classes or execute source")

    ImportRef.resolve = fail_activation
    SourceSpec.resolve = fail_activation

    rebuilt = store.rebuild_query_index()
    reconciled = store.reconcile_query_index()

    assert rebuilt.action == "rebuild"
    assert rebuilt.definitions_scanned == 2
    assert reconciled.action == "validate"
    assert not reconciled.changed
    status = store.query_index_status()
    assert status.state == "ready"
    assert status.row_counts["stored_roots"] == 2
    assert Path(store.query_index_path).read_bytes() != previous_sidecar
    with index.read_view() as view:
        for root in (v1, v2):
            exact = view.exact_ids(root)
            assert exact
            assert view.filter_stored_ids(exact) == exact

assert "tensorflow" not in sys.modules
assert "torch" not in sys.modules
        """
    )


def test_sqlite_rebuild_rejects_retired_globals_before_sidecar_activation():
    """Rebuild rejects historical authority without activating a replacement index."""

    import base64
    import hashlib
    import json

    from dryml.core.query.sqlite import SQLiteQueryIndexConfig
    from dryml.core.store.dir import DirStore

    fixture_path = Path(__file__).resolve().parents[1] / "fixtures" / "cdef_v1" / "manifest.json"
    manifest = json.loads(fixture_path.read_text())
    payload = base64.b64decode(manifest["payload"], validate=True)
    assert hashlib.sha256(payload).hexdigest() == manifest["payload_sha256"]
    assert payload[:2] == b"\x80\x05"

    with tempfile.TemporaryDirectory() as tmp:
        store = DirStore(Path(tmp) / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
        index = store.open_query_index()
        index.initialize_empty()
        sidecar_digest = hashlib.sha256(Path(store.query_index_path).read_bytes()).hexdigest()
        digest = "a" * 64
        root_path = Path(store.object_root_dir) / digest[:2] / digest / "def.pkl"
        root_path.parent.mkdir(parents=True)
        root_path.write_bytes(payload)
        authority_digest = hashlib.sha256(root_path.read_bytes()).hexdigest()

        with pytest.raises(ModuleNotFoundError, match="dryml.core" + "2"):
            store.rebuild_query_index()

        sidecar_path = Path(store.query_index_path)
        assert hashlib.sha256(sidecar_path.read_bytes()).hexdigest() == sidecar_digest
        assert hashlib.sha256(root_path.read_bytes()).hexdigest() == authority_digest
        assert store.query_index_status().state == "dirty"
        assert not list(sidecar_path.parent.glob(f"{sidecar_path.name}.rebuild-*.tmp*"))


def test_sqlite_torch_import_refs_do_not_import_torch_for_query_terminals():
    _run_import_probe(
        """
import sys
import tempfile
from pathlib import Path

assert "tensorflow" not in sys.modules
assert "torch" not in sys.modules

from dryml.core import Definition, Repo, SKIP_ARGS
from dryml.core.cdef_graph import ConcreteDefinitionGraph
from dryml.core.definition import ConcreteDefinition
from dryml.core.freeze import FrozenDict, FrozenTuple
from dryml.core.query.sqlite import SQLiteQueryIndexConfig
from dryml.core.query.sqlite.index import SQLiteStoreQueryIndex
from dryml.core.store.dir import DirStore
from dryml.core.symbol import ImportRef

with tempfile.TemporaryDirectory() as tmp:
    store = DirStore(Path(tmp) / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    child_cls = ImportRef("dryml.models.torch.base", "Wrapper")
    root_cls = ImportRef("dryml.models.torch.base", "Model")
    child = ConcreteDefinition._from_persisted_record(child_cls, FrozenTuple(()), FrozenDict({"name": "torch-child"}))
    root = ConcreteDefinition._from_persisted_record(root_cls, FrozenTuple(()), FrozenDict({"child": child, "name": "torch-root"}))
    index = SQLiteStoreQueryIndex(
        source_key=store.catalog_key(),
        path=store.query_index_path,
        config=SQLiteQueryIndexConfig(path=store.query_index_path, journal_mode="delete"),
    )
    index.register_stored_roots(ConcreteDefinitionGraph.from_root(root), [root])
    index.close()

    repo = Repo(stores=store)
    assert repo.index_status(store=store)[0].state == "ready"
    assert repo.query(root).stored().count() == 1
    assert list(repo.query(root).stored().defs()) == [root]
    selector = Definition(child_cls, SKIP_ARGS, name="torch-child")
    assert list(repo.query(selector).nested().definitions().defs()) == [child]
    assert list(repo.query(selector).nested().owners().defs()) == [root]
    occurrences = tuple(repo.query(selector).nested().max_occurrences(2).execute())
    assert len(occurrences) == 1
    assert occurrences[0].owner == root
    assert occurrences[0].definition == child

assert "tensorflow" not in sys.modules
assert "torch" not in sys.modules
        """
    )


def test_tensorflow_materialization_imports_tensorflow_when_backend_object_is_built():
    if not _module_importable("tensorflow"):
        import pytest

        pytest.skip("tensorflow is not importable in this Python environment")

    _run_import_probe(
        """
import sys

assert "tensorflow" not in sys.modules

from dryml.core import Definition
from dryml.core.symbol import ImportRef

obj = Definition(
    ImportRef("dryml.models.tf.base", "Wrapper"),
    ImportRef("tensorflow.keras.layers", "Dense"),
    1,
    input_shape=(1,),
).build(restore_state=False, build_missing=True, cache="none")

assert obj is not None
assert "tensorflow" in sys.modules
        """
    )


def test_torch_materialization_imports_torch_when_backend_object_is_built():
    if not _module_importable("torch"):
        import pytest

        pytest.skip("torch is not importable in this Python environment")

    _run_import_probe(
        """
import sys

assert "torch" not in sys.modules

from dryml.core import Definition
from dryml.core.symbol import ImportRef

obj = Definition(
    ImportRef("dryml.models.torch.base", "Wrapper"),
    ImportRef("torch.nn", "Linear"),
    1,
    1,
).build(restore_state=False, build_missing=True, cache="none")

assert obj is not None
assert "torch" in sys.modules
        """
    )


def test_query_protocol_and_planner_imports_do_not_import_sqlite():
    _run_import_probe(
        """
import sys

assert "sqlite3" not in sys.modules

from dryml.core.query.graph_plan import graph_candidate_ids
from dryml.core.query.protocols import DefinitionGraphIndex, QueryIndexReadView, StoreQueryIndex

assert graph_candidate_ids is not None
assert DefinitionGraphIndex is not None
assert QueryIndexReadView is not None
assert StoreQueryIndex is not None
assert "sqlite3" not in sys.modules
        """
    )


def test_sqlite_backend_package_import_does_not_import_sqlite3():
    _run_import_probe(
        """
import sys

assert "sqlite3" not in sys.modules
from dryml.core.query.sqlite import SQLiteQueryIndexConfig
from dryml.core.query.sqlite.connection import SQLiteConnectionManager
from dryml.core.query.sqlite.schema import SQLITE_QUERY_INDEX_SCHEMA_VERSION
from dryml.core.query.sqlite.utils import wal_runtime_is_known_safe

assert SQLiteQueryIndexConfig is not None
assert SQLiteConnectionManager is not None
assert SQLITE_QUERY_INDEX_SCHEMA_VERSION == 4
assert wal_runtime_is_known_safe((3, 51, 3))
assert "sqlite3" not in sys.modules
        """
    )


def test_dirstore_construction_does_not_import_sqlite3_or_open_index():
    _run_import_probe(
        """
import sys
import tempfile
from pathlib import Path

assert "sqlite3" not in sys.modules
from dryml.core.store.dir import DirStore

with tempfile.TemporaryDirectory() as tmp:
    store = DirStore(Path(tmp) / "store", query_index="auto")
    assert store.query_index_policy == "auto"
    assert not Path(store.dryml_dir).exists()

assert "sqlite3" not in sys.modules
        """
    )


def test_repo_construction_does_not_import_sqlite3_or_open_index():
    _run_import_probe(
        """
import sys
import tempfile
from pathlib import Path

assert "sqlite3" not in sys.modules
from dryml.core import Repo
from dryml.core.store.dir import DirStore

with tempfile.TemporaryDirectory() as tmp:
    store = DirStore(Path(tmp) / "store", query_index="auto")
    repo = Repo(stores=store)
    assert repo._query_index.store_bindings[0].store is store
    assert not Path(store.dryml_dir).exists()

assert "sqlite3" not in sys.modules
        """
    )
