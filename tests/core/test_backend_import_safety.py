import os
from pathlib import Path
import subprocess
import sys


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
from dryml.core2.tensor_spec import TensorSpec

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
from dryml.core2.tensor_spec import TensorSpec
from dryml.torch import TorchTensorSpec

spec = TensorSpec("float32", shape=(32,))
assert hasattr(spec, "torch")
torch_spec = TorchTensorSpec(shape=(32,), dtype="torch.float32")
assert torch_spec.layout == "torch.strided"
assert "torch" not in sys.modules
        """
    )


def test_query_indexing_canonical_refs_does_not_import_backends():
    _run_import_probe(
        """
import sys

assert "tensorflow" not in sys.modules
assert "torch" not in sys.modules

from dryml.core2 import Repo
from dryml.core2.definition import ConcreteDefinition
from dryml.core2.freeze import FrozenDict, FrozenTuple
from dryml.core2.query.fingerprint import legacy_target_fingerprint
from dryml.core2.symbol import ImportRef

cdef = ConcreteDefinition(
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
from dryml.core2 import Definition, Repo, SKIP_ARGS
from dryml.core2.definition import ConcreteDefinition
from dryml.core2.freeze import FrozenDict, FrozenTuple
from dryml.core2.symbol import ImportRef

class FakeStore:
    def catalog_key(self):
        return "fake-tf-store"

cdef = ConcreteDefinition(
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
from dryml.core2 import Definition, Repo, SKIP_ARGS
from dryml.core2.definition import ConcreteDefinition
from dryml.core2.freeze import FrozenDict, FrozenTuple
from dryml.core2.symbol import ImportRef

class FakeStore:
    def catalog_key(self):
        return "fake-torch-store"

cdef = ConcreteDefinition(
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


def test_graph_building_and_planning_canonical_refs_does_not_import_backends():
    _run_import_probe(
        """
import sys

assert "tensorflow" not in sys.modules
assert "torch" not in sys.modules

from dryml.core2 import Definition, Repo, SKIP_ARGS
from dryml.core2.cdef_graph import ConcreteDefinitionGraph
from dryml.core2.definition import ConcreteDefinition
from dryml.core2.freeze import FrozenDict, FrozenTuple
from dryml.core2.query.fingerprint import target_local_fingerprint
from dryml.core2.query.selector_graph import compile_selector_graph
from dryml.core2.symbol import ImportRef

child = ConcreteDefinition(
    ImportRef("dryml.models.tf.keras.base", "Sequential"),
    FrozenTuple(()),
    FrozenDict({"layer_defs": FrozenTuple(())}),
)
root = ConcreteDefinition(
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


def test_sqlite_tf_import_refs_do_not_import_tensorflow_for_query_terminals():
    _run_import_probe(
        """
import sys
import tempfile
from pathlib import Path

assert "tensorflow" not in sys.modules
assert "torch" not in sys.modules

from dryml.core2 import Definition, Repo, SKIP_ARGS
from dryml.core2.cdef_graph import ConcreteDefinitionGraph
from dryml.core2.definition import ConcreteDefinition
from dryml.core2.freeze import FrozenDict, FrozenTuple
from dryml.core2.query.sqlite import SQLiteQueryIndexConfig
from dryml.core2.query.sqlite.index import SQLiteStoreQueryIndex
from dryml.core2.store.dir import DirStore
from dryml.core2.symbol import ImportRef

with tempfile.TemporaryDirectory() as tmp:
    store = DirStore(Path(tmp) / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    child_cls = ImportRef("dryml.models.tf.keras.base", "Sequential")
    root_cls = ImportRef("dryml.models.tf.base", "Model")
    child = ConcreteDefinition(child_cls, FrozenTuple(()), FrozenDict({"name": "tf-child"}))
    root = ConcreteDefinition(root_cls, FrozenTuple(()), FrozenDict({"child": child, "name": "tf-root"}))
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


def test_sqlite_torch_import_refs_do_not_import_torch_for_query_terminals():
    _run_import_probe(
        """
import sys
import tempfile
from pathlib import Path

assert "tensorflow" not in sys.modules
assert "torch" not in sys.modules

from dryml.core2 import Definition, Repo, SKIP_ARGS
from dryml.core2.cdef_graph import ConcreteDefinitionGraph
from dryml.core2.definition import ConcreteDefinition
from dryml.core2.freeze import FrozenDict, FrozenTuple
from dryml.core2.query.sqlite import SQLiteQueryIndexConfig
from dryml.core2.query.sqlite.index import SQLiteStoreQueryIndex
from dryml.core2.store.dir import DirStore
from dryml.core2.symbol import ImportRef

with tempfile.TemporaryDirectory() as tmp:
    store = DirStore(Path(tmp) / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    child_cls = ImportRef("dryml.models.torch.base", "Wrapper")
    root_cls = ImportRef("dryml.models.torch.base", "Model")
    child = ConcreteDefinition(child_cls, FrozenTuple(()), FrozenDict({"name": "torch-child"}))
    root = ConcreteDefinition(root_cls, FrozenTuple(()), FrozenDict({"child": child, "name": "torch-root"}))
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
    _run_import_probe(
        """
import sys

assert "tensorflow" not in sys.modules

from dryml.core2 import Definition
from dryml.core2.symbol import ImportRef

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
    _run_import_probe(
        """
import sys

assert "torch" not in sys.modules

from dryml.core2 import Definition
from dryml.core2.symbol import ImportRef

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

from dryml.core2.query.graph_plan import graph_candidate_ids
from dryml.core2.query.protocols import DefinitionGraphIndex, QueryIndexReadView, StoreQueryIndex

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
from dryml.core2.query.sqlite import SQLiteQueryIndexConfig
from dryml.core2.query.sqlite.connection import SQLiteConnectionManager
from dryml.core2.query.sqlite.schema import SQLITE_QUERY_INDEX_SCHEMA_VERSION
from dryml.core2.query.sqlite.utils import wal_runtime_is_known_safe

assert SQLiteQueryIndexConfig is not None
assert SQLiteConnectionManager is not None
assert SQLITE_QUERY_INDEX_SCHEMA_VERSION == 2
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
from dryml.core2.store.dir import DirStore

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
from dryml.core2 import Repo
from dryml.core2.store.dir import DirStore

with tempfile.TemporaryDirectory() as tmp:
    store = DirStore(Path(tmp) / "store", query_index="auto")
    repo = Repo(stores=store)
    assert repo._query_index.store_bindings[0].store is store
    assert not Path(store.dryml_dir).exists()

assert "sqlite3" not in sys.modules
        """
    )
