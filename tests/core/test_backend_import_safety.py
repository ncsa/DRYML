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
from dryml.core2.query.fingerprint import target_fingerprint
from dryml.core2.symbol import ImportRef

cdef = ConcreteDefinition(
    ImportRef("dryml.models.tf.keras.base", "Sequential"),
    FrozenTuple(()),
    FrozenDict({"layer_defs": FrozenTuple(())}),
)

repo = Repo()
repo._query_catalog.register_cached(cdef)
target_fingerprint(cdef)
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
repo._query_catalog.register_stored_graph(cdef, FakeStore())

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
repo._query_catalog.register_stored_graph(cdef, FakeStore())

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
repo._query_catalog.register_stored_graph(root, FakeStore())
assert repo.query(Definition(ImportRef("dryml.models.torch.base", "Model"), SKIP_ARGS, child=child)).stored(refresh=False).count() == 1
assert repo.query(child).nested(refresh=False).owners().defs().count() == 1
repo.query(child).nested(refresh=False).owners().defs().explanation.format()

assert "tensorflow" not in sys.modules
assert "torch" not in sys.modules
        """
    )
