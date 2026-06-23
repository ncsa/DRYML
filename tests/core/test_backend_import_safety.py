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
