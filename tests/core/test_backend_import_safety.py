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
