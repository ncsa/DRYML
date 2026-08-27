import pytest
import sys

from dryml.core.backend import discover_backend
import numpy as np

torch = pytest.importorskip("torch")
if not hasattr(torch, "Tensor"):
    sys.modules.pop("torch", None)
    pytest.skip("PyTorch is not installed.", allow_module_level=True)
import dryml.torch as dryml_torch

from dryml.core.dtype import DType
from dryml.core.tensor_spec import Dynamic, Layout, TensorSpec, as_tensor_spec
from dryml.core.backend import discover_backend
from dryml.torch import TorchTensorSpec


def test_torch_dtype_from_dtype_object():
    assert dryml_torch.dtype(torch.float32) == DType("float", 32)
    assert dryml_torch.dtype(torch.int64) == DType("int", 64)
    assert dryml_torch.dtype(torch.bool) == DType("bool")


def test_torch_dtype_from_tensor():
    x = torch.zeros((2, 3), dtype=torch.float32)
    assert dryml_torch.dtype(x) == DType("float", 32)


def test_torch_tensor_spec_dense_unbatched():
    x = torch.zeros((4, 32), dtype=torch.float32)

    spec = dryml_torch.as_tensor_spec(x, batched=False)

    assert spec.dtype == DType("float", 32)
    assert spec.shape == (4, 32)
    assert spec.batch is None
    assert spec.layout is Layout.DENSE


def test_torch_tensor_spec_dense_batched():
    x = torch.zeros((4, 32), dtype=torch.float32)

    spec = dryml_torch.as_tensor_spec(x, batched=True)

    assert spec.dtype == DType("float", 32)
    assert spec.shape == (32,)
    assert spec.batch == 4
    assert spec.layout is Layout.DENSE
    assert spec.batch_axis_name == "batch"


def test_torch_tensor_spec_sparse_coo():
    indices = torch.tensor([[0, 1], [2, 0]])
    values = torch.tensor([3.0, 4.0])
    x = torch.sparse_coo_tensor(indices, values, size=(2, 3))

    spec = dryml_torch.as_tensor_spec(x)

    assert spec.dtype == DType("float", 32)
    assert spec.shape == (2, 3)
    assert spec.batch is None
    assert spec.layout is Layout.SPARSE
    assert spec.sparse_format == "coo"


def test_torch_tensor_spec_from_adapter():
    x = TorchTensorSpec(
        shape=(Dynamic, 32),
        dtype=torch.float32,
        layout=torch.strided,
    )

    spec = dryml_torch.as_tensor_spec(x, batched=True)

    assert spec.dtype == DType("float", 32)
    assert spec.shape == (32,)
    assert spec.batch is Dynamic
    assert spec.layout is Layout.DENSE


def test_torch_roundtrip_dense_if_forward_methods_installed():
    spec = TensorSpec(dtype="float32", shape=(32,), batch=Dynamic)

    if not hasattr(spec, "torch"):
        pytest.skip("TensorSpec.torch() is not installed.")

    torch_spec = spec.torch()

    assert isinstance(torch_spec, TorchTensorSpec)
    assert torch_spec.shape == (Dynamic, 32)
    assert torch_spec.dtype == torch.float32
    assert torch_spec.layout == torch.strided


def test_torch_roundtrip_sparse_if_forward_methods_installed():
    spec = TensorSpec(
        dtype="float32",
        shape=(16, 8),
        layout=Layout.SPARSE,
        sparse_format="coo",
    )

    if not hasattr(spec, "torch"):
        pytest.skip("TensorSpec.torch() is not installed.")

    torch_spec = spec.torch()

    assert isinstance(torch_spec, TorchTensorSpec)
    assert torch_spec.shape == (16, 8)
    assert torch_spec.dtype == torch.float32
    assert torch_spec.layout == torch.sparse_coo


def test_torch_backend_detectors():
    assert discover_backend(torch.tensor(1)) == "torch"
    assert discover_backend(torch.tensor(1.5, dtype=torch.float32)) == "torch"
    assert discover_backend(np.uint8(1)) == "numpy"
    assert discover_backend(np.float64(1.5)) == "numpy"


def test_torch_tensor_spec_auto_ingest():
    x = TorchTensorSpec((4, 32), dtype=torch.float32)
    spec = TensorSpec(dtype="float32", shape=(32,), batch=4)
    assert spec == as_tensor_spec(x, batched=True)

    x = torch.randn((4, 32), dtype=torch.float32)
    spec = TensorSpec(dtype="float32", shape=(4, 32,))
    assert spec == as_tensor_spec(x)
