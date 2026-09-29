import numpy as np
import pytest

from dryml.core.cardinality import Cardinality
from dryml.core.tensor_spec import TensorSpec
from dryml.data import Cast, Map, Scale
from dryml.data.dataset import Dataset
from dryml.data.image import ImageNormalize
from dryml.methods import Method


class ListDataset(Dataset):
    def __init__(self, items, spec):
        self.items = list(items)
        super().__init__(spec=spec)

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return Cardinality.finite(len(self.items))


def test_image_normalize_numpy_map():
    src = ListDataset(
        [np.array([0, 255], dtype=np.uint8)],
        TensorSpec("uint8", shape=(2,), backend="numpy"),
    )

    ds = Map(src, ImageNormalize())
    out = list(ds)

    assert ds.spec == TensorSpec("float32", shape=(2,), backend="numpy")
    assert out[0].dtype == np.dtype("float32")
    np.testing.assert_allclose(out[0], np.array([0.0, 1.0], dtype=np.float32))
    assert isinstance(ImageNormalize(), Method)


def test_backend_methods_preserve_tensorflow_results_after_explicit_registration():
    tf = pytest.importorskip("tensorflow")
    import dryml.tf

    value = tf.constant([0, 255], dtype=tf.uint8)

    normalized = ImageNormalize()(value)
    assert normalized.dtype == tf.float32
    np.testing.assert_allclose(normalized.numpy(), [0.0, 1.0])
    assert Cast("float32")(value).dtype == tf.float32
    np.testing.assert_allclose(Scale.from_range(0, 255)(value).numpy(), [0.0, 1.0])


def test_backend_methods_preserve_torch_results_after_explicit_registration():
    torch = pytest.importorskip("torch")
    import dryml.torch

    value = torch.tensor([0, 255], dtype=torch.uint8)

    normalized = ImageNormalize()(value)
    assert normalized.dtype == torch.float32
    assert torch.allclose(normalized, torch.tensor([0.0, 1.0]))
    assert Cast("float32")(value).dtype == torch.float32
    assert torch.allclose(Scale.from_range(0, 255)(value), torch.tensor([0.0, 1.0]))
