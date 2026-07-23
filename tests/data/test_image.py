import numpy as np

from dryml.core.cardinality import Cardinality
from dryml.core.tensor_spec import TensorSpec
from dryml.data import Map
from dryml.data.dataset import Dataset
from dryml.data.image import ImageNormalize


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
