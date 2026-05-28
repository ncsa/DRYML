import numpy as np

from dryml.data import ArrayDataset, Batch, collate_xy
from dryml.models.utils import prepare_training_data


def test_prepare_training_data_unbatches_and_takes_examples():
    x = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)
    y = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    ds = Batch(ArrayDataset((x, y)), 2)

    train_data = prepare_training_data(ds, num_examples=2)
    out_x, out_y, n = collate_xy(train_data)

    assert n == 2
    assert out_x.shape == (2, 1)
    assert out_y.shape == (2, 1)
    np.testing.assert_allclose(out_x[:, 0], np.array([0.0, 1.0], dtype=np.float32))


def test_backend_model_packages_import_without_backend_runtime():
    import dryml.models.tf as tf_models
    import dryml.models.torch as torch_models

    assert hasattr(tf_models, "BasicTraining")
    assert hasattr(tf_models, "Optimizer")
    assert hasattr(tf_models, "Wrapper")
    assert hasattr(torch_models, "BasicTraining")
    assert hasattr(torch_models, "Optimizer")
    assert hasattr(torch_models, "Wrapper")
