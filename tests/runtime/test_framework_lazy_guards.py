import importlib
import sys

import pytest

from dryml.runtime.errors import FrameworkImportSafetyError


def test_importing_tf_dataset_package_does_not_import_tensorflow(monkeypatch):
    monkeypatch.delitem(sys.modules, "tensorflow", raising=False)

    module = importlib.import_module("dryml.data.tf")
    _ = module.TFDataset

    assert "tensorflow" not in sys.modules


def test_importing_tf_utils_does_not_import_tensorflow(monkeypatch):
    monkeypatch.delitem(sys.modules, "tensorflow", raising=False)

    module = importlib.import_module("dryml.models.tf.utils")


    assert "tensorflow" not in sys.modules
    with pytest.raises(FrameworkImportSafetyError):
        module.keras_callback_wrapper(lambda: None)
    assert "tensorflow" not in sys.modules
