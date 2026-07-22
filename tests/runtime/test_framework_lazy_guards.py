import importlib
import sys

import pytest

import dryml.runtime as runtime
from dryml.runtime.errors import FrameworkImportSafetyError


def test_importing_tf_dataset_package_does_not_import_tensorflow(monkeypatch):
    monkeypatch.delitem(sys.modules, "tensorflow", raising=False)

    module = importlib.import_module("dryml.data.tf")

    assert module.__all__ == ["TensorFlowCacheView"]
    assert module.TensorFlowCacheView.__module__ == "dryml.data.tf.cache"
    assert not hasattr(module, "TFDataset")
    assert "tensorflow" not in sys.modules


def test_importing_tf_utils_does_not_import_tensorflow(monkeypatch):
    monkeypatch.delitem(sys.modules, "tensorflow", raising=False)

    module = importlib.import_module("dryml.models.tf.utils")

    assert "tensorflow" not in sys.modules
    with pytest.raises(FrameworkImportSafetyError):
        module.keras_callback_wrapper(lambda: None)
    assert "tensorflow" not in sys.modules


def test_importing_torch_dataset_package_does_not_import_torch(monkeypatch):
    monkeypatch.delitem(sys.modules, "torch", raising=False)

    module = importlib.import_module("dryml.data.torch")

    assert module.__all__ == ["TorchCacheView"]
    assert module.TorchCacheView.__module__ == "dryml.data.torch.cache"
    assert "torch" not in sys.modules
    assert not hasattr(module, "TorchDataset")
    assert not hasattr(module, "TorchIterableDatasetWrapper")
    assert "torch" not in sys.modules


def test_configured_framework_import_is_separate_from_workload_allocation():
    spec = runtime.RuntimeContextSpec.from_data({"mode": "probe", "frameworks": {"tensorflow": {}}, "device_visibility": {"policy": "none"}})
    plan = runtime.build_runtime_bootstrap_plan(spec, runtime.NoAllocation)

    with runtime.enter_runtime(runtime.RuntimeMode.PROBE, runtime.NoAllocation, spec):
        with runtime.activate_runtime_bootstrap(plan):
            runtime.assert_framework_import_configured("tensorflow")
            with pytest.raises(FrameworkImportSafetyError):
                runtime.require_workload_allocation("materialize TensorFlow dataset")


def test_import_configured_framework_reuses_already_imported_module(monkeypatch):
    fake = type(sys)("already_imported_framework")
    monkeypatch.setitem(sys.modules, "already_imported_framework", fake)

    assert runtime.import_configured_framework("already_imported_framework") is fake


def test_import_configured_framework_guards_new_import(monkeypatch):
    monkeypatch.delitem(sys.modules, "not_imported_framework", raising=False)

    with pytest.raises(FrameworkImportSafetyError):
        runtime.import_configured_framework("not_imported_framework")
