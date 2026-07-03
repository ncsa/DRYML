import importlib
import sys

import pytest

import dryml.runtime as runtime
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


def test_configured_framework_import_is_separate_from_workload_allocation():
    spec = runtime.RuntimeContextSpec.from_data({"mode": "probe", "frameworks": {"tensorflow": {}}, "device_visibility": {"policy": "none"}})
    plan = runtime.build_runtime_bootstrap_plan(spec, runtime.NoAllocation)

    with runtime.enter_runtime(runtime.RuntimeMode.PROBE, runtime.NoAllocation, spec):
        with runtime.activate_runtime_bootstrap(plan):
            runtime.assert_framework_import_configured("tensorflow")
            with pytest.raises(FrameworkImportSafetyError):
                runtime.require_workload_allocation("materialize TensorFlow dataset")
