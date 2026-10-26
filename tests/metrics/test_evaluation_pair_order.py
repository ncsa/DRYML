"""Named metric pairs remain valid when transport changes mapping iteration order."""

from pathlib import Path
import sys

import numpy as np
import pytest

from dryml.core import Par, Repo
from dryml.core.store.dir import DirStore
from dryml.core.tensor_spec import TensorSpec
from dryml.metrics.reductions import _AlignEvaluationTarget
from dryml.methods.errors import ImplementationSelectionError


@pytest.mark.parametrize("order", [("prediction", "target"), ("target", "prediction")])
def test_metric_pair_uses_field_names_not_mapping_order(order):
    specs = {key: TensorSpec("float32", shape=(2,), backend="numpy") for key in order}
    values = {"prediction": np.array([1, 2], dtype=np.float32), "target": np.array([1, 3], dtype=np.float32)}
    selected = _AlignEvaluationTarget().find_implementation(specs)
    result = selected({key: values[key] for key in order})
    for key in values:
        np.testing.assert_array_equal(result[key], values[key])
        assert not np.shares_memory(result[key], values[key])


@pytest.mark.parametrize("fields", [("prediction",), ("target",), ("prediction", "target", "extra")])
def test_metric_pair_still_rejects_missing_or_extra_fields(fields):
    specs = {key: TensorSpec("float32", shape=(2,), backend="numpy") for key in fields}
    with pytest.raises(ImplementationSelectionError, match="conflict"):
        _AlignEvaluationTarget().find_implementation(specs)


def _compute_metric(fold):
    from dryml.environments import introspection
    from tests.conftest import _synthetic_environment_record
    introspection.inspect_current = _synthetic_environment_record
    return fold.compute()


@pytest.mark.usefixtures("fixed_snapshot_environment")
def test_dispatched_symbolic_metric_completes_after_node_transport(tmp_path):
    from dryml import dispatch
    from dryml.core.execute import CoreOptions
    from dryml.execute.subprocess import SubProcessConfig
    from dryml.metrics import regressor_mse
    from tests.metrics.test_reduction_metrics import EvaluationDataset, IdentityModel

    repo = Repo(DirStore(tmp_path / "store"))
    try:
        source = repo.save(EvaluationDataset(({"x": [1.0, 5.0], "y": [2.0, 2.0]},)))
        model = repo.save(IdentityModel())
        recipe = regressor_mse(Par("source"), Par("model"), mode="global").sub(
            source=source, model=model, traverse_refs=True)
        view = dispatch.with_options(core=CoreOptions(repo=repo), backend=SubProcessConfig(
            automatic_environment_discovery=False, python_executable=Path(sys.executable)))
        result = view.run(_compute_metric, recipe)
        assert result.status.state == "completed"
        restored = repo.load_state_ref(result.final_state_ref, reuse_live="never")
        assert restored.ready and restored.value() == pytest.approx(5.0)
    finally:
        repo.close(flush=False)
