from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from dryml.core.store.dir import DirStore

from managed_workflow_fixtures import build_managed_workflow


DOCS = Path(__file__).parents[2] / "docs"


def test_documented_metric_example_runs_on_synthetic_local_data(tmp_path):
    store = DirStore(tmp_path / "store")
    _model, _train, _test, experiment, accuracy, confusion = build_managed_workflow(store)

    experiment.train(store=store)
    accuracy.compute(store=store)
    confusion.compute(store=store)

    assert accuracy.value(store=store) == pytest.approx(0.75)
    np.testing.assert_array_equal(
        confusion.matrix(store=store),
        ((1, 0, 0), (0, 1, 1), (0, 0, 1)),
    )


def test_managed_lifecycle_contract_is_linked_and_uses_stable_terms():
    adr = (DOCS / "adr" / "0009-managed-operation-lifecycle.md").read_text()
    artifacts = (DOCS / "artifacts.md").read_text()
    records = (DOCS / "records.md").read_text()
    index = (DOCS / "table_of_content.md").read_text()

    for heading in (
        "Identity Hierarchy",
        "Authority Hierarchy",
        "State Machine",
        "Fencing And Recovery",
        "Store Capabilities",
        "Schema And Compatibility",
        "Export Closure",
    ):
        assert f"## {heading}" in adr
    assert "rows are true labels" in artifacts
    assert "columns are predicted labels" in artifacts
    assert "exact consumed vector" in records
    assert "adr/0009-managed-operation-lifecycle.md" in index


def test_lightweight_metric_import_does_not_import_optional_frameworks():
    script = """
import json
import sys
from dryml.metrics import CategoricalAccuracy, ConfusionMatrix
roots = {name.split('.', 1)[0] for name in sys.modules}
print(json.dumps({
    'tensorflow': 'tensorflow' in roots,
    'torch': 'torch' in roots,
    'pyarrow': 'pyarrow' in roots,
    'exports': [CategoricalAccuracy.__name__, ConfusionMatrix.__name__],
}, sort_keys=True))
"""
    output = subprocess.check_output([sys.executable, "-c", script], text=True)
    result = json.loads(output)

    assert result == {
        "exports": ["CategoricalAccuracy", "ConfusionMatrix"],
        "pyarrow": False,
        "tensorflow": False,
        "torch": False,
    }
