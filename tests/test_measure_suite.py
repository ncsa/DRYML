import io
import json

import pytest

from tools import measure_suite


def test_fresh_output_dir_rejects_repository_and_existing_paths(tmp_path):
    with pytest.raises(ValueError, match="outside"):
        measure_suite.fresh_output_dir(str(measure_suite.ROOT))
    with pytest.raises(ValueError, match="fresh"):
        measure_suite.fresh_output_dir(str(tmp_path))
    assert measure_suite.fresh_output_dir(str(tmp_path / "new-output")) == (tmp_path / "new-output").resolve()


def test_measure_writes_versioned_non_mutating_artifacts(tmp_path, monkeypatch):
    output_dir = tmp_path / "measurement"

    def fake_run_phase(**kwargs):
        (kwargs["output_dir"] / "timing-smoke.json").write_text(json.dumps({"records": [{"nodeid": "tests/x.py::test_x"}]}))
        return {"phase": kwargs["phase"], "returncode": 0, "wall_seconds": 0.1, "stdout": {}, "stderr": {}}

    monkeypatch.setattr(measure_suite, "run_phase", fake_run_phase)

    assert measure_suite.measure(["--output-dir", str(output_dir), "smoke", "-q"]) == 0
    run = json.loads((output_dir / "run.json").read_text())
    nodes = json.loads((output_dir / "nodes.json").read_text())
    assert run["schema"] == 1
    assert run["status"] == "success"
    assert run["coverage"] is False
    assert nodes["records"] == [{"nodeid": "tests/x.py::test_x"}]


def test_measure_fails_before_overwriting_existing_output(tmp_path):
    output_dir = tmp_path / "measurement"
    output_dir.mkdir()

    with pytest.raises(SystemExit) as error:
        measure_suite.measure(["--output-dir", str(output_dir), "smoke"])

    assert error.value.code == 2


def test_bounded_capture_records_original_size_without_retaining_all_output():
    capture = measure_suite._BoundedCapture(io.BytesIO(b"x" * (measure_suite.LOG_LIMIT_BYTES + 1)))

    capture.read()

    assert capture.original_bytes == measure_suite.LOG_LIMIT_BYTES + 1
    assert len(capture.buffer) == measure_suite.LOG_LIMIT_BYTES
