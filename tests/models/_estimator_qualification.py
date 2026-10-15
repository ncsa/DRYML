"""Test-only installed-wheel origin evidence for estimator Core workers."""

import base64
import csv
import hashlib
from importlib import import_module, metadata
import io
import json
import os
from pathlib import Path
import sys


ORIGIN_MODE = "DRYML_TEST_INSTALLED_ESTIMATORS"


def installed_origins(kind, *, worker=False):
    """Return verified wheel paths/hashes, or fail on source/editable imports.

    ``kind`` selects XGBoost or statsmodels recipes without fitting. With
    ``worker=True``, also verify the running ``-m`` bootstrap, rather than only
    importing a second copy. Assertion failures reject missing wheel metadata,
    editable installs, non-RECORD origins, and bytes differing from RECORD.
    """
    distribution = metadata.distribution("dryml")
    direct = json.loads(distribution.read_text("direct_url.json") or "{}")
    assert not direct.get("dir_info", {}).get("editable"), "editable DRYML is not a wheel qualification"
    files = distribution.files or ()
    wheels = [item for item in files if str(item).endswith(".dist-info/WHEEL")]
    records = [item for item in files if str(item).endswith(".dist-info/RECORD")]
    assert len(wheels) == len(records) == 1, "installed wheel WHEEL/RECORD metadata required"
    wheel, record = wheels[0], records[0]
    wheel_path = Path(distribution.locate_file(wheel)).resolve(strict=True)
    record_path = Path(distribution.locate_file(record)).resolve(strict=True)
    assert "Wheel-Version:" in wheel_path.read_text(encoding="utf-8"), wheel_path
    rows = {row[0]: row[1:] for row in csv.reader(io.StringIO(record_path.read_text(encoding="utf-8")))}
    backend = "dryml.models." + ("xgb" if kind.startswith("xgb") else "statsmodels")
    modules = ("dryml", "dryml.execute._worker", backend, backend + ".base")
    origins = {}
    for name in modules:
        module = import_module(name)
        relative = name.replace(".", "/") + ("/__init__.py" if hasattr(module, "__path__") else ".py")
        expected = Path(distribution.locate_file(relative)).resolve(strict=True)
        actual = Path(module.__spec__.origin).resolve(strict=True)
        assert actual == expected, f"non-wheel origin for {name}: {actual}; RECORD expects {expected}"
        assert relative in rows, f"wheel RECORD lacks {relative}"
        digest, size = rows[relative]
        assert digest.startswith("sha256="), (relative, digest)
        payload = actual.read_bytes()
        sha256 = hashlib.sha256(payload).digest()
        assert base64.urlsafe_b64encode(sha256).rstrip(b"=").decode("ascii") == digest.split("=", 1)[1], relative
        assert len(payload) == int(size), relative
        origins[name] = {"path": str(actual), "sha256": sha256.hex()}
    if worker:
        main = sys.modules["__main__"]
        assert main.__spec__.name == "dryml.execute._worker", main.__spec__
        assert Path(main.__spec__.origin).resolve(strict=True) == Path(origins["dryml.execute._worker"]["path"])
    return {"wheel": str(wheel_path), "record": str(record_path), "modules": origins}


def qualification_origins(kind, *, worker=False):
    """Verify/return wheel evidence only when the CI origin mode is enabled.

    Local source-mode tests return an empty receipt. Enabled mode propagates
    origin assertion failures; worker evidence is printed before any native fit.
    """
    if os.environ.get(ORIGIN_MODE) != "1":
        return {}
    evidence = installed_origins(kind, worker=worker)
    if worker:
        print(json.dumps({"installed_origins": evidence}), flush=True)
    return evidence


def inert_worker_origins():
    """Construct inert recipes in a real Core worker, rejecting native imports.

    Return both backends' verified origins and the actual worker PID. No native
    dependencies or fitting are needed; unexpected optional imports fail loudly.
    """
    class RejectNative:
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in {"xgboost", "statsmodels", "sklearn", "scipy", "pandas"}:
                raise AssertionError("unexpected native import: " + fullname)

    blocker = RejectNative()
    sys.meta_path.insert(0, blocker)
    try:
        origins = {}
        for kind, backend in (("xgb-regressor", "xgb"), ("stats-logit", "statsmodels")):
            origins[kind] = installed_origins(kind, worker=True)
            module = import_module("dryml.models." + backend)
            module.RegressionModel()
            module.ClassifierModel()
            module.BasicTraining()
        assert not {"xgboost", "statsmodels", "sklearn", "scipy", "pandas"} & sys.modules.keys()
        return {"pid": os.getpid(), "origins": origins}
    finally:
        sys.meta_path.remove(blocker)
