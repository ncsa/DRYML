"""Real CPU estimator chains across managed, Artifact, Repo and Core workers.

Native dependencies are mandatory on explicit execution. Recovery tests use the
same fixtures without claiming continuation inside an interrupted opaque fit.
"""

from importlib import import_module
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pytest

from dryml.core import Definition, Par, Repo, StateRef
from dryml.core.execute import CoreExecutionError, CoreOptions, Executor, current_context
from dryml.core.store.dir import DirStore
from dryml.core.utils.graph.path import GraphPath, Parameter
from dryml.data import ArrayDataset, Batch, as_supervised
from dryml.execute.subprocess import SubProcessConfig
from dryml.execute.errors import CleanupError
from dryml.managed import ManagedConfig
from dryml.models import Experiment, ExperimentData

from .test_experiment_artifacts import SavedModelValue
from ._estimator_qualification import ORIGIN_MODE, qualification_origins


pytestmark = [pytest.mark.speed_heavy,
              pytest.mark.usefixtures("fixed_managed_snapshot_environment")]

KINDS = ("xgb-classifier", "xgb-regressor", "stats-ols", "stats-logit")
RECOVERY_KINDS = ("xgb-regressor", "stats-logit")


def _arrays(kind):
    rng = np.random.default_rng(20261015)
    x = rng.normal(size=(48, 2)).astype(np.float32)
    y = (0.3 + x[:, 0] - .4 * x[:, 1]).astype(np.float32)
    if kind in ("xgb-classifier", "stats-logit"):
        y = (rng.random(48) < 1 / (1 + np.exp(-y))).astype(np.int64)
    return x, y


def _recipe(kind):
    backend = import_module("dryml.models." + ("xgb" if kind.startswith("xgb") else "statsmodels"))
    cls = backend.ClassifierModel if kind in ("xgb-classifier", "stats-logit") else backend.RegressionModel
    options = dict(n_estimators=3, max_depth=2, min_child_weight=0) if kind.startswith("xgb") else dict(intercept=True)
    return cls(**options), backend.BasicTraining()


def _experiment(kind, repo):
    x, y = _arrays(kind)
    model, trainer = _recipe(kind)
    return Experiment(
        model, trainer, train_data=Batch(as_supervised(ArrayDataset((x, y)), 0, 1), 11),
        artifacts={"model": Definition(SavedModelValue, Par("this.model"))}, repo=repo,
    )


def _native_fit_counter(kind, monkeypatch):
    if kind.startswith("xgb"):
        native = import_module("xgboost")
        cls = native.XGBClassifier if kind == "xgb-classifier" else native.XGBRegressor
    elif kind == "stats-logit":
        cls = import_module("statsmodels.discrete.discrete_model").Logit
    else:
        cls = import_module("statsmodels.regression.linear_model").OLS
    original = cls.fit
    calls = []

    def fit(native, *args, **kwargs):
        calls.append(len(args[0]) if kind.startswith("xgb") else len(native.endog))
        return original(native, *args, **kwargs)

    monkeypatch.setattr(cls, "fit", fit)
    return calls


def _forbid_native_encoding(model, monkeypatch):
    def reject(*args, **kwargs):
        pytest.fail("associated-checkpoint replay must re-emit retained bytes, not serialize native state")

    monkeypatch.setattr(type(model.mdl), "save_model" if model.state_codec == "xgboost1" else "save", reject)


def _assert_row(repo, row, *, control=None):
    import pandas as pd

    assert row.evaluation_status == "completed"
    assert row.examples_seen == 48
    for name in ("training_loss", "total_parameters", "trainable_parameters"):
        assert pd.isna(row[name]), f"{name} must be unavailable, not fabricated zero"
    artifact = repo.load_state_ref(row.eval_artifacts["model"], reuse_live="never", cache="none")
    assert artifact.model == row.state_ref.at(GraphPath((Parameter("model"),)))
    assert artifact.ready
    assert artifact.compute.status(state_repo=repo, control_store=control).final_state_ref == row.eval_artifacts["model"]


@pytest.mark.parametrize("kind", KINDS)
def test_terminal_artifacts_callbacks_and_fresh_final_authority(tmp_path, kind):
    """Each publication is checked against its own association, not an assumed pin."""
    repo = Repo(DirStore(tmp_path / "store"))
    exp = _experiment(kind, repo)
    observed = []

    def terminal(obj, context):
        row = ExperimentData.find(context.checkpoint_state_ref.object_projection(), repo=repo).data.iloc[0]
        _assert_row(repo, row)
        assert row.state_ref == context.checkpoint_state_ref
        observed.append(row.state_ref)

    try:
        final = exp.train(managed=ManagedConfig(state_repo=repo, callbacks=[terminal]))
        status = exp.train.status(state_repo=repo)
        assert isinstance(final, StateRef)
        assert status.state == "completed" and status.final_state_ref == final
        assert len(observed) == 1
        row = ExperimentData.find(final.object_projection(), repo=repo).data.iloc[0]
        assert row.state_ref == observed[0]
        reopened = repo.load_state_ref(final, reuse_live="never", cache="none")
        assert reopened is not exp and reopened.model is not exp.model
        assert (reopened.state.step, reopened.state.epoch, reopened.state.examples_seen) == (1, 1, 48)
        assert reopened.state.pending_epoch_postlude is None
        assert reopened.model._accepted_fit.payload == exp.model._accepted_fit.payload
        np.testing.assert_array_equal(reopened.model(_arrays(kind)[0]), exp.model(_arrays(kind)[0]))
    finally:
        repo.close(flush=False)


def _worker_fit(kind, failure=None):
    """Fit in the admitted Core worker; return only references and ordinary evidence.

    ``kind`` selects a tiny CPU recipe. Optional test-only failure controls inject
    dependency admission, workload, or teardown errors. Accepted-state and failure
    evidence is printed before propagation; no live handles cross transport.
    """
    origins = qualification_origins(kind, worker=True)
    context = current_context()
    exp = _experiment(kind, context.repo)
    backend = import_module("dryml.models." + ("xgb" if kind.startswith("xgb") else "statsmodels") + ".base")
    if failure in ("missing", "incompatible"):
        from importlib.metadata import PackageNotFoundError

        def version(name):
            if failure == "missing":
                raise PackageNotFoundError(name)
            return "unqualified"

        backend.version = version
    if failure == "workload-cleanup":
        def fail_close(*args, **kwargs):
            raise OSError("injected worker teardown")

        context.repo.close = fail_close
    with tempfile.TemporaryDirectory(prefix="estimator-worker-") as work:
        try:
            if failure == "workload-cleanup":
                raise RuntimeError("injected worker workload")
            final = exp.train(managed=ManagedConfig(
                state_repo=context.repo, control_store=context.control_store))
        except Exception:
            status = exp.train.status(state_repo=context.repo, control_store=context.control_store)
            print(json.dumps({"pid": os.getpid(), "accepted": exp.model._accepted_fit is not None,
                              "step": exp.state.step, "examples": exp.state.examples_seen,
                              "final": status.final_state_ref is not None,
                              "origins": origins}), flush=True)
            raise
        return {"final": final, "pid": os.getpid(), "work": work, "origins": origins,
                "prediction": exp.model(_arrays(kind)[0]).tolist(),
                "payload_sha256": hashlib.sha256(exp.model._accepted_fit.payload).hexdigest()}


def _executor(tmp_path, repo, control):
    spool = tmp_path / "spool"
    spool.mkdir()
    temporary = tmp_path / "worker-tmp"
    temporary.mkdir()
    cache = tmp_path / "worker-cache"
    cache.mkdir()
    checkout = Path(__file__).resolve().parents[2]
    installed = os.environ.get(ORIGIN_MODE) == "1"
    paths = (checkout,) if installed else (checkout, checkout / "src")
    environment = {"PYTHONDONTWRITEBYTECODE": "1", "PYTHONFAULTHANDLER": "1", "TMPDIR": str(temporary),
                   "TMP": str(temporary), "TEMP": str(temporary),
                   "XDG_CACHE_HOME": str(cache), "MPLCONFIGDIR": str(cache / "matplotlib"),
                   ORIGIN_MODE: "1" if installed else "0",
                   "PYTHONPATH": os.pathsep.join(map(str, paths))}
    environment.update({name: "1" for name in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")})
    return Executor(SubProcessConfig(
        spool_directory=spool, env_vars=environment, execution_timeout=90,
        automatic_environment_discovery=False, admission_timeout=60,
    ), core=CoreOptions(repo=repo, control_store=control, return_objects=False)), temporary


@pytest.mark.parametrize("kind", KINDS)
def test_core_cpu_worker_publication_fresh_reopen_and_isolation(tmp_path, kind, record_property):
    """Two serial real workers own separate native state, authority and scratch paths."""
    coordinator_origins = qualification_origins(kind)
    record_property("coordinator_origins", json.dumps(coordinator_origins, sort_keys=True))
    results = []
    for index in range(2):
        root = tmp_path / str(index)
        root.mkdir()
        repo = Repo(DirStore(root / "store"))
        control = DirStore(root / "control")
        executor, temporary = _executor(root, repo, control)
        try:
            future = executor.submit(_worker_fit, kind)
            try:
                result = future.result(timeout=150)
            except BaseException:
                output = future.output.snapshot()
                evidence = json.dumps({"stdout": output.stdout, "stderr": output.stderr})
                record_property("failed_worker_output", evidence)
                print("Failed native worker output: " + evidence)
                raise
            assert result["origins"] == coordinator_origins
            assert result["pid"] != os.getpid()
            assert Path(result["work"]).parent == temporary
            assert not Path(result["work"]).exists()
            fresh_repo = Repo(DirStore(root / "store"))
            try:
                fresh = fresh_repo.load_state_ref(result["final"], reuse_live="never", cache="none")
                np.testing.assert_array_equal(fresh.model(_arrays(kind)[0]), result["prediction"])
                assert hashlib.sha256(fresh.model._accepted_fit.payload).hexdigest() == result["payload_sha256"]
                assert fresh.train.status(state_repo=fresh_repo, control_store=control).final_state_ref == result["final"]
                row = ExperimentData.find(result["final"].object_projection(), repo=fresh_repo).data.iloc[0]
                _assert_row(fresh_repo, row, control=control)
                results.append((result, fresh.model))
            finally:
                fresh_repo.close(flush=False)
            future.cleanup(timeout=15)
            snapshot = future.snapshot()
            assert snapshot.cleanup_state == "complete" and not snapshot.cleanup_issues
            record_property(f"worker_{index}", json.dumps({
                "pid": result["pid"], "final": result["final"].digest(),
                "cleanup": snapshot.cleanup_state, "payload": result["payload_sha256"],
                "origins": result["origins"],
            }, sort_keys=True))
        finally:
            executor.close(cancel=True, timeout=20)
            repo.close(flush=False)
            control.close()
    assert results[0][0]["pid"] != results[1][0]["pid"]
    assert results[0][0]["work"] != results[1][0]["work"]
    assert results[0][0]["final"].object != results[1][0]["final"].object
    assert results[0][1]._accepted_fit is not results[1][1]._accepted_fit
    assert results[0][1].mdl is not results[1][1].mdl


@pytest.mark.parametrize("kind", RECOVERY_KINDS)
@pytest.mark.parametrize("failure", ("missing", "incompatible", "workload-cleanup"))
def test_core_worker_failure_keeps_admission_and_cleanup_evidence_separate(
        tmp_path, kind, failure, record_property):
    """Worker errors never install accepted state or masquerade as completion."""
    coordinator_origins = qualification_origins(kind)
    record_property("coordinator_origins", json.dumps(coordinator_origins, sort_keys=True))
    repo = Repo(DirStore(tmp_path / "store"))
    control = DirStore(tmp_path / "control")
    executor, _ = _executor(tmp_path, repo, control)
    try:
        future = executor.submit(_worker_fit, kind, failure)
        with pytest.raises(CoreExecutionError) as caught:
            future.result(timeout=150)
        assert caught.value.phase == "invoke"
        expected = "dryml[" if failure == "missing" else ("reader/writer versions" if failure == "incompatible" else "injected worker workload")
        assert expected in str(caught.value)
        output = future.output.snapshot()
        facts = json.loads(output.stdout.strip().splitlines()[-1])
        assert facts["origins"] == coordinator_origins
        assert facts["pid"] != os.getpid()
        assert facts["accepted"] is facts["final"] is False
        assert facts["step"] == facts["examples"] == 0
        if failure == "workload-cleanup":
            with pytest.raises(CleanupError, match="incomplete"):
                future.cleanup(timeout=15)
            assert future.exception(timeout=1) is caught.value
        else:
            future.cleanup(timeout=15)
        snapshot = future.snapshot()
        if failure == "workload-cleanup":
            assert snapshot.cleanup_state == "incomplete" and snapshot.cleanup_issues
        else:
            assert not snapshot.cleanup_issues
        record_property("worker_failure", json.dumps({**facts, "failure": failure,
                        "cleanup_issues": [issue.code for issue in snapshot.cleanup_issues]}))
    finally:
        try:
            if failure == "workload-cleanup":
                with pytest.raises(CleanupError, match="incomplete"):
                    executor.close(cancel=True, timeout=20)
            else:
                executor.close(cancel=True, timeout=20)
        finally:
            repo.close(flush=False)
            control.close()
