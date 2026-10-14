"""Native CPU references and bounded maintained XGBoost adapter qualification.

Explicit execution requires the exact candidate stack; missing packages fail rather
than skip. Native imports occur only in fixtures/test bodies after collection.
"""

import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys

import pytest


@pytest.fixture(scope="module")
def native_xgb():
    """Admit the qualified candidate versions before importing native code."""
    expected = {
        "xgboost": "3.4.2", "scikit-learn": "1.9.0", "numpy": "2.1.3",
        "scipy": "1.18.0", "pandas": "3.0.5",
    }
    assert sys.version_info[:2] == (3, 12), "native reference requires Python 3.12"
    for package, required in expected.items():
        assert version(package) == required, f"{package} requires {required} for qualification"
    import numpy as np
    import xgboost as xgb

    return np, xgb, {**expected, "python": platform.python_version(),
                     "platform": platform.platform()}


@pytest.fixture
def native_reference(native_xgb, request):
    """Fit bounded seeded binary/multiclass/regression references with NaN features."""
    np, xgb, versions = native_xgb
    task = request.param
    rng = np.random.default_rng(1729)
    x = rng.normal(size=(192, 3)).astype(np.float32)
    if task == "binary":
        y = (x[:, 0] > 0).astype(np.int64)
    elif task == "multiclass":
        y = np.argmax(np.column_stack((x[:, 0], -x[:, 0], 1.5 * x[:, 1])), axis=1)
        y = y.astype(np.int64)
    else:
        y = (3 * x[:, 0] - 0.7 * x[:, 1] + rng.normal(0, 0.05, 192)).astype(np.float32)
    x[::13, 2] = np.nan
    options = dict(
        booster="gbtree", tree_method="hist", device="cpu", n_jobs=1,
        random_state=0, verbosity=0, n_estimators=48, max_depth=3, learning_rate=0.15,
    )
    if task == "regression":
        model = xgb.XGBRegressor(objective="reg:squarederror", **options)
    else:
        objective = "binary:logistic" if task == "binary" else "multi:softprob"
        model = xgb.XGBClassifier(objective=objective, **options)
    model.fit(x[:128], y[:128], verbose=False)
    return task, model, x, y, versions


@pytest.mark.parametrize("native_reference", ["binary", "multiclass", "regression"], indirect=True)
def test_native_xgb_quality_and_ubjson_reopen(native_reference, native_xgb, tmp_path, record_property):
    """Model IO retains predictions and fitted facts; byte identity is measured separately."""
    np, xgb, _ = native_xgb
    task, model, x, y, versions = native_reference
    held_x, held_y = x[128:], y[128:]
    prediction = model.predict(held_x)
    assert prediction.shape == (64,)
    assert model.n_features_in_ == 3
    config = json.loads(model.get_booster().save_config())["learner"]
    assert config["generic_param"]["device"] == "cpu"
    assert config["generic_param"]["n_jobs"] == "1"
    assert config["gradient_booster"]["name"] == "gbtree"
    assert config["gradient_booster"]["gbtree_train_param"]["tree_method"] == "hist"
    objective = {"binary": "binary:logistic", "multiclass": "multi:softprob",
                 "regression": "reg:squarederror"}[task]
    assert config["objective"]["name"] == objective

    if task == "regression":
        assert prediction.dtype == np.float32
        quality = float(np.mean((prediction - held_y) ** 2))
        baseline = float(np.mean((y[:128].mean() - held_y) ** 2))
        assert quality < baseline / 2
    else:
        classes = 2 if task == "binary" else 3
        np.testing.assert_array_equal(model.classes_, np.arange(classes, dtype=np.int64))
        assert prediction.dtype == model.classes_.dtype == np.int64
        probability = model.predict_proba(held_x)
        assert probability.shape == (64, classes)
        assert probability.dtype == np.float32
        assert np.isfinite(probability).all()
        assert ((probability >= 0) & (probability <= 1)).all()
        np.testing.assert_allclose(probability.sum(axis=1), 1, rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(prediction, probability.argmax(axis=1))
        quality = float(np.mean(prediction == held_y))
        baseline = float(np.bincount(held_y, minlength=classes).max() / held_y.size)
        assert quality >= baseline + 0.10

    path = tmp_path / "model.ubj"
    model.save_model(path)
    payload = path.read_bytes()
    assert payload.startswith(b"{")
    assert b"scikit_learn" in payload
    reopened = type(model)(n_jobs=1)
    reopened.load_model(path)
    buffered = type(model)(n_jobs=1)
    buffered.load_model(bytearray(payload))
    max_error = 0.0
    for restored in (reopened, buffered):
        assert restored.n_features_in_ == model.n_features_in_
        actual = restored.predict(held_x)
        assert actual.dtype == prediction.dtype
        np.testing.assert_allclose(actual, prediction, rtol=1e-6, atol=1e-6)
        assert restored.predict(held_x[:1]).shape == (1,)
        if task != "regression":
            np.testing.assert_array_equal(restored.classes_, model.classes_)
            assert restored.classes_.dtype == np.int64
            actual = restored.predict_proba(held_x)
            assert actual.dtype == np.float32
            assert restored.predict_proba(held_x[:1]).shape == (1, classes)
            np.testing.assert_allclose(actual, probability, rtol=1e-6, atol=1e-6)
            max_error = max(max_error, float(np.max(np.abs(actual - probability))))
        else:
            max_error = max(max_error, float(np.max(np.abs(actual - prediction))))
    second_path = tmp_path / "resaved.ubj"
    reopened.save_model(second_path)
    second = second_path.read_bytes()
    record_property("native_evidence", json.dumps({
        "versions": versions, "task": task, "quality": quality, "baseline": baseline,
        "max_reopen_error": max_error, "payload_bytes": len(payload),
        "save_load_save_equal": payload == second,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "resaved_sha256": hashlib.sha256(second).hexdigest(),
    }, sort_keys=True))


def test_native_xgb_failure_boundaries(native_xgb):
    """Native unfitted, width, infinity and label failures inform future admission."""
    np, xgb, _ = native_xgb
    from sklearn.exceptions import NotFittedError

    model = xgb.XGBClassifier(n_estimators=2, max_depth=1, n_jobs=1,
                              tree_method="hist", device="cpu", random_state=0, verbosity=0)
    x = np.array([[-1.0, 0.0], [-0.5, 1.0], [0.5, 0.0], [1.0, 1.0]], dtype=np.float32)
    y = np.array([0, 0, 1, 1], dtype=np.int64)
    with pytest.raises(NotFittedError):
        model.predict(x)
    with pytest.raises(ValueError, match="Invalid classes"):
        model.fit(x, y + 1, verbose=False)
    model.fit(x, y, verbose=False)
    with pytest.raises(ValueError, match="Feature shape mismatch"):
        model.predict(x[:, :1])
    invalid = x.copy()
    invalid[0, 0] = np.inf
    with pytest.raises(xgb.core.XGBoostError, match="inf"):
        model.fit(invalid, y, verbose=False)
    with pytest.raises(xgb.core.XGBoostError):
        model.load_model(bytearray(b"not a model"))


def test_native_xgb_rejects_nonfinite_regression_targets(native_xgb):
    """The native regression objective rejects both NaN and infinite labels."""
    np, xgb, _ = native_xgb
    x = np.arange(64, dtype=np.float32).reshape(32, 2)
    for invalid in (np.nan, np.inf):
        y = np.arange(32, dtype=np.float32)
        y[0] = invalid
        candidate = xgb.XGBRegressor(
            objective="reg:squarederror", tree_method="hist", device="cpu",
            n_jobs=1, n_estimators=2, max_depth=1, random_state=0, verbosity=0,
        )
        with pytest.raises(xgb.core.XGBoostError, match="Label contains"):
            candidate.fit(x, y, verbose=False)


def test_adapter_cpu_dense_fit_and_prediction(native_xgb):
    """The maintained adapter fits without retired Meta/compute preparation."""
    np, _, _ = native_xgb
    from dryml.models.xgb import ClassifierModel

    x = np.array([[-1.0], [-0.5], [0.5], [1.0]], dtype=np.float32)
    y = np.array([0, 0, 1, 1], dtype=np.int64)
    model = ClassifierModel(n_estimators=4, max_depth=1, min_child_weight=0)
    definition = model.definition
    assert model.fit(x, y) is model
    assert model.definition == definition
    assert model(x).shape == (4, 2)
    assert model(x).dtype == np.float32
    np.testing.assert_array_equal(model.predict(x), y)


@pytest.mark.parametrize("native_reference", ["binary", "multiclass", "regression"], indirect=True)
def test_adapter_quality_parity_and_authored_batch_exposure(native_reference, native_xgb, monkeypatch):
    """Plain and short authored batches submit identical ordered arrays and exact N."""
    np, _, _ = native_xgb
    from dryml.data import ArrayDataset, Batch, as_supervised
    from dryml.models import Experiment
    from dryml.models.xgb import ClassifierModel, RegressionModel, BasicTraining

    task, reference, x, y, _ = native_reference
    cls = RegressionModel if task == "regression" else ClassifierModel
    data = as_supervised(ArrayDataset((x[:128], y[:128])), 0, 1)
    models = []
    for authored in (data, Batch(data, 17)):
        model = cls(n_estimators=48, max_depth=3, learning_rate=0.15)
        before = model.definition
        fit = model._fit_candidate
        submitted = []

        def capture(actual_x, actual_y, **kwargs):
            submitted.append(len(actual_x))
            np.testing.assert_array_equal(actual_x, x[:128])
            np.testing.assert_array_equal(actual_y, y[:128])
            assert (actual_x.dtype, actual_y.dtype) == (x.dtype, y.dtype)
            return fit(actual_x, actual_y, **kwargs)

        monkeypatch.setattr(model, "_fit_candidate", capture)
        exp = Experiment(model, BasicTraining(), train_data=authored)
        assert exp.train_fn(exp) is model.mdl
        assert submitted == [128]
        assert model.definition == before
        assert (exp.state.step, exp.state.epoch, exp.state.examples_seen, exp.state.next_batch) == (1, 1, 128, 0)
        assert exp.state.stage_observation().training_loss is None
        assert exp.state.pending_epoch_postlude is None
        assert model.n_features_in_ == 3
        actual = model.predict(x[128:])
        expected = reference.predict(x[128:])
        assert actual.dtype == expected.dtype
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
        if task == "regression":
            assert float(np.mean((actual - y[128:]) ** 2)) < float(np.mean((y[:128].mean() - y[128:]) ** 2)) / 2
            np.testing.assert_array_equal(model(x[128:]), actual)
        else:
            k = 2 if task == "binary" else 3
            np.testing.assert_array_equal(model.classes_, np.arange(k, dtype=np.int64))
            probability = model.predict_proba(x[128:])
            assert probability.shape == (64, k) and probability.dtype == np.float32
            np.testing.assert_allclose(probability, reference.predict_proba(x[128:]), rtol=1e-6, atol=1e-6)
            np.testing.assert_array_equal(model(x[128:]), probability)
            np.testing.assert_array_equal(actual, probability.argmax(axis=1))
            baseline = float(np.bincount(y[128:]).max() / 64)
            assert float(np.mean(actual == y[128:])) >= baseline + .10
        models.append(model)
    assert models[0]._accepted_fit is not models[1]._accepted_fit
    assert models[0].mdl is not models[1].mdl
    np.testing.assert_array_equal(models[0](x[128:]), models[1](x[128:]))


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_adapter_selected_calls_prepared_map_and_refit_facts(native_xgb, task, monkeypatch):
    """Preparation is native-free, same-fact refits stay live, changed facts refuse reuse."""
    np, _, _ = native_xgb
    from dryml.core import TensorSpec
    from dryml.data import ArrayDataset, Batch, Map
    from dryml.methods import MethodError
    from dryml.models.xgb import ClassifierModel, RegressionModel

    x = np.arange(60, dtype=np.float32).reshape(20, 3)
    y = (np.arange(20) % 2).astype(np.int64) if task == "classification" else np.arange(20, dtype=np.float32)
    cls = ClassifierModel if task == "classification" else RegressionModel
    model = cls(n_estimators=4, max_depth=2, min_child_weight=0)
    model.fit(x, y)
    spec = TensorSpec("float32", shape=(3,), backend="numpy")
    source = ArrayDataset(x)
    with monkeypatch.context() as guard:
        guard.setattr(model, "_new_native", lambda *a: pytest.fail("native construction during preparation"))
        guard.setattr(model.mdl, "predict", lambda *a: pytest.fail("prediction during preparation"))
        guard.setattr(model.mdl, "predict_proba", lambda *a: pytest.fail("prediction during preparation"), raising=False)
        element = model.find_implementation(spec)
        batched = model.find_implementation(spec.with_batch(4))
        model.learn(spec.with_batch(4))
        mapped = Map(source, model).prepare()
        batch_map = Map(Batch(source, 7), model).prepare()
        assert mapped.spec.dtype.name == batch_map.spec.dtype.name == "float32"
        assert mapped.spec.shape == (() if task == "regression" else (2,))
    raw = model.predict_proba(x) if task == "classification" else model.predict(x)
    assert element(x[0]).shape == raw[0].shape
    assert element(x[0]).dtype == np.float32
    np.testing.assert_array_equal(element(x[0]), raw[0])
    np.testing.assert_array_equal(batched(x[:4]), raw[:4])
    np.testing.assert_array_equal(model(x[:4]), raw[:4])
    np.testing.assert_array_equal(np.asarray(_collect_prepared(mapped)), raw)
    np.testing.assert_array_equal(np.concatenate(_collect_prepared(batch_map)), raw)
    before = raw.copy()
    model.fit(x, 1 - y if task == "classification" else y + 100)
    current = model.predict_proba(x) if task == "classification" else model.predict(x)
    assert not np.array_equal(before, current)
    np.testing.assert_array_equal(np.asarray(_collect_prepared(mapped)), current)
    np.testing.assert_array_equal(model(x[:4]), current[:4])
    if task == "classification":
        model.fit(x, (np.arange(20) % 3).astype(np.int64))
    else:
        model.fit(x[:, :2], y)
    for invoke in (lambda: element(x[0]), lambda: _collect_prepared(mapped), lambda: model(x[:4])):
        with pytest.raises((MethodError, ValueError), match="re-prepare|width"):
            invoke()
    changed_source = source if task == "classification" else ArrayDataset(x[:, :2])
    remapped = Map(changed_source, model).prepare()
    actual_x = x if task == "classification" else x[:, :2]
    np.testing.assert_array_equal(np.asarray(_collect_prepared(remapped)), model.predict_proba(actual_x) if task == "classification" else model.predict(actual_x))


def _collect_prepared(dataset):
    with dataset.iterator() as cursor:
        return list(cursor)


def test_adapter_explicit_spec_checked_at_fit_and_prefit_preparation(native_xgb):
    np, _, _ = native_xgb
    from dryml.core import TensorSpec
    from dryml.data import ArrayDataset, Map
    from dryml.models.xgb import ClassifierModel

    x = np.arange(30, dtype=np.float32).reshape(10, 3)
    y = (np.arange(10) % 2).astype(np.int64)
    model = ClassifierModel(output_spec=TensorSpec("float32", shape=(2,), backend="numpy"), n_estimators=2)
    mapped = Map(ArrayDataset(x), model).prepare()
    model.fit(x, y)
    np.testing.assert_array_equal(np.asarray(_collect_prepared(mapped)), model.predict_proba(x))
    prior = model._accepted_fit
    with pytest.raises(ValueError, match="output_spec"):
        model.fit(x, (np.arange(10) % 3).astype(np.int64))
    assert model._accepted_fit is prior


def test_adapter_invalid_input_preserves_previous_bundle(native_xgb):
    np, _, _ = native_xgb
    from dryml.models.xgb import ClassifierModel, RegressionModel

    x = np.arange(24, dtype=np.float32).reshape(12, 2)
    y = (np.arange(12) % 2).astype(np.int64)
    model = ClassifierModel(n_estimators=2).fit(x, y)
    prior = model._accepted_fit
    prediction = model(x)
    bad_x = x.copy()
    bad_x[0, 0] = np.inf
    for features, targets in ((x[:0], y[:0]), (x, y + 1), (x, y * 2), (x, np.zeros_like(y)),
                              (x, y.astype(np.float32)), (x, y[:-1]), (bad_x, y)):
        with pytest.raises((TypeError, ValueError)):
            model.fit(features, targets)
        assert model._accepted_fit is prior
    for features in (x[:0], x[:, :1], bad_x, x[0], np.ma.array(x), x.astype(np.complex64)):
        with pytest.raises((TypeError, ValueError)):
            model.predict(features)
    np.testing.assert_array_equal(model(x), prediction)
    reg = RegressionModel(n_estimators=2).fit(x, y.astype(np.float32))
    accepted = reg._accepted_fit
    for nonfinite in (np.nan, np.inf):
        targets = y.astype(np.float32)
        targets[0] = nonfinite
        with pytest.raises(ValueError, match="nonfinite"):
            reg.fit(x, targets)
        assert reg._accepted_fit is accepted


@pytest.mark.parametrize("stage,error_type", [("native", RuntimeError), ("validate", RuntimeError),
                                              ("encode", RuntimeError), ("encode", KeyboardInterrupt)])
def test_adapter_candidate_failure_keeps_native_facts_bytes_and_counts(native_xgb, monkeypatch, stage, error_type):
    np, _, _ = native_xgb
    from dryml.data import ArrayDataset, as_supervised
    from dryml.models import Experiment
    from dryml.models.xgb import RegressionModel, BasicTraining

    x = np.arange(20, dtype=np.float32).reshape(10, 2)
    y = np.arange(10, dtype=np.float32)
    exp = Experiment(RegressionModel(n_estimators=2), BasicTraining(),
                     train_data=as_supervised(ArrayDataset((x, y)), 0, 1))
    exp.train_fn(exp)
    prior, state = exp.model._accepted_fit, exp.state
    prediction = exp.model(x)
    error = error_type(stage)

    def fail(*args, **kwargs):
        raise error

    if stage == "validate":
        monkeypatch.setattr(exp.model, "_validate_native", fail)
    else:
        target = "fit" if stage == "native" else "save_model"
        monkeypatch.setattr(type(exp.model.mdl), target, fail)
    for action in (lambda: exp.train_fn(exp), lambda: exp.model.fit(x, y + 1)):
        with pytest.raises(error_type) as caught:
            action()
        assert caught.value is error
        assert exp.model._accepted_fit is prior and exp.state is state
        assert (state.step, state.epoch, state.examples_seen) == (1, 1, 10)
        np.testing.assert_array_equal(exp.model(x), prediction)


def test_adapter_invalid_native_predictions_cannot_be_accepted(native_xgb, monkeypatch):
    """A native fit return alone is insufficient: candidate output facts must hold."""
    np, _, _ = native_xgb
    from dryml.models.xgb import ClassifierModel, RegressionModel

    x = np.arange(24, dtype=np.float32).reshape(12, 2)
    y = (np.arange(12) % 2).astype(np.int64)
    for cls, prediction_method, bad in (
        (ClassifierModel, "predict_proba", np.full((12, 2), 2, dtype=np.float32)),
        (RegressionModel, "predict", np.full(12, np.nan, dtype=np.float32)),
    ):
        model = cls(n_estimators=2).fit(x, y)
        prior = model._accepted_fit
        make_native = model._new_native

        def invalid_candidate(classes):
            candidate = make_native(classes)
            monkeypatch.setattr(candidate, prediction_method, lambda inputs: bad)
            return candidate

        monkeypatch.setattr(model, "_new_native", invalid_candidate)
        with pytest.raises(ValueError, match="predictions|probabilities"):
            model.fit(x, y)
        assert model._accepted_fit is prior


def test_adapter_cleanup_precedes_single_native_encoding(native_xgb, monkeypatch, tmp_path):
    """Real native bytes are encoded once only after cleanup permits acceptance."""
    np, _, _ = native_xgb
    from dryml.data import ArrayDataset, as_supervised
    from dryml.models import Experiment
    from dryml.models.xgb import RegressionModel, BasicTraining

    x = np.arange(20, dtype=np.float32).reshape(10, 2)
    y = np.arange(10, dtype=np.float32)
    model = RegressionModel(n_estimators=2).fit(x, y)
    exp = Experiment(model, BasicTraining(), train_data=as_supervised(ArrayDataset((x, y + 1)), 0, 1))
    prior, state = model._accepted_fit, exp.state
    events = []
    save_model = type(model.mdl).save_model

    def encode(native, path):
        assert events == ["prep_eval"]
        assert model._accepted_fit is prior and exp.state is state
        events.append("encode")
        return save_model(native, path)

    def reject_cleanup():
        events.append("prep_eval")
        raise RuntimeError("cleanup rejected candidate")

    monkeypatch.setattr(type(model.mdl), "save_model", encode)
    monkeypatch.setattr(model, "prep_eval", reject_cleanup)
    with pytest.raises(RuntimeError, match="cleanup rejected"):
        exp.train_fn(exp)
    assert events == ["prep_eval"]
    assert model._accepted_fit is prior and exp.state is state
    assert (state.step, state.epoch, state.examples_seen) == (0, 0, 0)

    events.clear()
    monkeypatch.setattr(model, "prep_eval", lambda: events.append("prep_eval"))
    exp.train_fn(exp)
    assert events == ["prep_eval", "encode"]
    assert model._accepted_fit is not prior
    assert (exp.state.step, exp.state.epoch, exp.state.examples_seen) == (1, 1, 10)
    model.save_state_to_dir(tmp_path, codec="xgboost1")
    assert events == ["prep_eval", "encode"]
    assert Path(tmp_path, "model.ubj").read_bytes() == model._accepted_fit.payload


@pytest.mark.usefixtures("fixed_managed_snapshot_environment")
def test_adapter_local_experiment_publishes_native_model_and_terminal_history(native_xgb, tmp_path):
    """One local happy path connects the real codec to the unchanged managed owner."""
    np, _, _ = native_xgb
    from dryml.core import Repo, StateRef
    from dryml.core.store.dir import DirStore
    from dryml.data import ArrayDataset, Batch, as_supervised
    from dryml.managed import ManagedConfig
    from dryml.models import Experiment, ExperimentData
    from dryml.models.xgb import RegressionModel, BasicTraining

    x = np.arange(20, dtype=np.float32).reshape(10, 2)
    y = np.arange(10, dtype=np.float32)
    repo = Repo(DirStore(tmp_path / "store"))
    exp = Experiment(RegressionModel(n_estimators=2), BasicTraining(),
                     train_data=Batch(as_supervised(ArrayDataset((x, y)), 0, 1), 3),
                     repo=repo, checkpoint_every_steps=1)
    terminal = []
    final = exp.train(managed=ManagedConfig(
        state_repo=repo, callbacks=[lambda obj, context: terminal.append(context.checkpoint_state_ref)]))
    assert isinstance(final, StateRef) and terminal == [final]
    assert exp.train.status(state_repo=repo).final_state_ref == final
    history = ExperimentData.find(final.object_projection(), repo=repo).data
    assert len(history) == 1
    assert history.iloc[0].evaluation_status == "completed"
    assert history.iloc[0].state_ref == final and history.iloc[0].examples_seen == 10
    fresh = repo.load_state_ref(final, reuse_live="never")
    assert fresh.model is not exp.model
    assert (fresh.state.step, fresh.state.epoch, fresh.state.examples_seen) == (1, 1, 10)
    assert fresh.model._accepted_fit.payload == exp.model._accepted_fit.payload
    np.testing.assert_array_equal(fresh.model(x), exp.model(x))


@pytest.mark.usefixtures("fixed_snapshot_environment")
@pytest.mark.parametrize("task", ["binary", "multiclass", "regression", "unfitted"])
def test_adapter_exact_repo_restore_and_retained_payload_identity(native_xgb, tmp_path, monkeypatch, task):
    """Fresh exact Repo load, not live-cache reuse, reproduces payload and predictions."""
    np, _, _ = native_xgb
    from dryml.core import Repo
    from dryml.core.store.dir import DirStore
    from dryml.models.xgb import ClassifierModel, RegressionModel

    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    model = (RegressionModel if task == "regression" else ClassifierModel)(n_estimators=3, max_depth=2)
    x = np.arange(60, dtype=np.float32).reshape(20, 3)
    y = np.arange(20, dtype=np.float32) if task == "regression" else (np.arange(20) % (3 if task == "multiclass" else 2)).astype(np.int64)
    unfitted = model.save(repo=repo)
    if task != "unfitted":
        model.fit(x, y)
    state = model.save(repo=repo)
    payload_dir = Path(store.open_local_state(state, next(iter(state.states))).handle, "data")
    inventory = {path.name: path.read_bytes() for path in payload_dir.iterdir()}
    assert set(inventory) == ({"metadata.json"} if task == "unfitted" else {"metadata.json", "model.ubj"})
    assert all(value.startswith("xgboost1-") for value in state.states.values())
    with monkeypatch.context() as guard:
        if task != "unfitted":
            def no_probe(*args, **kwargs):
                pytest.fail("native prediction during restore")

            guard.setattr(type(model.mdl), "predict", no_probe)
            guard.setattr(type(model.mdl), "predict_proba", no_probe, raising=False)
        reopened = Repo(store).load_state_ref(state, reuse_live="never")
    assert reopened is not model
    assert reopened.definition == model.definition
    assert reopened.call_mode == "eager" and reopened.default_batched is None
    if task != "unfitted":
        np.testing.assert_array_equal(reopened(x), model(x))
        assert reopened._accepted_fit.payload == model._accepted_fit.payload
        assert reopened._accepted_fit.metadata == model._accepted_fit.metadata
        monkeypatch.setattr(reopened.mdl, "save_model", lambda *a: pytest.fail("native re-encoding"))
    else:
        assert reopened.mdl is None
        with pytest.raises(ValueError, match="unfitted"):
            reopened(x)
    resaved = reopened.save(repo=repo)
    assert resaved.states == state.states
    second_dir = Path(store.open_local_state(resaved, next(iter(resaved.states))).handle, "data")
    assert {path.name: path.read_bytes() for path in second_dir.iterdir()} == inventory
    if task != "unfitted":
        repo.restore_state_ref_into(model, unfitted)
        assert model._accepted_fit is None and model.mdl is None
        with pytest.raises(ValueError, match="unfitted"):
            model.n_features_in_
        with pytest.raises(ValueError, match="unfitted"):
            model.classes_


def test_adapter_restore_malformed_metadata_is_candidate_first(native_xgb, tmp_path):
    np, xgb, _ = native_xgb
    from dryml.models.xgb import ClassifierModel

    x = np.arange(24, dtype=np.float32).reshape(12, 2)
    y = (np.arange(12) % 2).astype(np.int64)
    model = ClassifierModel(n_estimators=2).fit(x, y)
    model.save_state_to_dir(tmp_path, codec="xgboost1")
    prior = model._accepted_fit
    original = json.loads(prior.metadata)
    for change in ({"feature_width": 3}, {"feature_width": True}, {"classes": [0, 1, 2]},
                   {"classes": [1, 2]}, {"task": "regression"}, {"version": 2}, {"extra": 1},
                   {"writer_versions": {}}, {"native_sha256": "bad"}, {"call_dtype": "float64"},
                   {"fitted": False}, {"fitted": 1}):
        Path(tmp_path, "metadata.json").write_bytes(json.dumps({**original, **change}, sort_keys=True, separators=(",", ":")).encode())
        with pytest.raises(ValueError, match="metadata|class|feature"):
            model.restore_state_from_dir(tmp_path, codec="xgboost1")
        assert model._accepted_fit is prior
    Path(tmp_path, "metadata.json").write_bytes(prior.metadata + b"\n")
    with pytest.raises(ValueError, match="canonical"):
        model.restore_state_from_dir(tmp_path, codec="xgboost1")
    payload = b"not a native model"
    Path(tmp_path, "model.ubj").write_bytes(payload)
    envelope = {**original, "native_sha256": hashlib.sha256(payload).hexdigest()}
    Path(tmp_path, "metadata.json").write_bytes(json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode())
    with pytest.raises(xgb.core.XGBoostError):
        model.restore_state_from_dir(tmp_path, codec="xgboost1")
    assert model._accepted_fit is prior
    np.testing.assert_array_equal(model(x), prior.native.predict_proba(x))
    Path(tmp_path, "model.ubj").write_bytes(prior.payload)
    Path(tmp_path, "metadata.json").write_bytes(prior.metadata)
    Path(tmp_path, "model.text").write_bytes(b"legacy")
    with pytest.raises(ValueError, match="Legacy"):
        model.restore_state_from_dir(tmp_path, codec="xgboost1")
    with pytest.raises(ValueError, match="codec"):
        model.restore_state_from_dir(tmp_path, codec="pkl")
    assert model._accepted_fit is prior


@pytest.mark.usefixtures("fixed_snapshot_environment")
def test_adapter_failed_entered_restore_obeys_core_invalidation(native_xgb, tmp_path, monkeypatch):
    np, _, _ = native_xgb
    from dryml.core import Repo
    from dryml.core.repo import RepoLoadError, RepoSaveError
    from dryml.core.store.dir import DirStore
    from dryml.models.xgb import RegressionModel

    repo = Repo(DirStore(tmp_path / "store"))
    x = np.arange(20, dtype=np.float32).reshape(10, 2)
    model = RegressionModel(n_estimators=2).fit(x, np.arange(10, dtype=np.float32))
    state = model.save(repo=repo)
    prior = model._accepted_fit

    def fail(candidate):
        raise ValueError("entered native metadata rejection")

    monkeypatch.setattr(model, "_validate_native", fail)
    with pytest.raises(RepoLoadError, match="Targeted exact restore failed"):
        repo.restore_state_ref_into(model, state)
    assert model._accepted_fit is prior
    assert model._restore_failed and model.last_state_ref is None
    with pytest.raises(RepoSaveError, match="invalidated"):
        model.save(repo=repo)
    fresh = repo.load_state_ref(state, reuse_live="never")
    assert fresh is not model and not fresh._restore_failed
    np.testing.assert_array_equal(fresh(x), prior.native.predict(x))
