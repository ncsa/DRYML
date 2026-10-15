"""Array-native references and bounded maintained OLS/Logit qualification.

Exact candidate dependencies are mandatory at execution, optional at collection.
Full Results payloads deliberately retain training data; pickle byte stability
is recorded independently from numerical and inspection equivalence.
"""

import gc
import hashlib
from importlib.metadata import version
import io
import json
from pathlib import Path
import platform
import subprocess
import sys
import warnings
import weakref

import pytest


@pytest.fixture(scope="module")
def native_stats():
    """Require matching candidate readers without importing sklearn or mutating warnings."""
    expected = {"statsmodels": "0.15.0", "numpy": "2.1.3",
                "scipy": "1.18.0", "pandas": "3.0.5"}
    assert sys.version_info[:2] == (3, 12), "native reference requires Python 3.12"
    for package, required in expected.items():
        assert version(package) == required, f"{package} requires {required} for qualification"
    import numpy as np
    from statsmodels.discrete.discrete_model import Logit
    from statsmodels.regression.linear_model import OLS
    from statsmodels.iolib.smpickle import load_pickle

    return np, OLS, Logit, load_pickle, {
        **expected, "python": platform.python_version(), "platform": platform.platform(),
    }


@pytest.fixture
def ols_reference(native_stats, request):
    """Return conditioned or rank-deficient 96-row OLS and held-out design arrays."""
    np, OLS, _, _, _ = native_stats
    intercept, deficient = request.param
    rng = np.random.default_rng(20261014)
    x = rng.normal(size=(160, 2))
    if deficient:
        x = np.column_stack((x, x[:, 0], np.ones(160)))
    design = np.column_stack((np.ones(160), x)) if intercept else x.copy()
    beta = np.arange(1, design.shape[1] + 1, dtype=np.float64) / 2
    y = design @ beta + rng.normal(0, 0.025, 160)
    model = OLS(y[:96], design[:96], missing="raise")
    if deficient:
        from statsmodels.tools.sm_exceptions import SingularMatrixWarning

        with pytest.warns(SingularMatrixWarning, match="rank-deficient"):
            result = model.fit(method="pinv")
    else:
        result = model.fit(method="pinv")
    return result, x, design, y


@pytest.fixture
def logit_reference(native_stats, request):
    """Fit a nonseparable binary reference with explicit intercept-on/off design."""
    np, _, Logit, _, _ = native_stats
    intercept = request.param
    rng = np.random.default_rng(42)
    # Opposing labels at identical feature values preclude complete separation.
    base = rng.normal(size=(64, 2))
    x_train = np.repeat(base, 2, axis=0)
    y_train = (x_train[:, 0] > 0).astype(np.float64)
    y_train[::10] = 1 - y_train[::10]
    held_x = rng.normal(size=(64, 2))
    held_y = (held_x[:, 0] > 0).astype(np.int64)
    design = np.column_stack((np.ones(128), x_train)) if intercept else x_train.copy()
    held_design = np.column_stack((np.ones(64), held_x)) if intercept else held_x.copy()
    model = Logit(y_train, design, missing="raise")
    model.raise_on_perfect_prediction = True
    result = model.fit(method="newton", maxiter=100, tol=1e-8, disp=False, full_output=True)
    return result, design, y_train, held_design, held_y


@pytest.mark.parametrize("ols_reference", [(False, False), (True, False),
                                           (False, True), (True, True)], indirect=True,
                         ids=["no-intercept", "intercept", "duplicate", "duplicate-intercept"])
def test_native_ols_rank_intercept_and_quality(ols_reference, native_stats, record_property):
    """Pseudoinverse policy accepts finite rank-deficient fits without detecting constants."""
    np, _, _, _, _ = native_stats
    result, x, design, y = ols_reference
    expected, _, rank, _ = np.linalg.lstsq(design[:96], y[:96], rcond=None)
    assert result.model.rank == rank
    assert np.isfinite(result.params).all()
    assert result.model.exog.shape == (96, design.shape[1])
    if design.shape[1] > x.shape[1]:
        np.testing.assert_array_equal(design[:, 0], 1)
    np.testing.assert_allclose(result.params, expected, rtol=1e-8, atol=1e-8)
    prediction = result.predict(design[96:])
    assert prediction.shape == (64,) and prediction.dtype == np.float64
    np.testing.assert_allclose(prediction, design[96:] @ expected, rtol=1e-8, atol=1e-8)
    assert result.predict(design[96:97]).shape == (1,)
    mse = float(np.mean((prediction - y[96:]) ** 2))
    baseline = float(np.mean((y[:96].mean() - y[96:]) ** 2))
    assert mse < baseline / 2
    record_property("native_evidence", json.dumps({
        "rank": int(rank), "design_width": design.shape[1], "mse": mse,
        "baseline": baseline, "max_lstsq_error": float(np.max(np.abs(result.params - expected))),
    }, sort_keys=True))


@pytest.mark.parametrize("logit_reference", [False, True], indirect=True)
def test_native_logit_convergence_probabilities_and_quality(logit_reference, native_stats, record_property):
    """Newton success needs explicit convergence and finite params, not fit return alone."""
    np, _, _, _, _ = native_stats
    result, _, _, held_design, held_y = logit_reference
    assert result.mle_retvals["converged"] is True
    assert result.model.raise_on_perfect_prediction is True
    assert np.isfinite(result.params).all()
    positive = result.predict(held_design)
    expected = 1 / (1 + np.exp(-(held_design @ result.params)))
    np.testing.assert_allclose(positive, expected, rtol=1e-8, atol=1e-8)
    probability = np.column_stack((1 - positive, positive))
    labels = (positive >= 0.5).astype(np.int64)
    assert probability.shape == (64, 2) and probability.dtype == np.float64
    assert labels.shape == (64,) and labels.dtype == np.int64
    assert np.isfinite(probability).all()
    assert ((probability >= 0) & (probability <= 1)).all()
    np.testing.assert_allclose(probability.sum(axis=1), 1, rtol=1e-8, atol=1e-8)
    assert result.predict(held_design[:1]).shape == (1,)
    assert result.predict(np.zeros((1, held_design.shape[1])))[0] == 0.5
    assert (result.predict(np.zeros((1, held_design.shape[1]))) >= 0.5).astype(np.int64)[0] == 1
    accuracy = float(np.mean(labels == held_y))
    baseline = float(np.bincount(held_y).max() / held_y.size)
    assert accuracy >= baseline + 0.10
    record_property("native_evidence", json.dumps({
        "accuracy": accuracy, "baseline": baseline, "iterations": result.mle_retvals["iterations"],
        "max_probability_error": float(np.max(np.abs(positive - expected))),
    }, sort_keys=True))


@pytest.mark.parametrize("family", ["ols", "logit"])
def test_native_full_results_pickle_retention_and_byte_stability(native_stats, family, tmp_path, record_property):
    """Full save/load preserves data and inspection; diagnostic caches can alter bytes."""
    np, OLS, Logit, load_pickle, versions = native_stats
    rng = np.random.default_rng(17)
    design = np.column_stack((np.ones(96), rng.normal(size=(96, 2))))
    y = design @ np.array([0.5, 1.0, -0.25]) + rng.normal(0, 0.1, 96)
    if family == "ols":
        result = OLS(y, design, missing="raise").fit(method="pinv")
    else:
        y = (rng.random(96) < 1 / (1 + np.exp(-y))).astype(np.float64)
        model = Logit(y, design, missing="raise")
        model.raise_on_perfect_prediction = True
        result = model.fit(method="newton", maxiter=100, tol=1e-8, disp=False, full_output=True)
        assert result.mle_retvals["converged"] is True
    params = result.params.copy()
    prediction = result.predict(design)
    exog = result.model.exog.copy()
    endog = result.model.endog.copy()
    before_cache = set(result._results._cache)

    first_path = tmp_path / "results.pkl"
    result.save(first_path, remove_data=False)
    payload = first_path.read_bytes()
    assert payload.startswith(b"\x80")
    assert set(result._results._cache) == before_cache
    np.testing.assert_array_equal(result.model.exog, exog)
    np.testing.assert_array_equal(result.model.endog, endog)
    np.testing.assert_array_equal(result.params, params)
    np.testing.assert_array_equal(result.predict(design), prediction)
    restored = load_pickle(io.BytesIO(payload))
    second_path = tmp_path / "resaved.pkl"
    restored.save(second_path, remove_data=False)
    second = second_path.read_bytes()

    np.testing.assert_array_equal(restored.model.exog, exog)
    np.testing.assert_array_equal(restored.model.endog, endog)
    np.testing.assert_array_equal(restored.model.data.orig_exog, exog)
    np.testing.assert_array_equal(restored.model.data.orig_endog, endog)
    np.testing.assert_allclose(restored.params, params, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(restored.predict(design), prediction, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(restored.bse, result.bse, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(restored.cov_params(), result.cov_params(), rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(restored.conf_int(), result.conf_int(), rtol=1e-8, atol=1e-8)
    assert np.isfinite(restored.llf)
    assert restored.nobs == 96
    # Native summaries stamp their rendering time, not a fitted scientific fact.
    def scientific_summary(result):
        return [line for line in str(result.summary()).splitlines()
                if not line.lstrip().startswith(("Date:", "Time:"))]

    assert scientific_summary(restored) == scientific_summary(result)
    inspected_path = tmp_path / "inspected.pkl"
    restored.save(inspected_path, remove_data=False)
    inspected = inspected_path.read_bytes()
    inspected_result = load_pickle(io.BytesIO(inspected))
    np.testing.assert_allclose(inspected_result.predict(design), prediction, rtol=1e-8, atol=1e-8)
    record_property("native_evidence", json.dumps({
        "versions": versions, "family": family, "payload_bytes": len(payload),
        "resaved_bytes": len(second), "inspected_bytes": len(inspected),
        "save_load_save_equal": payload == second,
        "after_inspection_equal": payload == inspected,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "resaved_sha256": hashlib.sha256(second).hexdigest(),
        "inspected_sha256": hashlib.sha256(inspected).hexdigest(),
        "retained_arrays": ["exog", "endog", "data.orig_exog", "data.orig_endog"],
        "inspection_cache_keys": sorted(restored._results._cache),
    }, sort_keys=True))


def test_native_logit_nonconvergence_is_a_returned_failure_fact(native_stats):
    """A short Newton fit can return finite Results while explicitly not converged."""
    np, _, Logit, _, _ = native_stats
    from statsmodels.tools.sm_exceptions import ConvergenceWarning

    x = np.tile(np.array([-2.0, -1.0, 1.0, 2.0]), 24)[:, None]
    y = np.tile(np.array([0.0, 0.0, 1.0, 1.0]), 24)
    y[::9] = 1 - y[::9]
    model = Logit(y, x, missing="raise")
    model.raise_on_perfect_prediction = True
    with pytest.warns(ConvergenceWarning):
        result = model.fit(method="newton", maxiter=1, tol=1e-8, disp=False, full_output=True)
    assert result.mle_retvals["converged"] is False
    assert np.isfinite(result.params).all()


def test_native_logit_perfect_prediction_raising_is_instance_local(native_stats):
    """Exact prediction triggers the private candidate flag, not a universal separation check."""
    np, _, Logit, _, _ = native_stats
    from statsmodels.tools.sm_exceptions import PerfectSeparationError, PerfectSeparationWarning

    x = np.tile(np.array([-1.0, 1.0]), 32)[:, None]
    y = (x[:, 0] > 0).astype(np.float64)
    candidate = Logit(y, x, missing="raise")
    other = Logit(y, x, missing="raise")
    assert other.raise_on_perfect_prediction is False
    candidate.raise_on_perfect_prediction = True
    filters = list(warnings.filters)
    # A deterministic perfect starting prediction avoids relying on divergent iterates.
    with pytest.raises(PerfectSeparationError):
        candidate.fit(start_params=np.array([30.0]), method="newton", maxiter=1,
                      tol=1e-8, disp=False, full_output=True)
    with pytest.warns(PerfectSeparationWarning):
        other._check_perfect_pred(np.array([30.0]))
    assert other.raise_on_perfect_prediction is False
    assert warnings.filters == filters


@pytest.mark.parametrize("family", ["ols", "logit"])
def test_native_statsmodels_missing_raise_never_drops_rows(native_stats, family):
    """NaN target/design admission fails rather than silently reducing nobs."""
    np, OLS, Logit, _, _ = native_stats
    from statsmodels.tools.sm_exceptions import MissingDataError

    cls = OLS if family == "ols" else Logit
    x = np.column_stack((np.ones(64), np.linspace(-1, 1, 64)))
    y = np.tile(np.array([0.0, 1.0]), 32)
    for branch in ("features", "targets"):
        invalid_x, invalid_y = x.copy(), y.copy()
        if branch == "features":
            invalid_x[0, 1] = np.nan
        else:
            invalid_y[0] = np.nan
        with pytest.raises(MissingDataError):
            cls(invalid_y, invalid_x, missing="raise")


def test_native_ols_missing_raise_is_not_a_finite_parameter_guard(native_stats):
    """Infinite targets can yield nonfinite parameters, requiring adapter validation."""
    np, OLS, _, _, _ = native_stats
    x = np.ones((64, 1), dtype=np.float64)
    y = np.ones(64, dtype=np.float64)
    y[0] = np.inf
    result = OLS(y, x, missing="raise").fit(method="pinv")
    assert result.nobs == 64
    assert not np.isfinite(result.params).all()


@pytest.mark.parametrize("ols_reference", [(False, False), (True, False),
                                           (False, True), (True, True)], indirect=True)
def test_adapter_ols_native_parity_quality_and_original_width(ols_reference, native_stats):
    """OLS retains pseudoinverse rank semantics and prepends exactly one constant."""
    np, _, _, _, _ = native_stats
    from dryml.models.statsmodels import RegressionModel

    reference, x, design, y = ols_reference
    model = RegressionModel(intercept=design.shape[1] > x.shape[1])
    definition = model.definition
    original_x, original_y = x.copy(), y.copy()
    assert model.fit(x[:96], y[:96]) is model
    assert model.definition == definition
    assert model.n_features_in_ == x.shape[1]
    np.testing.assert_array_equal(x, original_x)
    np.testing.assert_array_equal(y, original_y)
    np.testing.assert_allclose(model.results.params, reference.params, rtol=1e-8, atol=1e-8)
    np.testing.assert_array_equal(model.results.model.exog, design[:96])
    actual = model(x[96:])
    assert actual.shape == (64,) and actual.dtype == np.float64
    np.testing.assert_allclose(actual, reference.predict(design[96:]), rtol=1e-8, atol=1e-8)
    assert np.mean((actual - y[96:]) ** 2) < np.mean((y[:96].mean() - y[96:]) ** 2) / 2
    assert model.predict(x[96:97]).shape == (1,)
    with pytest.raises(ValueError, match="width"):
        model.predict(np.ones((1, x.shape[1] + 1)))


@pytest.mark.parametrize("logit_reference", [False, True], indirect=True)
def test_adapter_logit_native_parity_quality_and_threshold(logit_reference, native_stats):
    """Binary probabilities are float64 [0,1] columns; equality selects label one."""
    np, _, _, _, _ = native_stats
    from dryml.models.statsmodels import ClassifierModel

    reference, design, y, held_design, held_y = logit_reference
    intercept = design.shape[1] == 3
    x = design[:, 1:] if intercept else design
    held_x = held_design[:, 1:] if intercept else held_design
    model = ClassifierModel(intercept=intercept).fit(x, y)
    np.testing.assert_allclose(model.results.params, reference.params, rtol=1e-8, atol=1e-8)
    probability = model(held_x)
    assert probability.shape == (64, 2) and probability.dtype == np.float64
    np.testing.assert_allclose(probability[:, 1], reference.predict(held_design), rtol=1e-8, atol=1e-8)
    np.testing.assert_array_equal(model.predict_proba(held_x), probability)
    np.testing.assert_array_equal(model.classes_, np.array([0, 1], dtype=np.int64))
    labels = model.predict(held_x)
    assert labels.dtype == np.int64
    assert np.mean(labels == held_y) >= np.bincount(held_y).max() / len(held_y) + .10
    np.testing.assert_array_equal(labels, (probability[:, 1] >= .5).astype(np.int64))
    assert model.predict_proba(held_x[:1]).shape == (1, 2)
    if not intercept:
        np.testing.assert_array_equal(model.predict(np.zeros((1, 2))), [1])
    alternative = ClassifierModel(intercept=intercept, threshold=.8).fit(x, y)
    np.testing.assert_array_equal(alternative.predict(held_x), (probability[:, 1] >= .8).astype(np.int64))


def _small_stats_data(np, *, classification=False):
    rng = np.random.default_rng(17)
    x = rng.normal(size=(96, 2))
    y = .5 + x @ np.array([1., -.25]) + rng.normal(0, .1, 96)
    if classification:
        y = (rng.random(96) < 1 / (1 + np.exp(-y))).astype(np.float64)
    return x, y


@pytest.mark.parametrize("classification", [False, True])
def test_adapter_dataset_order_exposure_and_float64_copy(native_stats, classification, monkeypatch):
    """Plain/short authored batches submit exact ordered N without mutating inputs."""
    np, _, _, _, _ = native_stats
    from dryml.data import ArrayDataset, Batch, as_supervised
    from dryml.models import Experiment
    from dryml.models.statsmodels import ClassifierModel, RegressionModel, BasicTraining

    x, y = _small_stats_data(np, classification=classification)
    x = x.astype(np.float32)
    y = y.astype(np.int64 if classification else np.float32)
    data = as_supervised(ArrayDataset((x, y)), 0, 1)
    models = []
    for authored in (data, Batch(data, 17)):
        model = (ClassifierModel if classification else RegressionModel)(intercept=True)
        definition, fit = model.definition, model._fit_candidate
        submitted = []

        def capture(actual_x, actual_y):
            submitted.append(len(actual_x))
            assert (actual_x.dtype, actual_y.dtype) == (x.dtype, y.dtype)
            np.testing.assert_array_equal(actual_x, x)
            np.testing.assert_array_equal(actual_y, y)
            return fit(actual_x, actual_y)

        monkeypatch.setattr(model, "_fit_candidate", capture)
        exp = Experiment(model, BasicTraining(), train_data=authored)
        assert exp.train_fn(exp) is model.results
        assert submitted == [96] and model.definition == definition
        assert (exp.state.step, exp.state.epoch, exp.state.examples_seen, exp.state.next_batch) == (1, 1, 96, 0)
        assert exp.state.pending_epoch_postlude is None
        assert exp.state.stage_observation().training_loss is None
        assert model.results.model.exog.dtype == model.results.model.endog.dtype == np.float64
        assert not np.shares_memory(model.results.model.exog, x)
        assert not np.shares_memory(model.results.model.endog, y)
        models.append(model)
    assert models[0].results is not models[1].results
    np.testing.assert_array_equal(models[0](x), models[1](x))
    before = models[0](x)
    x[:] = 0
    y[:] = 0
    np.testing.assert_array_equal(models[0].results.predict(models[0].results.model.exog),
                                  before[:, 1] if classification else before)


@pytest.mark.parametrize("classification", [False, True])
def test_adapter_selected_prepared_calls_use_current_fit_and_refuse_changed_width(native_stats, classification, monkeypatch):
    """Learning and weak selected invokers retain width guards, including trait-only use."""
    np, _, _, _, _ = native_stats
    from dryml.core import TensorSpec
    from dryml.data import ArrayDataset, Batch, Map
    from dryml.methods import MethodError
    from dryml.models.statsmodels import ClassifierModel, RegressionModel

    x, y = _small_stats_data(np, classification=classification)
    model = (ClassifierModel if classification else RegressionModel)(intercept=True)
    spec = TensorSpec("float64", shape=(2,), backend="numpy")
    with monkeypatch.context() as guard:
        guard.setattr(model, "_new_native", lambda *a: pytest.fail("construction during preparation"))
        element = model.find_implementation(spec)
        batched = model.find_implementation(spec.with_batch(4)).prepared_invoker()
        prefit = model.find_implementation(backend="numpy", batch_mode="batched").prepared_invoker()
        mapped = Map(ArrayDataset(x), model).prepare()
        batch_map = Map(Batch(ArrayDataset(x), 17), model).prepare()
        model.learn(spec.with_batch(4))
        assert mapped.spec.shape == (() if not classification else (2,))
        assert mapped.spec.dtype.name == "float64"
    with pytest.raises(ValueError, match="unfitted"):
        prefit(x)
    model.fit(x, y)
    with monkeypatch.context() as guard:
        guard.setattr(model.results, "predict", lambda *a: pytest.fail("prediction during inference"))
        assert model.infer_output_spec(spec).shape == mapped.spec.shape
        model.find_implementation(spec).prepared_invoker()
        trait_only = model.find_implementation(backend="numpy", batch_mode="batched").prepared_invoker()
        uninvoked = model.find_implementation(backend="numpy", batch_mode="batched").prepared_invoker()
    expected = model.predict_proba(x) if classification else model.predict(x)
    for invoke in (prefit, trait_only):
        np.testing.assert_array_equal(invoke(x), expected)

    def collect(dataset):
        with dataset.iterator() as cursor:
            return list(cursor)

    np.testing.assert_array_equal(element(x[0]), expected[0])
    np.testing.assert_array_equal(batched(x[:4]), expected[:4])
    np.testing.assert_array_equal(model(x[:4]), expected[:4])
    np.testing.assert_allclose(np.asarray(collect(mapped)), expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.concatenate(collect(batch_map)), expected, rtol=1e-12, atol=1e-12)
    assert model.default_batched is None
    model.eager()
    np.testing.assert_array_equal(model(x[:4]), expected[:4])
    np.testing.assert_array_equal(model(x=x[:4]), expected[:4])
    for keyword in (True, False):
        model.learn()
        assert model.call_mode == "learning" and model.cached_signature is None

        def learned_call():
            return model(x=x[:4]) if keyword else model(x[:4])

        np.testing.assert_array_equal(learned_call(), expected[:4])
        assert model.call_mode == "cached" and model.cached_signature is not None
        signature = model.cached_signature
        assert signature.batch_mode is None
        np.testing.assert_array_equal(learned_call(), expected[:4])
        model.fit(x, y)
        np.testing.assert_array_equal(learned_call(), expected[:4])
        assert model.cached_signature == signature
    model.fit(x, 1 - y if classification else y + 100)
    current = model.predict_proba(x) if classification else model.predict(x)
    assert not np.array_equal(expected, current)
    for invoke in (prefit, trait_only):
        np.testing.assert_array_equal(invoke(x=x), current)
    np.testing.assert_allclose(np.asarray(collect(mapped)), current, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(model(x[:4]), current[:4])
    model.fit(x[:, :1], y)
    for invoke in (prefit, trait_only, uninvoked):
        with pytest.raises(MethodError, match="re-prepare"):
            invoke(x[:, :1])
    for action in (lambda: element(x[0]), lambda: batched(x[:4]), lambda: collect(mapped), lambda: model(x[:4])):
        with pytest.raises((MethodError, ValueError), match="re-prepare|width"):
            action()
    remapped = Map(ArrayDataset(x[:, :1]), model).prepare()
    np.testing.assert_allclose(np.asarray(collect(remapped)), model.predict_proba(x[:, :1]) if classification
                               else model.predict(x[:, :1]), rtol=1e-12, atol=1e-12)
    fresh = model.find_implementation(backend="numpy", batch_mode="batched").prepared_invoker()
    np.testing.assert_array_equal(fresh(x[:, :1]), model.predict_proba(x[:, :1]) if classification else model.predict(x[:, :1]))

    def discarded_owner():
        owner = (ClassifierModel if classification else RegressionModel)(intercept=True).fit(x, y)
        return weakref.ref(owner), owner.find_implementation(backend="numpy", batch_mode="batched").prepared_invoker()

    owner_ref, orphan = discarded_owner()
    gc.collect()
    assert owner_ref() is None
    with pytest.raises(MethodError, match="no longer live"):
        orphan(x)


def test_adapter_invalid_data_and_fit_options_preserve_prior_state(native_stats):
    np, _, _, _, _ = native_stats
    from dryml.models.statsmodels import ClassifierModel, RegressionModel

    x, y = _small_stats_data(np, classification=True)
    model = ClassifierModel().fit(x, y)
    prior, prediction = model._accepted_fit, model(x)
    bad = x.copy()
    bad[0, 0] = np.inf
    for features, targets in ((x[:0], y[:0]), (x, y + 1), (x, y * 2), (x, np.zeros_like(y)),
                              (x, y[:-1]), (bad, y), (x, y.astype(bool)), (x, y.astype(object))):
        with pytest.raises((TypeError, ValueError)):
            model.fit(features, targets)
        assert model._accepted_fit is prior
    for features in (x[:0], x[:, :1], x[0], bad, np.ma.array(x), x.astype(bool),
                     x.astype(np.complex128), x.astype(object), x.astype(np.float16)):
        with pytest.raises((TypeError, ValueError)):
            model.predict(features)
    for cls in (ClassifierModel, RegressionModel):
        target = cls().fit(x.astype(np.int64), y)
        accepted = target._accepted_fit
        for nonfinite in (np.nan, np.inf):
            values = y.copy()
            values[0] = nonfinite
            with pytest.raises(ValueError, match="nonfinite"):
                target.fit(x, values)
            assert target._accepted_fit is accepted
    np.testing.assert_array_equal(model(x), prediction)


@pytest.mark.parametrize("failure", ["params", "false_convergence", "missing_convergence", "perfect_prediction",
                                    "prediction", "encode_interrupt", "cleanup"])
def test_adapter_native_failure_preserves_results_bytes_and_success_counts(native_stats, monkeypatch, failure):
    """Real native candidate failures cannot advance accepted state or TrainState."""
    np, _, Logit, _, _ = native_stats
    from dryml.data import ArrayDataset, as_supervised
    from dryml.models import Experiment
    from dryml.models.statsmodels import ClassifierModel, BasicTraining
    from statsmodels.tools.sm_exceptions import PerfectSeparationError

    x, y = _small_stats_data(np, classification=True)
    model = ClassifierModel(intercept=True)
    exp = Experiment(model, BasicTraining(), train_data=as_supervised(ArrayDataset((x, y)), 0, 1))
    exp.train_fn(exp)
    prior, state, prediction = model._accepted_fit, exp.state, model(x)
    original_fit = Logit.fit

    def invalid_fit(native, **options):
        if failure == "perfect_prediction":
            assert native.raise_on_perfect_prediction is True
            options["start_params"] = np.array([0., 30.])
        result = original_fit(native, **options)
        if failure == "params":
            result._results.params[0] = np.inf
        elif failure == "false_convergence":
            result.mle_retvals["converged"] = False
        elif failure == "missing_convergence":
            del result.mle_retvals["converged"]
        return result

    if failure in ("params", "false_convergence", "missing_convergence", "perfect_prediction"):
        monkeypatch.setattr(Logit, "fit", invalid_fit)
        if failure == "perfect_prediction":
            separated_x = np.tile(np.array([-1., 1.]), 48)[:, None]
            separated_y = (separated_x[:, 0] > 0).astype(np.float64)
            exp.train_data = as_supervised(ArrayDataset((separated_x, separated_y)), 0, 1)
    elif failure == "prediction":
        monkeypatch.setattr(type(model.results._results), "predict", lambda self, exog: np.full(len(exog), 2.))
    elif failure == "encode_interrupt":
        def interrupt(*a, **k):
            raise KeyboardInterrupt("encoding interrupted")
        monkeypatch.setattr(type(model.results), "save", interrupt)
    else:
        def fail_cleanup():
            raise RuntimeError("cleanup failed")
        monkeypatch.setattr(model, "prep_eval", fail_cleanup)
    filters = list(warnings.filters)
    error = KeyboardInterrupt if failure == "encode_interrupt" else (RuntimeError if failure == "cleanup" else ValueError)
    if failure == "perfect_prediction":
        error = PerfectSeparationError
    with pytest.raises(error):
        exp.train_fn(exp)
    assert warnings.filters == filters
    assert model._accepted_fit is prior and exp.state is state
    assert (state.step, state.epoch, state.examples_seen) == (1, 1, 96)
    monkeypatch.undo()
    np.testing.assert_array_equal(model(x), prediction)


def test_adapter_real_logit_nonconvergence_is_not_accepted(native_stats):
    np, _, _, _, _ = native_stats
    from dryml.models.statsmodels import ClassifierModel
    from statsmodels.tools.sm_exceptions import ConvergenceWarning

    x, y = _small_stats_data(np, classification=True)
    model = ClassifierModel(fit_options={"maxiter": 1})
    with pytest.warns(ConvergenceWarning), pytest.raises(ValueError, match="convergence"):
        model.fit(x, y)
    assert model.results is None


@pytest.mark.usefixtures("fixed_snapshot_environment")
@pytest.mark.parametrize("family", ["ols", "logit", "unfitted"])
def test_adapter_complete_results_exact_repo_restore_and_diagnostic_reemission(native_stats, tmp_path, monkeypatch, family):
    """Fresh load retains complete inspection and exact bytes despite native cache changes."""
    np, _, _, _, _ = native_stats
    from dryml.core import Repo
    from dryml.core.store.dir import DirStore
    from dryml.models.statsmodels import ClassifierModel, RegressionModel

    x, y = _small_stats_data(np, classification=family == "logit")
    model = (ClassifierModel if family == "logit" else RegressionModel)(intercept=True)
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    unfitted = model.save(repo=repo)
    if family != "unfitted":
        model.fit(x, y)
        model.learn()
    before = None if model.results is None else set(model.results._results._cache)
    state = model.save(repo=repo)
    path = Path(store.open_local_state(state, next(iter(state.states))).handle, "data")
    inventory = {p.name: p.read_bytes() for p in path.iterdir()}
    assert set(inventory) == ({"metadata.json"} if family == "unfitted" else {"metadata.json", "results.pkl"})
    assert all(v.startswith("statsmodels1-") for v in state.states.values())
    if before is not None:
        assert set(model.results._results._cache) == before
    with monkeypatch.context() as guard:
        if model.results is not None:
            guard.setattr(type(model.results._results), "predict", lambda *a: pytest.fail("prediction during restore"))
        restored = Repo(store).load_state_ref(state, reuse_live="never")
    assert restored is not model and restored.definition == model.definition
    assert restored.call_mode == "eager" and restored.default_batched is None
    if family != "unfitted":
        assert restored._accepted_fit.payload == model._accepted_fit.payload
        assert restored._accepted_fit.metadata == model._accepted_fit.metadata
        np.testing.assert_array_equal(restored(x), model.predict_proba(x) if family == "logit" else model.predict(x))
        for field in ("exog", "endog"):
            np.testing.assert_array_equal(getattr(restored.results.model, field), getattr(model.results.model, field))
            np.testing.assert_array_equal(getattr(restored.results.model.data, "orig_" + field), getattr(model.results.model, field))
        for field in ("params", "bse"):
            np.testing.assert_allclose(getattr(restored.results, field), getattr(model.results, field), rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(restored.results.cov_params(), model.results.cov_params(), rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(restored.results.conf_int(), model.results.conf_int(), rtol=1e-8, atol=1e-8)
        assert np.isfinite(restored.results.llf) and restored.results.nobs == 96
        assert "Results" in str(restored.results.summary())
        monkeypatch.setattr(type(restored.results), "save", lambda *a, **k: pytest.fail("native re-encoding"))
    else:
        assert restored.results is None
    resaved = restored.save(repo=repo)
    assert resaved.states == state.states
    second = Path(store.open_local_state(resaved, next(iter(resaved.states))).handle, "data")
    assert {p.name: p.read_bytes() for p in second.iterdir()} == inventory
    if family != "unfitted":
        identity = model.object_id
        repo.restore_state_ref_into(model, unfitted)
        assert model.object_id == identity and model.results is None
        with pytest.raises(ValueError, match="unfitted"):
            model.n_features_in_


def test_adapter_corrupt_incompatible_and_inconsistent_state_is_candidate_first(native_stats, tmp_path):
    np, _, _, load_pickle, _ = native_stats
    from dryml.models.statsmodels import ClassifierModel

    x, y = _small_stats_data(np, classification=True)
    model = ClassifierModel(intercept=True).fit(x, y)
    model.save_state_to_dir(tmp_path, codec="statsmodels1")
    prior, prediction = model._accepted_fit, model(x)
    original = json.loads(prior.metadata)
    for change in ({"version": 2}, {"extra": 1}, {"feature_width": True}, {"feature_width": 3},
                   {"classes": [False, True]}, {"classes": [1, 0]}, {"task": "regression"},
                   {"intercept": False}, {"design_policy": "detect"}, {"call_dtype": "float32"},
                   {"writer_versions": {}}, {"writer_python_patch": "3.13.0"}, {"native_sha256": "bad"},
                   {"fitted": False}, {"fitted": 1}, {"fit_options": {"maxiter": 100, "tol": 1e-8, "disp": 0}}):
        Path(tmp_path, "metadata.json").write_bytes(json.dumps({**original, **change}, sort_keys=True, separators=(",", ":")).encode())
        with pytest.raises(ValueError):
            model.restore_state_from_dir(tmp_path, codec="statsmodels1")
        assert model._accepted_fit is prior
    for payload in (b"corrupt pickle", prior.payload + b"trailing bytes"):
        Path(tmp_path, "results.pkl").write_bytes(payload)
        envelope = {**original, "native_sha256": hashlib.sha256(payload).hexdigest()}
        Path(tmp_path, "metadata.json").write_bytes(json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode())
        with pytest.raises(ValueError, match="pickle"):
            model.restore_state_from_dir(tmp_path, codec="statsmodels1")
        assert model._accepted_fit is prior
    for corrupt in ("intercept", "params", "convergence"):
        native = load_pickle(io.BytesIO(prior.payload))
        if corrupt == "intercept":
            native.model.exog[:, 0] = 0
        elif corrupt == "params":
            native._results.params[0] = np.nan
        else:
            native.mle_retvals["converged"] = False
        stream = io.BytesIO()
        native.save(stream, remove_data=False)
        payload = stream.getvalue()
        Path(tmp_path, "results.pkl").write_bytes(payload)
        envelope = {**original, "native_sha256": hashlib.sha256(payload).hexdigest()}
        Path(tmp_path, "metadata.json").write_bytes(json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode())
        with pytest.raises(ValueError):
            model.restore_state_from_dir(tmp_path, codec="statsmodels1")
        assert model._accepted_fit is prior
    Path(tmp_path, "metadata.json").write_bytes(prior.metadata + b"\n")
    with pytest.raises(ValueError, match="canonical"):
        model.restore_state_from_dir(tmp_path, codec="statsmodels1")
    Path(tmp_path, "heavy.pkl").write_bytes(b"legacy")
    with pytest.raises(ValueError, match="Legacy"):
        model.restore_state_from_dir(tmp_path, codec="statsmodels1")
    with pytest.raises(ValueError, match="codec"):
        model.restore_state_from_dir(tmp_path, codec="pkl")
    np.testing.assert_array_equal(model(x), prediction)


@pytest.mark.usefixtures("fixed_managed_snapshot_environment")
def test_adapter_local_managed_experiment_terminal_history_and_restore(native_stats, tmp_path):
    np, _, _, _, _ = native_stats
    from dryml.core import Repo, StateRef
    from dryml.core.store.dir import DirStore
    from dryml.data import ArrayDataset, Batch, as_supervised
    from dryml.managed import ManagedConfig
    from dryml.models import Experiment, ExperimentData
    from dryml.models.statsmodels import RegressionModel, BasicTraining

    x, y = _small_stats_data(np)
    repo = Repo(DirStore(tmp_path / "store"))
    exp = Experiment(RegressionModel(intercept=True), BasicTraining(), repo=repo,
                     train_data=Batch(as_supervised(ArrayDataset((x, y)), 0, 1), 17))
    terminal = []
    final = exp.train(managed=ManagedConfig(state_repo=repo, callbacks=[
        lambda obj, context: terminal.append(context.checkpoint_state_ref)]))
    assert isinstance(final, StateRef) and terminal == [final]
    assert exp.train.status(state_repo=repo).final_state_ref == final
    history = ExperimentData.find(final.object_projection(), repo=repo).data
    assert len(history) == 1 and history.iloc[0].state_ref == final
    assert history.iloc[0].evaluation_status == "completed" and history.iloc[0].examples_seen == 96
    fresh = repo.load_state_ref(final, reuse_live="never")
    assert fresh.model is not exp.model and fresh.model._accepted_fit.payload == exp.model._accepted_fit.payload
    assert (fresh.state.step, fresh.state.epoch, fresh.state.examples_seen) == (1, 1, 96)
    np.testing.assert_array_equal(fresh.model(x), exp.model(x))


@pytest.mark.usefixtures("fixed_snapshot_environment")
def test_adapter_failed_entered_restore_obeys_core_identity_invalidation(native_stats, tmp_path, monkeypatch):
    np, _, _, _, _ = native_stats
    from dryml.core import Repo
    from dryml.core.repo import RepoLoadError, RepoSaveError
    from dryml.core.store.dir import DirStore
    from dryml.models.statsmodels import RegressionModel

    x, y = _small_stats_data(np)
    repo = Repo(DirStore(tmp_path / "store"))
    model = RegressionModel().fit(x, y)
    state = model.save(repo=repo)
    prior = model._accepted_fit

    def reject(candidate):
        raise ValueError("native metadata rejection")

    monkeypatch.setattr(model, "_validate_native", reject)
    with pytest.raises(RepoLoadError, match="Targeted exact restore failed"):
        repo.restore_state_ref_into(model, state)
    assert model._accepted_fit is prior and model._restore_failed and model.last_state_ref is None
    with pytest.raises(RepoSaveError, match="invalidated"):
        model.save(repo=repo)
    fresh = repo.load_state_ref(state, reuse_live="never")
    assert fresh is not model and not fresh._restore_failed
    np.testing.assert_array_equal(fresh(x), prior.native.predict(x))


def test_adapter_real_fit_and_fitted_restore_never_require_sklearn_or_xgb(native_stats, tmp_path):
    """The real statsmodels dependency path remains independent of other ML adapters."""
    script = """
import sys
class RejectOtherModels:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'sklearn', 'xgboost', 'tensorflow', 'torch', 'jax'}:
            raise AssertionError('unexpected native import: ' + fullname)
sys.meta_path.insert(0, RejectOtherModels())
import numpy as np
from dryml.models.statsmodels import ClassifierModel, RegressionModel
rng = np.random.default_rng(17)
x = rng.normal(size=(96, 2))
for cls in (ClassifierModel, RegressionModel):
    y = (rng.random(96) < .5).astype(np.float64) if cls is ClassifierModel else x[:, 0] + .3
    model = cls(intercept=True).fit(x, y)
    model.save_state_to_dir(sys.argv[1], codec='statsmodels1')
    restored = cls(intercept=True)
    restored.restore_state_from_dir(sys.argv[1], codec='statsmodels1')
    np.testing.assert_array_equal(restored(x), model(x))
assert not {'sklearn', 'xgboost', 'tensorflow', 'torch', 'jax'} & sys.modules.keys()
"""
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path)], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr


def test_adapter_ols_cleanup_precedes_single_encoding_and_nonfinite_params_refuse(native_stats, tmp_path, monkeypatch):
    """OLS shares the failure boundary and never encodes during later saves."""
    np, OLS, _, _, _ = native_stats
    from dryml.data import ArrayDataset, as_supervised
    from dryml.models import Experiment
    from dryml.models.statsmodels import RegressionModel, BasicTraining

    x, y = _small_stats_data(np)
    model = RegressionModel().fit(x, y)
    exp = Experiment(model, BasicTraining(), train_data=as_supervised(ArrayDataset((x, y + 1)), 0, 1))
    prior, state = model._accepted_fit, exp.state
    original_fit, original_save = OLS.fit, type(model.results).save
    events = []

    def invalid_fit(native, **options):
        result = original_fit(native, **options)
        result._results.params[0] = np.nan
        return result

    with monkeypatch.context() as guard:
        guard.setattr(OLS, "fit", invalid_fit)
        with pytest.raises(ValueError, match="params"):
            exp.train_fn(exp)
    assert model._accepted_fit is prior and exp.state is state

    def encode(result, *args, **kwargs):
        assert events == ["prep_eval"]
        assert model._accepted_fit is prior and exp.state is state
        assert kwargs == {"remove_data": False}
        events.append("encode")
        return original_save(result, *args, **kwargs)

    monkeypatch.setattr(model, "prep_eval", lambda: events.append("prep_eval"))
    monkeypatch.setattr(type(model.results), "save", encode)
    exp.train_fn(exp)
    assert events == ["prep_eval", "encode"]
    assert (exp.state.step, exp.state.epoch, exp.state.examples_seen) == (1, 1, 96)
    cache = set(model.results._results._cache)
    model.save_state_to_dir(tmp_path, codec="statsmodels1")
    assert events == ["prep_eval", "encode"] and set(model.results._results._cache) == cache
    assert Path(tmp_path, "results.pkl").read_bytes() == model._accepted_fit.payload


def test_adapter_incompatible_native_reader_refuses_fit_and_restore(native_stats, tmp_path, monkeypatch):
    np, _, _, _, _ = native_stats
    from dryml.models.statsmodels import RegressionModel, base

    x, y = _small_stats_data(np)
    model = RegressionModel().fit(x, y)
    prior = model._accepted_fit
    model.save_state_to_dir(tmp_path, codec="statsmodels1")
    monkeypatch.setattr(base, "version", lambda name: "unqualified")
    for action in (lambda: model.fit(x, y), lambda: model.restore_state_from_dir(tmp_path, codec="statsmodels1")):
        with pytest.raises(ValueError, match="reader/writer versions"):
            action()
        assert model._accepted_fit is prior
