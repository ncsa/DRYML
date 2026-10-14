"""Array-native OLS/Logit references before implementing DRYML adapters.

Exact candidate dependencies are mandatory at execution, optional at collection.
Full Results payloads deliberately retain training data; pickle byte stability
is recorded independently from numerical and inspection equivalence.
"""

import hashlib
from importlib.metadata import version
import io
import json
import platform
import sys
import warnings

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
    assert str(restored.summary()) == str(result.summary())
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
