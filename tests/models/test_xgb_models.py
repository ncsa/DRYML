"""Native CPU references for the pending XGBoost adapter, not adapter qualification.

Explicit execution requires the exact candidate stack; missing packages fail rather
than skip. Native imports occur only in fixtures/test bodies after collection.
"""

import hashlib
from importlib.metadata import version
import json
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
