"""Passive CPU XGBoost recipes, NumPy prediction, and retained native model IO.

Native imports and version admission occur only for fitting or fitted restoration.
The supported reader/writer uses XGBoost 3.4.1; fitted state written with other
XGBoost versions is rejected rather than migrated or silently reinterpreted.
One accepted bundle owns the borrowed estimator, fitted facts, and exact bytes;
same-model concurrent fit/restore and external estimator mutation are unsupported.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import sys
import tempfile
import weakref

import numpy as np

from dryml.core.backend import Backend
from dryml.core.object import Serializable
from dryml.core.tensor_spec import Dynamic, Layout, TensorSpec, match_input_batch
from dryml.methods import ImplementationSelectionError, MethodError, traits
from dryml.models.model import Model as BaseModel
from dryml.models.train_func import TrainFunction
from dryml.models.utils import _fit_one_shot


_VERSIONS = {
    "xgboost": "3.4.1", "scikit-learn": "1.9.0", "numpy": "2.1.3",
    "scipy": "1.18.0", "pandas": "3.0.5", "python": "3.12",
}
_DEFAULTS = {
    "n_estimators": 100, "max_depth": 6, "learning_rate": 0.3,
    "min_child_weight": 1.0, "gamma": 0.0, "subsample": 1.0,
    "colsample_bytree": 1.0, "colsample_bylevel": 1.0, "colsample_bynode": 1.0,
    "reg_alpha": 0.0, "reg_lambda": 1.0, "n_jobs": 1, "random_state": 0,
}


def _options(options):
    unknown = set(options) - set(_DEFAULTS)
    if unknown:
        raise TypeError(f"Unsupported XGBoost estimator options: {', '.join(sorted(unknown))}.")
    result = {**_DEFAULTS, **options}
    for name, value in result.items():
        if name in ("n_estimators", "max_depth", "n_jobs", "random_state"):
            if type(value) is not int:
                raise TypeError(f"XGBoost {name} must be an exact integer.")
            minimum = 0 if name in ("max_depth", "random_state") else 1
            if value < minimum or value > 2**31 - 1:
                raise ValueError(f"XGBoost {name} must be in [{minimum}, 2147483647].")
        else:
            if type(value) not in (int, float):
                raise TypeError(f"XGBoost {name} must be a finite real number.")
            try:
                finite = math.isfinite(value)
            except OverflowError:
                finite = False
            if not finite or not 0 <= value <= float(np.finfo(np.float32).max):
                raise ValueError(f"XGBoost {name} must be finite, nonnegative and representable in float32.")
            if name in (
                "learning_rate", "subsample", "colsample_bytree",
                "colsample_bylevel", "colsample_bynode",
            ) and not 0 < value <= 1:
                raise ValueError(f"XGBoost {name} must be in (0, 1].")
    return result


def _native():
    try:
        import xgboost
        installed = {name: version(name) for name in _VERSIONS if name != "python"}
    except (ImportError, PackageNotFoundError) as error:
        raise ImportError("XGBoost requires the xgboost extra; install 'dryml[xgboost]'.") from error
    installed["python"] = f"{sys.version_info.major}.{sys.version_info.minor}"
    if installed != _VERSIONS:
        raise ValueError(f"Unsupported XGBoost reader/writer versions; requires {_VERSIONS}, found {installed}.")
    return xgboost, installed


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _array(value, *, name, rank):
    if (not isinstance(value, np.ndarray)
            or isinstance(value, (np.ma.MaskedArray, np.matrix))):
        raise TypeError(f"XGBoost {name} requires a dense NumPy array; author conversion explicitly.")
    if value.dtype.kind not in "iuf" or (value.dtype.kind == "f" and value.dtype.itemsize not in (4, 8)):
        raise TypeError(f"XGBoost {name} requires real integer or float32/float64 dtype.")
    if value.ndim != rank or any(size == 0 for size in value.shape):
        raise ValueError(f"XGBoost {name} requires a nonempty rank-{rank} array.")
    return value


@dataclass(frozen=True, slots=True)
class _Candidate:
    native: object
    feature_width: int
    classes: tuple[int, ...] | None


@dataclass(frozen=True, slots=True)
class _AcceptedFit:
    native: object
    feature_width: int
    classes: tuple[int, ...] | None
    payload: bytes
    metadata: bytes


class Model(BaseModel, Serializable):
    """CPU gbtree/hist estimator recipe with candidate-first native state.

    Args:
        task: ``'classification'`` or ``'regression'``; concrete subclasses fix it.
        output_spec: Optional unbatched dense NumPy float32 TensorSpec assertion:
            scalar regression or fixed ``(K,)`` classification, K >= 2.
        **options: Closed scalar estimator options listed in ``docs/xgboost.md``.
            Defaults include n_jobs=1, random_state=0 and 100 trees. Device,
            objective, booster and tree_method are adapter-owned, not options.

    Raises:
        TypeError, ValueError: Invalid task, options or output specification.

    Construction stores configuration only; it imports no XGBoost/sklearn,
    constructs no estimator, scans no host and acquires no data/resources.
    Configuration is CDef state; fitted native/fact/byte bundles are Serializable
    state. Direct calls require nonempty 2D NumPy batches. Selected elements lift
    one row and squeeze the result. No implicit foreign-array handoff is allowed.
    """

    native_backend = "numpy"
    state_codec = "xgboost1"

    def __init__(
        self, task, *, output_spec=None, n_estimators=100, max_depth=6,
        learning_rate=0.3, min_child_weight=1.0, gamma=0.0, subsample=1.0,
        colsample_bytree=1.0, colsample_bylevel=1.0, colsample_bynode=1.0,
        reg_alpha=0.0, reg_lambda=1.0, n_jobs=1, random_state=0,
    ):
        """Record an inert recipe; parameters, errors and side effects follow Model.

        ``task`` is a task string, ``output_spec`` an optional TensorSpec and
        ``options`` closed scalar native configuration. Returns a new unfitted
        wrapper without native construction or resource acquisition.
        """
        if task not in ("classification", "regression"):
            raise ValueError("XGBoost task must be classification or regression.")
        self.task = task
        self.estimator_options = _options({name: value for name, value in locals().items() if name in _DEFAULTS})
        self.output_spec = output_spec
        if output_spec is not None:
            if (not isinstance(output_spec, TensorSpec) or output_spec.batched
                    or output_spec.backend is not Backend.numpy or output_spec.layout is not Layout.DENSE
                    or output_spec.dtype.name != "float32"):
                raise TypeError("XGBoost output_spec must be an unbatched dense NumPy float32 TensorSpec.")
            shape = output_spec.shape
            if task == "regression":
                valid = shape == ()
            else:
                valid = shape is not None and len(shape) == 1 and type(shape[0]) is int and shape[0] >= 2
            if not valid:
                raise ValueError("XGBoost output_spec requires scalar regression or classification shape (K,), K >= 2.")
        self._accepted_fit = None

    def __getattr__(self, name):
        """Expose bundle-owned inspection attributes without duplicating fitted state.

        ``mdl`` borrows the estimator (None if unfitted); ``n_features_in_`` returns
        fitted int width; ``classes_`` returns a detached int64 vocabulary array.
        The latter two raise ValueError if unavailable; unknown names raise
        AttributeError. Reads import nothing and mutate nothing. External mutation
        through ``mdl`` is unsupported because retained bytes remain authoritative.
        """
        if name == "mdl":
            fitted = vars(self).get("_accepted_fit")
            return None if fitted is None else fitted.native
        if name == "n_features_in_":
            return self._require_fit().feature_width
        if name == "classes_":
            fitted = self._require_fit()
            if fitted.classes is None:
                raise ValueError("XGBoost regression has no classes_.")
            return np.asarray(fitted.classes, dtype=np.int64)
        raise AttributeError(name)

    def _require_fit(self):
        if self._accepted_fit is None:
            raise ValueError("XGBoost model is unfitted; call fit or BasicTraining first.")
        return self._accepted_fit

    def _objective(self, classes):
        if self.task == "regression":
            return "reg:squarederror"
        return "binary:logistic" if len(classes) == 2 else "multi:softprob"

    def _new_native(self, classes):
        xgb, _ = _native()
        cls = xgb.XGBClassifier if self.task == "classification" else xgb.XGBRegressor
        return cls(booster="gbtree", tree_method="hist", device="cpu", verbosity=0,
                   objective=self._objective(classes), **self.estimator_options)

    def _validate_native(self, candidate):
        native, width, classes = candidate.native, candidate.feature_width, candidate.classes
        learner = json.loads(native.get_booster().save_config())["learner"]
        params = learner["learner_model_param"]
        if (native.n_features_in_ != width or int(params["num_feature"]) != width
                or learner["objective"]["name"] != self._objective(classes)
                or learner["gradient_booster"]["name"] != "gbtree"
                or learner["generic_param"]["device"] != "cpu"):
            raise ValueError("Inconsistent XGBoost native feature/task/CPU metadata.")
        expected_count = len(classes) if classes is not None and len(classes) > 2 else 0
        if int(params["num_class"]) != expected_count or int(params["num_target"]) != 1:
            raise ValueError("Inconsistent XGBoost native class/target metadata.")
        if classes is not None and tuple(native.classes_) != classes:
            raise ValueError("Inconsistent XGBoost native class vocabulary.")

    def _check_output(self, result, *, rows, classes, labels=False):
        shape = (rows,) if labels or classes is None else (rows, len(classes))
        dtype = np.dtype("int64" if labels and classes is not None else "float32")
        if not isinstance(result, np.ndarray) or result.shape != shape or result.dtype != dtype:
            raise ValueError("XGBoost returned inconsistent prediction shape/dtype.")
        if not np.isfinite(result).all():
            raise ValueError("XGBoost returned nonfinite predictions.")
        if classes is not None:
            if labels:
                if not np.isin(result, classes).all():
                    raise ValueError("XGBoost returned invalid class labels.")
            elif (not ((result >= 0) & (result <= 1)).all()
                  or not np.allclose(result.sum(axis=1), 1, rtol=1e-6, atol=1e-6)):
                raise ValueError("XGBoost returned invalid class probabilities.")
        if not labels and self.output_spec is not None and not self.output_spec.compatible_with_shape(shape[1:]):
            raise ValueError("XGBoost prediction violates explicit output_spec.")
        return result

    def _fit_candidate(self, x, y, *, verbose=False):
        x = _array(x, name="features", rank=2)
        y = _array(y, name="targets", rank=1)
        if len(x) != len(y):
            raise ValueError("XGBoost features and targets require matching example counts.")
        if np.isinf(x).any() or not np.isfinite(y).all():
            raise ValueError("XGBoost rejects infinite features and nonfinite targets (NaN features are supported).")
        classes = None
        if self.task == "classification":
            if y.dtype.kind not in "iu":
                raise TypeError("XGBoost classification targets must have integer dtype.")
            observed = np.unique(y)
            if len(observed) < 2 or not np.array_equal(observed, np.arange(len(observed))):
                raise ValueError("XGBoost classification requires >=2 observed classes exactly 0..K-1.")
            classes = tuple(int(label) for label in observed)
        native = self._new_native(classes)
        native.fit(x, y, verbose=verbose)
        candidate = _Candidate(native, x.shape[1], classes)
        self._validate_native(candidate)
        result = native.predict(x) if classes is None else native.predict_proba(x)
        self._check_output(result, rows=len(x), classes=classes)
        return candidate

    def _metadata(self, *, candidate=None, payload=b"", versions=None):
        return {
            "version": 1, "fitted": candidate is not None, "task": self.task,
            "feature_width": None if candidate is None else candidate.feature_width,
            "classes": None if candidate is None or candidate.classes is None else list(candidate.classes),
            "prediction_dtype": "int64" if self.task == "classification" else "float32",
            "call_dtype": "float32", "writer_versions": versions,
            "native_sha256": None if candidate is None else hashlib.sha256(payload).hexdigest(),
        }

    def _encode_candidate(self, candidate):
        _, versions = _native()
        with tempfile.TemporaryDirectory(prefix="dryml-xgboost-") as directory:
            path = Path(directory, "model.ubj")
            candidate.native.save_model(path)
            payload = path.read_bytes()
        metadata = _json_bytes(self._metadata(candidate=candidate, payload=payload, versions=versions))
        return _AcceptedFit(candidate.native, candidate.feature_width, candidate.classes, payload, metadata)

    def fit(self, x, y, *, verbose=False):
        """Fit one detached estimator from dense NumPy arrays and return self.

        Args:
            x: Nonempty (N, F) integer/float32/float64 ndarray. NaNs allowed;
                infinities rejected. No coercion from foreign arrays occurs.
            y: Nonempty (N,) finite ndarray; classification requires integer
                observed vocabulary exactly 0..K-1, K >= 2.
            verbose: Exact bool, default False; the sole native fit option.

        Raises:
            TypeError, ValueError: Invalid arrays, options, facts or output spec.
            ImportError: Missing xgboost extra. Native errors/interruption propagate.

        Side Effects:
            Imports/admit native dependencies after array admission, fits and
            validates a private candidate, encodes once, then replaces the bundle.
            Failure preserves previous accepted state. No Experiment accounting
            is invented; concurrent fit/restore on this model is unsupported.
        """
        if type(verbose) is not bool:
            raise TypeError("XGBoost verbose must be an exact bool.")
        candidate = self._fit_candidate(x, y, verbose=verbose)
        accepted = self._encode_candidate(candidate)
        self._accepted_fit = accepted
        return self

    def _predict(self, x, *, labels=False):
        x = _array(x, name="features", rank=2)
        fitted = self._require_fit()
        if x.shape[1] != fitted.feature_width:
            raise ValueError("XGBoost feature width differs from fitted original feature width.")
        if np.isinf(x).any():
            raise ValueError("XGBoost features must not contain infinity.")
        result = fitted.native.predict(x) if labels or fitted.classes is None else fitted.native.predict_proba(x)
        return self._check_output(result, rows=len(x), classes=fitted.classes, labels=labels)

    def predict(self, x):
        """Predict a nonempty 2D NumPy batch: (N,) int64 labels or float32 regression.

        NaN features are supported. Foreign arrays, invalid rank/dtype, infinity,
        wrong fitted width or unfitted state raise TypeError/ValueError before
        native prediction; native failures propagate. Accepted state is unchanged.
        """
        return self._predict(x, labels=True)

    def _runtime_selection_facts(self, args, kwargs):
        x = args[0] if args else kwargs.get("x")
        _array(x, name="features", rank=2)
        return Backend.numpy, None

    @traits(backend="numpy")
    def raw_call(self, x):
        """Return batch float32 probabilities/regression; same admission/errors as predict."""
        return self._predict(x)

    @traits(backend="numpy", batch_mode="batched")
    def batched_call(self, x):
        """Return (N,K)/(N,) float32 for one selected batch, without a second axis."""
        return self._predict(x)

    @traits(backend="numpy", batch_mode="element")
    def element_call(self, x):
        """Lift a selected 1D NumPy element; return (K,)/scalar float32.

        Invalid arrays, infinity, width or unfitted state raise TypeError/ValueError;
        native errors propagate. Only the temporary batch view is created.
        """
        x = _array(x, name="element features", rank=1)
        return self._predict(x[None, :])[0]

    def _input_spec(self, spec):
        if (not isinstance(spec, TensorSpec) or spec.backend is not Backend.numpy
                or spec.layout is not Layout.DENSE or spec.shape is None or len(spec.shape) != 1
                or spec.dtype.kind not in ("int", "uint", "float")
                or (spec.dtype.kind in ("int", "uint") and spec.dtype.bits not in (8, 16, 32, 64))
                or (spec.dtype.kind == "float" and spec.dtype.bits not in (32, 64))):
            raise TypeError("XGBoost requires a dense real NumPy feature TensorSpec; author conversion explicitly.")
        if spec.shape[0] != Dynamic and spec.shape[0] <= 0:
            raise ValueError("XGBoost feature width must be positive.")
        if self._accepted_fit is not None and spec.shape[0] not in (Dynamic, self._accepted_fit.feature_width):
            raise ValueError("XGBoost spec feature width differs from fitted width.")

    def infer_output_spec(self, input_spec, *additional_input_specs):
        """Infer NumPy float32 output from pure recipe/fitted facts, never native calls.

        Args:
            input_spec: Dense real NumPy sample shape (F,), optional batch axis.
            *additional_input_specs: Unsupported; this Model is unary.

        Returns:
            Scalar regression or (K,) classification TensorSpec with input batching.
            Unfitted classification requires the constructor's explicit output_spec.

        Raises:
            TypeError, ValueError: Invalid inputs or conflicting explicit output spec.
            NotImplementedError: Unfitted classification lacks declared K.

        Side Effects: None: no estimator construction, imports, prediction or data.
        """
        if additional_input_specs:
            raise TypeError("XGBoost Model accepts exactly one input specification.")
        self._input_spec(input_spec)
        if self.task == "regression":
            shape = ()
        elif self._accepted_fit is not None:
            shape = (len(self._accepted_fit.classes),)
        elif self.output_spec is not None:
            shape = self.output_spec.shape
        else:
            raise NotImplementedError("Unfitted XGBoost classification requires explicit output_spec with class count.")
        if self.output_spec is not None and self.output_spec.shape != shape:
            raise ValueError("XGBoost fitted classes conflict with explicit output_spec.")
        spec = self.output_spec or TensorSpec("float32", shape=shape, backend="numpy")
        return match_input_batch(spec, input_spec)

    def _admit_selection(self, input_spec, additional=(), backend=None):
        try:
            if additional or (backend is not None and Backend(backend) is not Backend.numpy):
                raise TypeError("XGBoost accepts one NumPy input only.")
            if input_spec is not None:
                self._input_spec(input_spec)
        except (TypeError, ValueError) as error:
            raise ImplementationSelectionError("conflict") from error

    def find_implementation(self, input_spec=None, *additional_input_specs, backend=None,
                            batch_mode=None, output_spec=None):
        """Select one NumPy call, rejecting foreign specs before generic fallback.

        Parameters follow Method.find_implementation: optional unary TensorSpec,
        backend/batch constraints and runtime output assertion. Returns a pure
        selected carrier; raises ImplementationSelectionError for invalid facts.
        Imports/acquires/converts nothing and does not change preparation state.
        Invokers borrow this model weakly and raise MethodError if it is gone or
        fitted width/classes change, including trait-only selections. Unfitted
        selections capture fitted facts lazily on first fitted invocation.
        """
        self._admit_selection(input_spec, additional_input_specs, backend)
        selected = super().find_implementation(input_spec, backend=backend, batch_mode=batch_mode,
                                               output_spec=output_spec)
        return self._specialize(selected, input_spec)

    def _prepare_implementation(self, input_spec, *, backend, batch_mode):
        # Unknown-intent learning supplies physical shape; raw calls own this batch axis.
        if (batch_mode is None and isinstance(input_spec, TensorSpec)
                and input_spec.backend is Backend.numpy and input_spec.layout is Layout.DENSE
                and input_spec.batch is None and input_spec.shape is not None and len(input_spec.shape) == 2):
            input_spec = replace(input_spec, shape=input_spec.shape[1:], batch=input_spec.shape[0],
                                 axis_names=None if input_spec.axis_names is None else input_spec.axis_names[1:],
                                 batch_axis_name="batch")
        self._admit_selection(input_spec, backend=backend)
        selected = super()._prepare_implementation(input_spec, backend=backend, batch_mode=batch_mode)
        return self._specialize(selected, input_spec)

    def _specialize(self, selected, input_spec):
        try:
            output = None if input_spec is None else self.infer_output_spec(input_spec)
        except NotImplementedError:
            output = None
        facts = None if self._accepted_fit is None else (self._accepted_fit.feature_width, self._accepted_fit.classes)
        receiver_ref = weakref.ref(self)
        name = selected.name

        def invoke(x):
            nonlocal facts
            receiver = receiver_ref()
            if receiver is None:
                raise MethodError("The selected XGBoost Model is no longer live.")
            current = receiver._require_fit()
            current_facts = (current.feature_width, current.classes)
            if facts is not None and facts != current_facts:
                raise MethodError("XGBoost fitted specifications changed; explicitly re-prepare the Method/Map.")
            if output is not None and not output.equal_exact(receiver.infer_output_spec(input_spec)):
                raise MethodError("XGBoost fitted specifications changed; explicitly re-prepare the Method/Map.")
            if facts is None:
                facts = current_facts
            return getattr(receiver, name)(x)

        if output is None:
            return replace(selected, _invoker=invoke)
        return replace(selected, _invoker=invoke, _output_inference=lambda children: output)

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        """Write retained native bytes and canonical metadata without re-encoding.

        Args: dest_dir: Core-owned empty payload directory; codec: 'xgboost1'.
        Raises: ValueError for unsupported codec; filesystem errors propagate.
        Side Effects: Writes metadata.json and, when fitted, model.ubj only. No
            native import for unfitted saves, estimator pickle or authority bypass.
        """
        self._check_codec(codec)
        fitted = self._accepted_fit
        metadata = _json_bytes(self._metadata()) if fitted is None else fitted.metadata
        Path(dest_dir, "metadata.json").write_bytes(metadata)
        if fitted is not None:
            Path(dest_dir, "model.ubj").write_bytes(fitted.payload)

    @staticmethod
    def _check_codec(codec):
        if codec != "xgboost1":
            raise ValueError("Unsupported XGBoost codec; requires xgboost1. Legacy model.text/compute state is not supported; rebuild and refit.")

    def restore_state_from_dir_imp(self, src_dir, *, codec):
        """Decode/validate a detached native candidate, then replace accepted state.

        Args: src_dir: Core-owned authenticated payload directory; codec: xgboost1.
        Raises: ValueError for legacy codec, malformed/inconsistent envelope or
            version mismatch; native/import/filesystem failures propagate.
        Side Effects: Imports XGBoost only for fitted restore; installs only after
            full validation and retains exact loaded bytes. Unfitted state clears
            all fitted facts. Core owns failed-entered-hook invalidation; this hook
            never rescues validity, changes IDs/bindings or rewrites a Store.
        """
        self._check_codec(codec)
        directory = Path(src_dir)
        if (directory / "model.text").exists():
            raise ValueError("Legacy XGBoost model.text compute payload is unsupported; rebuild and refit.")
        metadata = (directory / "metadata.json").read_bytes()
        try:
            envelope = json.loads(metadata)
            if (type(envelope) is not dict or set(envelope) != set(self._metadata())
                    or type(envelope["version"]) is not int or envelope["version"] != 1
                    or type(envelope["fitted"]) is not bool or envelope["task"] != self.task
                    or envelope["call_dtype"] != "float32"
                    or envelope["prediction_dtype"] != self._metadata()["prediction_dtype"]
                    or _json_bytes(envelope) != metadata):
                raise ValueError("closed canonical version-one envelope required")
            if not envelope["fitted"]:
                if envelope != self._metadata() or (directory / "model.ubj").exists():
                    raise ValueError("inconsistent unfitted metadata")
                self._accepted_fit = None
                return
            width, classes = envelope["feature_width"], envelope["classes"]
            if type(width) is not int or width <= 0:
                raise ValueError("invalid original feature width")
            if self.task == "classification":
                if (type(classes) is not list or len(classes) < 2 or any(type(v) is not int for v in classes)
                        or classes != list(range(len(classes)))):
                    raise ValueError("invalid class vocabulary")
                classes = tuple(classes)
            elif classes is not None:
                raise ValueError("regression metadata has classes")
            if envelope["writer_versions"] != _VERSIONS:
                raise ValueError(f"unsupported writer versions; requires {_VERSIONS}")
            if self.output_spec is not None and self.output_spec.shape != (() if classes is None else (len(classes),)):
                raise ValueError("explicit output_spec conflicts with restored facts")
            payload = (directory / "model.ubj").read_bytes()
            if envelope["native_sha256"] != hashlib.sha256(payload).hexdigest():
                raise ValueError("native payload digest mismatch")
        except (TypeError, ValueError, KeyError) as error:
            raise ValueError(f"Invalid xgboost1 metadata: {error}. Rebuild/refit; no legacy migration is supported.") from error
        native = self._new_native(classes)
        native.load_model(bytearray(payload))
        candidate = _Candidate(native, width, classes)
        self._validate_native(candidate)
        self._accepted_fit = _AcceptedFit(native, width, classes, payload, metadata)


class ClassifierModel(Model):
    """CPU classifier: call/proba (N,K) float32 and predict (N,) int64.

    Args: output_spec: Optional NumPy float32 (K,) assertion; **options: closed
        Model estimator options. Constructor is inert; errors follow Model.
    Class columns are the observed integer vocabulary 0..K-1, K >= 2.
    """

    def __init__(
        self, *, output_spec=None, n_estimators=100, max_depth=6,
        learning_rate=0.3, min_child_weight=1.0, gamma=0.0, subsample=1.0,
        colsample_bytree=1.0, colsample_bylevel=1.0, colsample_bynode=1.0,
        reg_alpha=0.0, reg_lambda=1.0, n_jobs=1, random_state=0,
    ):
        """Create an unfitted classifier from optional TensorSpec/closed scalar options.

        Returns an inert recipe; invalid configuration raises TypeError/ValueError
        as described by Model, without native construction or resource acquisition.
        """
        options = {name: value for name, value in locals().items() if name in _DEFAULTS}
        super().__init__("classification", output_spec=output_spec, **options)

    def predict_proba(self, x):
        """Return (N,K) float32 probabilities from a nonempty 2D NumPy batch.

        Admission/errors match Model.predict; NaN features supported, no coercion
        or accepted-state mutation. Columns are ordered 0..K-1.
        """
        return self._predict(x)


class RegressionModel(Model):
    """CPU scalar regressor: call/predict (N,) float32.

    Args: output_spec: Optional scalar NumPy float32 assertion; **options: closed
        Model estimator options. Constructor is inert; errors follow Model.
    """

    def __init__(
        self, *, output_spec=None, n_estimators=100, max_depth=6,
        learning_rate=0.3, min_child_weight=1.0, gamma=0.0, subsample=1.0,
        colsample_bytree=1.0, colsample_bylevel=1.0, colsample_bynode=1.0,
        reg_alpha=0.0, reg_lambda=1.0, n_jobs=1, random_state=0,
    ):
        """Create an unfitted regressor from optional TensorSpec/closed scalar options.

        Returns an inert recipe; invalid configuration raises TypeError/ValueError
        as described by Model, without native construction or resource acquisition.
        """
        options = {name: value for name, value in locals().items() if name in _DEFAULTS}
        super().__init__("regression", output_spec=output_spec, **options)


class BasicTraining(TrainFunction):
    """One-shot Dataset-owned CPU fit with no intermediate callbacks/observers.

    Args: verbose: Exact bool native fit option, default False. All other native
        fit controls are unsupported and rejected by the constructor signature.
    Raises: TypeError for invalid/unsupported options. Construction acquires nothing.
    One successful fit records one step/epoch and exact submitted examples via the
    private Models helper, without tree/iteration/loss telemetry.
    """

    supports_safe_points = False
    supports_observers = False

    def __init__(self, *, verbose=False):
        """Record exact-bool verbose configuration (False by default).

        Returns an inert trainer; raises TypeError for invalid configuration;
        imports/acquires nothing and retains no training continuation.
        """
        if type(verbose) is not bool:
            raise TypeError("XGBoost verbose must be an exact bool.")
        self.verbose = verbose

    def __call__(self, exp, *, callbacks=(), observer_session=None):
        """Accept one detached fit for an Experiment and return its native estimator.

        Args: exp: Experiment with this backend Model and finite canonical dense
            NumPy (x,y) Dataset, plain or authored batches; callbacks: intermediate
            requests (must be empty); observer_session: must be None.
        Raises: TypeError/ValueError for invalid owner/data/options/facts;
            NotImplementedError for intermediate controls before source open;
            native, cleanup, encoding and interruption errors propagate.
        Side Effects: Strictly materializes once, validates/fits/encodes privately,
            then installs the bundle and detached TrainState together. Earlier
            state survives preacceptance failure. Experiment owns terminal
            lifecycle/publication; later errors retain the accepted bundle.
        """
        if not isinstance(exp.model, Model):
            raise TypeError("XGBoost BasicTraining requires an XGBoost Model.")
        accepted = _fit_one_shot(
            exp, fit_candidate=lambda x, y: exp.model._fit_candidate(x, y, verbose=self.verbose),
            encode_candidate=exp.model._encode_candidate, callbacks=callbacks,
            observer_session=observer_session,
        )
        return accepted.native
