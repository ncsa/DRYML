"""Dense NumPy OLS/Logit with candidate-first, complete Results persistence.

Recipes and Method inference are passive. A single accepted bundle owns native
Results, fitted facts and immutable payload/envelope bytes. Results inspection is
borrowed; external mutation and same-model concurrent fit/restore are unsupported.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from importlib.metadata import PackageNotFoundError, version
import io
import json
import math
from pathlib import Path
import sys
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
    "statsmodels": "0.15.0", "numpy": "2.1.3", "scipy": "1.18.0",
    "pandas": "3.0.5", "python": "3.12",
}


def _positive_real(value, name):
    if type(value) not in (int, float):
        raise TypeError(f"statsmodels {name} must be a finite Python real number.")
    try:
        valid = math.isfinite(value) and value > 0
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"statsmodels {name} must be positive and finite.")
    return float(value)


def _options(algorithm, options):
    if options is None:
        options = {}
    if type(options) is not dict:
        raise TypeError("statsmodels fit_options must be a plain dict or None.")
    allowed = set() if algorithm == "ols" else {"maxiter", "tol", "disp"}
    if set(options) - allowed:
        raise ValueError("Unsupported statsmodels fit_options; OLS accepts none, Logit only maxiter/tol/disp.")
    if algorithm == "ols":
        return {}
    result = {"maxiter": 100, "tol": 1e-8, "disp": False, **options}
    if type(result["maxiter"]) is not int:
        raise TypeError("statsmodels maxiter must be a positive exact int.")
    if result["maxiter"] <= 0:
        raise ValueError("statsmodels maxiter must be positive.")
    result["tol"] = _positive_real(result["tol"], "tol")
    if type(result["disp"]) is not bool:
        raise TypeError("statsmodels disp must be an exact bool.")
    return result


def _native():
    try:
        installed = {name: version(name) for name in _VERSIONS if name != "python"}
        from statsmodels.regression.linear_model import OLS
        from statsmodels.discrete.discrete_model import Logit
        from statsmodels.iolib.smpickle import load_pickle
    except (ImportError, PackageNotFoundError) as error:
        raise ImportError("statsmodels requires the optional extra; install 'dryml[statsmodels]'.") from error
    installed["python"] = f"{sys.version_info.major}.{sys.version_info.minor}"
    if installed != _VERSIONS:
        raise ValueError(f"Unsupported statsmodels reader/writer versions; requires {_VERSIONS}, found {installed}.")
    return OLS, Logit, load_pickle, installed


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _validate_array(value, *, name, rank):
    if (not isinstance(value, np.ndarray)
            or isinstance(value, (np.ma.MaskedArray, np.matrix))):
        raise TypeError(f"statsmodels {name} requires a dense NumPy array; author conversion explicitly.")
    if value.dtype.kind not in "iuf" or (value.dtype.kind == "f" and value.dtype.itemsize not in (4, 8)):
        raise TypeError(f"statsmodels {name} requires real integer or float32/float64 dtype.")
    if value.ndim != rank or any(size == 0 for size in value.shape):
        raise ValueError(f"statsmodels {name} requires a nonempty rank-{rank} array.")
    if not np.isfinite(value).all():
        raise ValueError(f"statsmodels {name} rejects nonfinite values.")
    return value


def _array(value, *, name, rank):
    return _validate_array(value, name=name, rank=rank).astype(np.float64, copy=True)


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
    """Passive OLS or binary Logit recipe, with complete borrowed Results state.

    Args:
        algorithm: Exact algorithm spelling ``'ols'`` or ``'logit'``.
        intercept: Exact bool, default False. True prepends exactly one ones column
            even when the caller already supplied a constant.
        threshold: Finite Python int/float strictly between zero and one, default
            0.5; binary prediction equality selects class one. Inert for OLS.
        fit_options: Plain dict or None. OLS accepts only empty options. Logit
            accepts positive exact-int maxiter (100), positive finite tol (1e-8),
            exact-bool disp (False). Solver is fixed: pinv / Newton respectively.
        output_spec: Optional unbatched dense NumPy float64 TensorSpec assertion,
            scalar for OLS or shape (2,) for Logit.

    Raises:
        TypeError, ValueError: Invalid recipe/options/specification.

    Construction imports no native framework and acquires no resources/data.
    Configuration belongs to CDef; fitted Results, training arrays and facts are
    Serializable state, never definition parameters. Direct calls require 2D
    batches; selected elements explicitly lift and squeeze one row. All admitted
    integer/float32/float64 arrays are copied to float64 without caller mutation.
    """

    native_backend = "numpy"
    state_codec = "statsmodels1"

    def __init__(self, algorithm, *, intercept=False, threshold=0.5, fit_options=None, output_spec=None):
        """Record Model's inert recipe; arguments/errors/side effects follow Model.

        Returns an unfitted wrapper. No native model is constructed or imported.
        """
        if algorithm not in ("ols", "logit"):
            raise ValueError("statsmodels algorithm must be ols or logit.")
        if type(intercept) is not bool:
            raise TypeError("statsmodels intercept must be an exact bool.")
        threshold = _positive_real(threshold, "threshold")
        if threshold >= 1:
            raise ValueError("statsmodels threshold must be strictly between zero and one.")
        self.algorithm = algorithm
        self.task = "regression" if algorithm == "ols" else "classification"
        self.intercept = intercept
        self.threshold = threshold
        self.fit_options = _options(algorithm, fit_options)
        self.output_spec = output_spec
        if output_spec is not None:
            if (not isinstance(output_spec, TensorSpec) or output_spec.batched
                    or output_spec.backend is not Backend.numpy or output_spec.layout is not Layout.DENSE
                    or output_spec.dtype.name != "float64"):
                raise TypeError("statsmodels output_spec must be an unbatched dense NumPy float64 TensorSpec.")
            if output_spec.shape != (() if algorithm == "ols" else (2,)):
                raise ValueError("statsmodels output_spec requires scalar OLS or binary Logit shape (2,).")
        self._accepted_fit = None

    def __getattr__(self, name):
        """Read bundle-owned inspection facts without native imports or duplication.

        ``results`` (also ``mdl``) borrows complete native Results, None if unfitted.
        ``n_features_in_`` is the original int width; ``classes_`` returns a detached
        int64 [0,1] array for Logit. These fitted facts raise ValueError if absent;
        other names raise AttributeError. Results diagnostics may populate native
        caches; external mutation is unsupported and cannot rewrite retained bytes.
        """
        if name in ("results", "mdl"):
            fitted = vars(self).get("_accepted_fit")
            return None if fitted is None else fitted.native
        if name == "n_features_in_":
            return self._require_fit().feature_width
        if name == "classes_":
            fitted = self._require_fit()
            if fitted.classes is None:
                raise ValueError("statsmodels OLS has no classes_.")
            return np.asarray(fitted.classes, dtype=np.int64)
        raise AttributeError(name)

    def _require_fit(self):
        if self._accepted_fit is None:
            raise ValueError("statsmodels model is unfitted; call fit or BasicTraining first.")
        return self._accepted_fit

    def _design(self, x):
        return np.column_stack((np.ones(len(x), dtype=np.float64), x)) if self.intercept else x

    def _new_native(self, x, y):
        OLS, Logit, _, _ = _native()
        native = (OLS if self.algorithm == "ols" else Logit)(y, self._design(x), missing="raise")
        if self.algorithm == "logit":
            native.raise_on_perfect_prediction = True
        return native

    def _validate_native(self, candidate):
        OLS, Logit, _, _ = _native()
        from statsmodels.regression.linear_model import RegressionResultsWrapper, OLSResults
        from statsmodels.discrete.discrete_model import BinaryResultsWrapper, LogitResults

        result = candidate.native
        wrapper, results_type, model_type = ((RegressionResultsWrapper, OLSResults, OLS)
                                             if self.algorithm == "ols"
                                             else (BinaryResultsWrapper, LogitResults, Logit))
        if (type(result) is not wrapper or type(result._results) is not results_type
                or type(result.model) is not model_type):
            raise ValueError("Inconsistent statsmodels native Results/task metadata.")
        width = candidate.feature_width + int(self.intercept)
        design, targets, params = result.model.exog, result.model.endog, result.params
        for name, array, rank in (("design", design, 2), ("targets", targets, 1), ("params", params, 1)):
            if (not isinstance(array, np.ndarray) or array.ndim != rank
                    or array.dtype != np.dtype("float64") or not np.isfinite(array).all()):
                raise ValueError(f"Invalid statsmodels native {name}: finite float64 array required.")
        if (design.shape != (len(targets), width) or len(targets) == 0
                or params.shape != (width,) or result.nobs != len(targets)):
            raise ValueError("Inconsistent statsmodels native feature/row metadata.")
        if self.intercept and not np.array_equal(design[:, 0], np.ones(len(targets))):
            raise ValueError("Inconsistent statsmodels native intercept design.")
        if (not np.array_equal(result.model.data.orig_exog, design)
                or not np.array_equal(result.model.data.orig_endog, targets)):
            raise ValueError("statsmodels complete Results must retain original training arrays.")
        if self.algorithm == "logit":
            if candidate.classes != (0, 1) or not np.array_equal(np.unique(targets), [0, 1]):
                raise ValueError("Inconsistent statsmodels native binary class metadata.")
            if (type(getattr(result, "mle_retvals", None)) is not dict
                    or result.mle_retvals.get("converged") is not True):
                raise ValueError("statsmodels Logit requires convergence exactly True.")
            if result.model.raise_on_perfect_prediction is not True:
                raise ValueError("statsmodels Logit requires instance-local perfect prediction raising.")
        elif candidate.classes is not None:
            raise ValueError("statsmodels OLS metadata cannot have classes.")

    def _check_output(self, positive, *, rows):
        if (not isinstance(positive, np.ndarray) or positive.shape != (rows,)
                or positive.dtype != np.dtype("float64") or not np.isfinite(positive).all()):
            raise ValueError("statsmodels returned inconsistent/nonfinite prediction shape/dtype.")
        if self.algorithm == "logit":
            if not ((positive >= 0) & (positive <= 1)).all():
                raise ValueError("statsmodels returned invalid binary probabilities.")
            result = np.column_stack((1 - positive, positive))
        else:
            result = positive
        if self.output_spec is not None and not self.output_spec.compatible_with_shape(result.shape[1:]):
            raise ValueError("statsmodels prediction violates explicit output_spec.")
        return result

    def _fit_candidate(self, x, y):
        x = _array(x, name="features", rank=2)
        y = _array(y, name="targets", rank=1)
        if len(x) != len(y):
            raise ValueError("statsmodels features and targets require matching example counts.")
        classes = (0, 1) if self.algorithm == "logit" else None
        if classes is not None and not np.array_equal(np.unique(y), [0, 1]):
            raise ValueError("statsmodels Logit requires both binary labels exactly 0 and 1.")
        native = self._new_native(x, y)
        if self.algorithm == "ols":
            result = native.fit(method="pinv")
        else:
            result = native.fit(method="newton", full_output=True, **self.fit_options)
        candidate = _Candidate(result, x.shape[1], classes)
        self._validate_native(candidate)
        self._check_output(result.predict(result.model.exog), rows=len(x))
        return candidate

    def _metadata(self, *, candidate=None, payload=b"", versions=None, python_patch=None):
        return {
            "version": 1, "fitted": candidate is not None, "task": self.task,
            "algorithm": self.algorithm, "intercept": self.intercept,
            "design_policy": "prepend_one" if self.intercept else "original",
            "threshold": self.threshold, "fit_options": self.fit_options,
            "feature_width": None if candidate is None else candidate.feature_width,
            "classes": None if candidate is None or candidate.classes is None else list(candidate.classes),
            "prediction_dtype": "int64" if self.task == "classification" else "float64",
            "call_dtype": "float64", "writer_versions": versions, "writer_python_patch": python_patch,
            "native_sha256": None if candidate is None else hashlib.sha256(payload).hexdigest(),
        }

    def _encode_candidate(self, candidate):
        _, _, _, versions = _native()
        stream = io.BytesIO()
        candidate.native.save(stream, remove_data=False)
        payload = stream.getvalue()
        metadata = _json_bytes(self._metadata(candidate=candidate, payload=payload, versions=versions,
                                             python_patch=sys.version.split()[0]))
        return _AcceptedFit(candidate.native, candidate.feature_width, candidate.classes, payload, metadata)

    def fit(self, x, y):
        """Fit privately from dense finite NumPy (N,F) features and (N,) targets.

        Integer/float32/float64 arrays are copied to float64; Logit requires both
        labels exactly 0 and 1. Returns self after native validation and one full
        Results encoding. TypeError/ValueError reject invalid data/fitted facts;
        missing dependencies raise actionable ImportError. Native warnings/errors
        and BaseException interruptions propagate. All preacceptance failures
        preserve prior Results, facts and bytes. No Experiment accounting is added.
        Same-model concurrent fit/restore and borrowed Results mutation are unsupported.
        """
        candidate = self._fit_candidate(x, y)
        accepted = self._encode_candidate(candidate)
        self._accepted_fit = accepted
        return self

    def _predict(self, x, *, labels=False):
        x = _array(x, name="features", rank=2)
        fitted = self._require_fit()
        if x.shape[1] != fitted.feature_width:
            raise ValueError("statsmodels feature width differs from fitted original feature width.")
        result = self._check_output(fitted.native.predict(self._design(x)), rows=len(x))
        if labels and fitted.classes is not None:
            return (result[:, 1] >= self.threshold).astype(np.int64)
        return result

    def predict(self, x):
        """Return (N,) float64 OLS predictions or int64 binary threshold labels.

        x must be a nonempty finite real 2D NumPy batch with original fitted width.
        Invalid inputs/unfitted state raise TypeError/ValueError before prediction;
        no foreign coercion occurs. Copies input to float64; native errors propagate.
        Accepted state is unchanged (native diagnostic caches may be populated).
        """
        return self._predict(x, labels=True)

    def _runtime_selection_facts(self, args, kwargs):
        _validate_array(args[0] if args else kwargs.get("x"), name="features", rank=2)
        return Backend.numpy, None

    @traits(backend="numpy")
    def raw_call(self, x):
        """Return float64 (N,) OLS / (N,2) Logit; admission/errors match predict."""
        return self._predict(x)

    @traits(backend="numpy", batch_mode="batched")
    def batched_call(self, x):
        """Predict one selected batch without another axis; behavior follows raw_call."""
        return self._predict(x)

    @traits(backend="numpy", batch_mode="element")
    def element_call(self, x):
        """Lift a finite real 1D NumPy row and squeeze to float64 scalar / (2,).

        Copies input to float64. Invalid inputs, width or unfitted state raise
        TypeError/ValueError; native errors propagate without accepted-state changes.
        """
        x = _validate_array(x, name="element features", rank=1)
        return self._predict(x[None, :])[0]

    def _input_spec(self, spec):
        if (not isinstance(spec, TensorSpec) or spec.backend is not Backend.numpy
                or spec.layout is not Layout.DENSE or spec.shape is None or len(spec.shape) != 1
                or spec.dtype.kind not in ("int", "uint", "float")
                or (spec.dtype.kind in ("int", "uint") and spec.dtype.bits not in (8, 16, 32, 64))
                or (spec.dtype.kind == "float" and spec.dtype.bits not in (32, 64))):
            raise TypeError("statsmodels requires a dense real NumPy feature TensorSpec; author conversion explicitly.")
        if spec.shape[0] != Dynamic and spec.shape[0] <= 0:
            raise ValueError("statsmodels feature width must be positive.")
        if self._accepted_fit is not None and spec.shape[0] not in (Dynamic, self._accepted_fit.feature_width):
            raise ValueError("statsmodels spec feature width differs from fitted width.")

    def infer_output_spec(self, input_spec, *additional_input_specs):
        """Infer known scalar OLS / (2,) Logit float64 output without native calls.

        input_spec is a dense NumPy real feature TensorSpec, sample shape (F,)
        with optional batching; additional specs are unsupported. Returns output
        with matching batching, including before fitting. TypeError/ValueError
        reject invalid specs or wrong fitted width. No imports, data, prediction,
        construction, resource acquisition or Model mutation occur.
        """
        if additional_input_specs:
            raise TypeError("statsmodels Model accepts exactly one input specification.")
        self._input_spec(input_spec)
        spec = self.output_spec or TensorSpec("float64", shape=() if self.algorithm == "ols" else (2,), backend="numpy")
        return match_input_batch(spec, input_spec)

    def _admit_selection(self, input_spec, additional=(), backend=None):
        try:
            if additional or (backend is not None and Backend(backend) is not Backend.numpy):
                raise TypeError("statsmodels accepts one NumPy input only.")
            if input_spec is not None:
                self._input_spec(input_spec)
        except (TypeError, ValueError) as error:
            raise ImplementationSelectionError("conflict") from error

    def find_implementation(self, input_spec=None, *additional_input_specs, backend=None,
                            batch_mode=None, output_spec=None):
        """Select a unary NumPy call, refusing foreign specs before generic fallback.

        Parameters follow Method.find_implementation: optional unary TensorSpec,
        backend/batch constraints and result assertion. Returns a pure selected
        carrier; invalid facts raise ImplementationSelectionError. Imports,
        acquisition, conversion and preparation-state mutation never occur.
        """
        self._admit_selection(input_spec, additional_input_specs, backend)
        selected = super().find_implementation(input_spec, backend=backend, batch_mode=batch_mode,
                                               output_spec=output_spec)
        return self._specialize(selected, input_spec)

    def _prepare_implementation(self, input_spec, *, backend, batch_mode):
        self._admit_selection(input_spec, backend=backend)
        selected = super()._prepare_implementation(input_spec, backend=backend, batch_mode=batch_mode)
        return self._specialize(selected, input_spec)

    def _specialize(self, selected, input_spec):
        if input_spec is None:
            return selected
        output = self.infer_output_spec(input_spec)
        facts = None if self._accepted_fit is None else self._accepted_fit.feature_width
        receiver_ref, name = weakref.ref(self), selected.name

        def invoke(x):
            nonlocal facts
            receiver = receiver_ref()
            if receiver is None:
                raise MethodError("The selected statsmodels Model is no longer live.")
            current = receiver._require_fit()
            if facts is not None and facts != current.feature_width:
                raise MethodError("statsmodels fitted specifications changed; explicitly re-prepare the Method/Map.")
            if facts is None:
                facts = current.feature_width
            return getattr(receiver, name)(x)

        return replace(selected, _invoker=invoke, _output_inference=lambda children: output)

    @staticmethod
    def _check_codec(codec):
        if codec != "statsmodels1":
            raise ValueError("Unsupported statsmodels codec; requires statsmodels1. Legacy state is unsupported; rebuild and refit.")

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        """Write retained complete Results bytes and canonical metadata, never re-encode.

        dest_dir is Core's empty payload directory; codec must be 'statsmodels1'
        or ValueError is raised. Writes metadata.json and fitted results.pkl only;
        filesystem errors propagate. No automatic heavy.pkl, native import or data
        removal occurs. Read-only diagnostics cannot change these retained bytes.
        """
        self._check_codec(codec)
        fitted = self._accepted_fit
        Path(dest_dir, "metadata.json").write_bytes(_json_bytes(self._metadata()) if fitted is None else fitted.metadata)
        if fitted is not None:
            Path(dest_dir, "results.pkl").write_bytes(fitted.payload)

    def restore_state_from_dir_imp(self, src_dir, *, codec):
        """Validate/decode a private complete Results candidate, then install once.

        src_dir is Core's authenticated payload directory; codec is statsmodels1.
        ValueError rejects unsupported codec, noncanonical/closed metadata, version
        mismatch, corrupt pickle or inconsistent native facts. Missing dependencies
        raise ImportError; filesystem errors propagate. Fitted restore lazily admits
        native dependencies but never probes predict or fabricates samples. Retains
        exact loaded bytes; unfitted restore clears fitted state. Core owns identity,
        receipts, graph invalidation and cache eviction after entered-hook failures;
        this hook neither rescues validity nor rewrites Store authority.
        """
        self._check_codec(codec)
        directory = Path(src_dir)
        if (directory / "heavy.pkl").exists():
            raise ValueError("Legacy statsmodels heavy.pkl is unsupported; rebuild and refit.")
        metadata = (directory / "metadata.json").read_bytes()
        try:
            envelope = json.loads(metadata)
            expected = self._metadata()
            varying = {"fitted", "feature_width", "classes", "writer_versions", "writer_python_patch", "native_sha256"}
            if (type(envelope) is not dict or set(envelope) != set(expected)
                    or type(envelope["version"]) is not int or type(envelope["fitted"]) is not bool
                    or type(envelope["intercept"]) is not bool
                    or any(_json_bytes(envelope[key]) != _json_bytes(expected[key])
                           for key in set(expected) - varying)
                    or _json_bytes(envelope) != metadata):
                raise ValueError("closed canonical version-one recipe envelope required")
            if not envelope["fitted"]:
                if envelope != expected or (directory / "results.pkl").exists():
                    raise ValueError("inconsistent unfitted metadata")
                self._accepted_fit = None
                return
            width, classes = envelope["feature_width"], envelope["classes"]
            if type(width) is not int or width <= 0:
                raise ValueError("invalid original feature width")
            if self.task == "classification":
                if type(classes) is not list or classes != [0, 1] or any(type(v) is not int for v in classes):
                    raise ValueError("invalid binary class vocabulary")
                classes = tuple(classes)
            elif classes is not None:
                raise ValueError("OLS metadata has classes")
            patch = envelope["writer_python_patch"]
            if (envelope["writer_versions"] != _VERSIONS or type(patch) is not str
                    or len(patch.split(".")) != 3 or not all(v.isdigit() for v in patch.split("."))
                    or ".".join(patch.split(".")[:2]) != _VERSIONS["python"]):
                raise ValueError(f"unsupported writer versions; requires {_VERSIONS}")
            payload = (directory / "results.pkl").read_bytes()
            if envelope["native_sha256"] != hashlib.sha256(payload).hexdigest():
                raise ValueError("native payload digest mismatch")
        except (TypeError, ValueError, KeyError) as error:
            raise ValueError(f"Invalid statsmodels1 metadata: {error}. Rebuild/refit; no legacy migration is supported.") from error
        _, _, load_pickle, _ = _native()
        try:
            stream = io.BytesIO(payload)
            native = load_pickle(stream)
            if stream.read():
                raise ValueError("trailing Results payload bytes")
        except Exception as error:
            raise ValueError("Invalid statsmodels1 complete Results pickle; rebuild/refit with the matching stack.") from error
        candidate = _Candidate(native, width, classes)
        self._validate_native(candidate)
        self._accepted_fit = _AcceptedFit(native, width, classes, payload, metadata)


class ClassifierModel(Model):
    """Binary Logit: float64 [0,1] probability columns and int64 threshold labels.

    Args: intercept: Exact bool False; threshold: finite Python real in (0,1),
        default .5; fit_options: optional maxiter/tol/disp dict; output_spec:
        optional NumPy float64 (2,) assertion. Defaults/errors follow Model.
    Constructor is passive; native Results retains training data after fitting.
    """

    def __init__(self, *, intercept=False, threshold=0.5, fit_options=None, output_spec=None):
        """Return an inert Logit recipe; Model documents parameters/errors/side effects."""
        super().__init__("logit", intercept=intercept, threshold=threshold,
                         fit_options=fit_options, output_spec=output_spec)

    def predict_proba(self, x):
        """Return (N,2) float64 probabilities ordered [0,1] for a dense NumPy batch.

        Input admission, copying, exceptions and side effects match Model.predict.
        No implicit foreign conversion occurs; threshold does not change probabilities.
        """
        return self._predict(x)


class RegressionModel(Model):
    """OLS pseudoinverse regression with (N,) float64 predictions.

    Args: intercept: Exact bool False; fit_options: None/empty dict only;
        output_spec: optional scalar NumPy float64 assertion. Defaults/errors
        follow Model. Constructor is passive; finite rank-deficient fits are valid.
    """

    def __init__(self, *, intercept=False, fit_options=None, output_spec=None):
        """Return an inert OLS recipe; Model documents parameters/errors/side effects."""
        super().__init__("ols", intercept=intercept, fit_options=fit_options, output_spec=output_spec)


class BasicTraining(TrainFunction):
    """One-shot finite canonical NumPy Dataset fit with Model-owned solver options.

    Takes no constructor options; imports/acquires nothing. Unsupported arguments
    raise TypeError. A successful call records one coarse step/epoch and exact N,
    next_batch zero, without loss/solver telemetry or inside-fit continuation.
    """

    supports_safe_points = False
    supports_observers = False

    def __call__(self, exp, *, callbacks=(), observer_session=None):
        """Fit an Experiment privately and return its accepted complete native Results.

        exp must own this backend Model and finite canonical dense NumPy (x,y)
        Dataset, either plain or authored batches. callbacks must be empty and
        observer_session None, otherwise NotImplementedError/TypeError precedes
        source acquisition. Invalid owners/data/facts raise TypeError/ValueError;
        native, cleanup, encoding and BaseException failures propagate. Strict
        ordered materialization, preparation/cleanup, validation and one encoding
        precede atomic local bundle/accounting acceptance with rollback. Prior
        success survives preacceptance errors; Experiment owns phase, final StateRef,
        terminal callbacks/history and replay. Later publication failures retain
        accepted work. No native iteration callbacks or training continuation exist.
        """
        if not isinstance(exp.model, Model):
            raise TypeError("statsmodels BasicTraining requires a statsmodels Model.")
        accepted = _fit_one_shot(exp, fit_candidate=exp.model._fit_candidate,
                                 encode_candidate=exp.model._encode_candidate,
                                 callbacks=callbacks, observer_session=observer_session)
        return accepted.native
