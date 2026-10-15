# CPU XGBoost models

`dryml.models.xgb` supplies `Model`, `ClassifierModel`, `RegressionModel`, and
`BasicTraining` through the maintained Model/Method/Dataset/Serializable contracts.
Importing the backend and constructing its configuration are passive: no XGBoost
or sklearn import, estimator construction, host scan, data acquisition, or native
resource allocation occurs. Missing native dependencies at fit/fitted restore
raise an actionable error naming `dryml[xgboost]`; DRYML never installs packages.

## Supported recipes and arrays

```python
import numpy as np
from dryml.data import ArrayDataset, Batch, as_supervised
from dryml.models import Experiment
from dryml.models.xgb import ClassifierModel, BasicTraining

x = np.array([[-1.0], [-0.5], [0.5], [1.0]], dtype=np.float32)
y = np.array([0, 0, 1, 1], dtype=np.int64)
model = ClassifierModel(n_estimators=8, max_depth=2, min_child_weight=0)
data = Batch(as_supervised(ArrayDataset((x, y)), 0, 1), 3)
experiment = Experiment(model, BasicTraining(), train_data=data)
experiment.train_fn(experiment)  # returns the accepted borrowed native estimator
probabilities = model(x)         # (4, 2), float32
labels = model.predict(x)        # (4,), int64
```

For publication, call `experiment.train(managed=ManagedConfig(state_repo=repo))`
with an explicitly configured Repo. Its result is the terminal `StateRef`, not the
native estimator. Experiment owns terminal processing, Artifacts/history, final
association, and recovery. This opaque trainer has no intermediate safe points or
observers; terminal managed callbacks retain their existing separate lifecycle.

Constructors take keyword-only `output_spec=None` and the closed, explicitly named
estimator options below. Both concrete classes capture every option default in
their CDef. The backend-local base additionally
takes `task='classification'` or `task='regression'` as its first argument. These
are independent model instances; no global XGBoost configuration is changed.

Only CPU `gbtree` with `hist` is supported. DRYML selects `binary:logistic` for two
classes, `multi:softprob` for more classes, and `reg:squarederror` for regression,
with `verbosity=0`. The closed estimator option set is:

| Option | Default | Accepted values |
|---|---:|---|
| `n_estimators` | 100 | exact integer 1..2147483647 |
| `max_depth` | 6 | exact integer 0..2147483647; 0 uses native unlimited-depth semantics |
| `n_jobs` | 1 | exact integer 1..2147483647; caller must bound native threads |
| `random_state` | 0 | exact integer 0..2147483647 |
| `learning_rate` | 0.3 | finite Python real in (0, 1] |
| `min_child_weight` | 1.0 | finite nonnegative Python real |
| `gamma` | 0.0 | finite nonnegative Python real |
| `subsample` | 1.0 | finite Python real in (0, 1] |
| `colsample_bytree`, `colsample_bylevel`, `colsample_bynode` | 1.0 | finite Python real in (0, 1] |
| `reg_alpha` | 0.0 | finite nonnegative Python real |
| `reg_lambda` | 1.0 | finite nonnegative Python real |

Nonnegative real options must also be representable in native float32. Booleans
are not integers/reals for this configuration. Unknown options, callable
controls, and overrides of `device`, `booster`, `tree_method`, `objective` or
`verbosity` fail during construction. Stable options/output specs reside in the
CDef; fitting leaves the definition unchanged.

`BasicTraining(*, verbose=False)` accepts only an exact boolean `verbose`. The
same is true of `model.fit(x, y, *, verbose=False)`, which returns `model` and
does not invent Experiment accounting. Continuation (`xgb_model`), early stopping,
evaluation sets, sample weights, margins/offsets, callbacks and custom objectives
are unsupported. Unknown trainer/fit keywords and intermediate callback/observer
requests fail before acquiring a Dataset or constructing a candidate.

The trainer strictly materializes a finite canonical Dataset yielding `(x, y)`.
Plain examples and caller-authored batches preserve order/dtypes and submit the
same arrays, including short final batches. Collection counts actual submitted
examples. A successful invocation records one coarse fit step, one completed
epoch, exact N exposure and next-batch zero; no loss/tree/iteration telemetry or
pending epoch postlude is fabricated.

Direct `fit` requires nonempty dense NumPy ndarrays: features `(N,F)` and targets
`(N,)`, matching N and positive F. Real signed/unsigned integer, float32 and
float64 dtypes are accepted; masked arrays, matrices, bool, float16, complex,
object, sparse and foreign arrays are rejected without coercion. Classification
targets must have integer dtype and at least two observed classes exactly
`0..K-1`. Regression targets must be finite. Feature NaNs are supported;
infinities are rejected. Dataset declaration/runtime admission follows the
Data-owned strict NumPy contract and does not insert conversion edges.

## Prediction and pure Method metadata

| API | Classification | Regression |
|---|---|---|
| Direct `model(x)` | `(N,K)` float32 probabilities | `(N,)` float32 |
| `predict_proba(x)` (classifier only) | same probabilities, columns `0..K-1` | unavailable |
| `predict(x)` | `(N,)` int64 labels | `(N,)` float32 |
| Selected element call | `(K,)` float32 probabilities | scalar float32 |
| Selected batch call | `(N,K)` float32 probabilities | `(N,)` float32 |

Direct prediction requires a nonempty 2D NumPy batch of fitted original width.
Unfitted state, invalid arrays/width, infinity, or inconsistent native output
fails with `TypeError`/`ValueError`; native failures propagate. No accepted state
is changed. Selected element calls explicitly lift a 1D example to one row and
remove exactly that batch axis. `Map` and reusable preparation use the same
semantics. A Method-local eager `default_batched` does not authorize raw 1D calls;
use an explicit element TensorSpec/Map for element adaptation.

`infer_output_spec` uses only pure recipe/fitted facts. Its input must be one
dense real NumPy TensorSpec with sample shape `(F,)`; the result preserves its
batch axis. Regression output is known even while unfitted. Classification K is
a fitted fact, so prefit inference needs an explicit unbatched dense NumPy
float32 `output_spec=TensorSpec('float32', shape=(K,), backend='numpy')`, K >= 2.
Regression's optional assertion has sample shape `()`. Assertions are checked
against fitted/native results, rather than overriding their actual shape/dtype.

Selection, learning/preparation and `Map` reject foreign or missing-backend
producer specs **before** inherited generic conversion fallback, acquisition or
adapter activation. A caller-authored conversion that produces NumPy is allowed.
Raw prediction also refuses foreign arrays before `__array__` coercion.

Prepared calls read current accepted weights on same-fact refits. Changed fitted
feature width/class vocabulary requires explicit re-preparation; stale selected
calls/Maps fail instead of silently returning changed output facts. Preparations
are process-local and are never saved in model state.

`mdl` borrows the accepted native estimator (None if unfitted),
`n_features_in_` exposes the original fitted width and `classes_` returns a
detached int64 class array. Missing fitted facts raise `ValueError`. Native
inspection is borrowed, not a second ownership lane: external mutation and
direct concurrent fit/restore of the same model are unsupported. Separate model
instances have independent native state.

## Acceptance and `xgboost1` persistence

Fit constructs a private estimator, validates task/feature/class facts and
finite prediction/probability shape/dtypes, finishes model preparation cleanup
for Experiment training, then encodes the candidate once. Only afterward is the
accepted native/fact/byte bundle installed. The private Models one-shot helper
installs it with detached TrainState accounting under its local rollback boundary.
Native fitting, validation, cleanup, encoding and interruption failures preserve
the prior accepted bundle and successful-fit counters. Later managed publication
failure retains the accepted fit and propagates without claiming completion.

These models inherit `Serializable`, never `Pickleable`. The `xgboost1` payload
contains only:

- `model.ubj`: native versioned UBJSON model IO, when fitted. Native sklearn
  wrapper model IO is invoked once in a disposable directory; restoration loads
  its retained byte buffer into a detached fresh wrapper.
- `metadata.json`: closed canonical UTF-8 JSON version 1. Exact fields are
  `version`, `fitted`, `task`, `feature_width`, `classes`, `prediction_dtype`,
  `call_dtype`, `writer_versions`, `native_sha256`. The digest binds the native
  bytes to the envelope; it does not replace core Store authentication.

Fitted writer versions are recorded below; unfitted envelopes have null
feature/class/version/digest facts and no native payload. Fit/restore retains the
exact native bytes and canonical metadata, so unchanged re-save does not call
native serialization again or depend on native save/load/save stability. No
automatic `heavy.pkl` or extra native pickle is produced. Unfitted restore clears
all fitted facts. Recipe/training options remain in the definition, not the
native-state envelope.

The initial reader/writer accepts exactly XGBoost **3.4.2**, sklearn **1.9.0**,
NumPy **2.1.3**, SciPy **1.18.0**, pandas **3.0.5**, and Python major/minor
**3.12**. Fit/fitted restore checks installed package versions only at native-use
admission. Restore requires the same exact envelope versions and validates native
objective, feature width, class/target counts, CPU and gbtree facts before install.
Python patch versions are not part of this reader gate. This deliberately does
not generalize XGBoost's upstream portable model IO promise into DRYML support.

Malformed/noncanonical/unknown metadata, inconsistent native/fitted facts,
unsupported codec/version and legacy `model.text` compute state fail actionably.
There is no automatic migration, Store rewrite, hot reload or live-Object rescue.
Core owns failed-entered-restore-hook invalidity, IDs, bindings and cache eviction;
load a fresh exact graph with `reuse_live='never'` after such failure.

## Evidence boundary

Bounded CPU tests exercise binary/multiclass/regression native parity at `1e-6`,
held-out accuracy at least majority + 0.10 or MSE below half the constant baseline,
plain/short-batch exposure, pure metadata, selected/prepared predictions,
candidate failures, exact fitted/unfitted Repo reconstruction and byte-preserving
re-save on the specified Linux/Python 3.12 stack. Existing U1 native reference
tests are retained. Dependency-light import/config/spec/admission contracts stay
in `tests/models/test_estimator_contracts.py`; exact native qualification stays
in the heavy-by-default `tests/models/test_xgb_models.py`.

U5 adds real Experiment/Artifact/callback associations and same-host Core CPU
workers for both classification and regression in
`tests/models/test_estimator_experiments.py`. Two independent submissions per
family publish into separate state/control Stores and scratch roots, then reopen
fresh exact caller graphs with matching predictions and retained payload digests.
Representative regression recovery cases inject pre-association publication,
post-association, Artifact compute/completion, and history publication failures.
Only an **associated** terminal checkpoint can replay one occurrence/time/N
without refitting; replay must reproduce its exact StateRef from retained UBJSON
and metadata before continuing history, with native serialization disabled.
Pre-association failure retains live accepted work but reopening prior authority
may refit. DirStore/ZipStore graph restore cases verify fitted/unfitted state,
entered-hook invalidation and fresh recovery. Worker dependency and teardown
failures retain honest, separate invocation/cleanup outcomes. See
[Models recovery boundary](models.md#opaque-fit-recovery-and-worker-evidence-u5).

Windows, isolated installation, cross-version models, GPU, sparse/categorical
data and broader platform/recovery matrices remain unqualified. Packaging/extras
and wider CI qualification are later
rollout units. See [Models](models.md), [Methods](methods.md), and [Data](data.md)
for lower-layer ownership and lifecycle contracts.
