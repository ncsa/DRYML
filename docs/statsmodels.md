# Statsmodels: dense CPU OLS and binary Logit

`dryml.models.statsmodels` exports `Model`, `RegressionModel`, `ClassifierModel`
and `BasicTraining`. Imports, construction, definitions, output inference and
Method selection/preparation are passive: they do not import statsmodels,
sklearn, XGBoost, TensorFlow, Torch or JAX, scan the host, open data, or acquire
runtime resources. Statsmodels never requires the XGBoost or sklearn adapter.
Missing native dependencies at fit/fitted restore report `dryml[statsmodels]`.
Install with `python -m pip install 'dryml[statsmodels]'` in a Python 3.12
environment. The extra pins statsmodels 0.15.0, NumPy 2.1.3, SciPy 1.18.0 and
pandas 3.0.5, without sklearn or XGBoost. It declares the native dependencies
without Python markers that could silently omit the requested backend. Resolution
on another Python version is not support evidence: core's Python >=3.10 and
loose numerical dependencies do not expand the native reader gate. DRYML itself
does not install or provision dependencies.

## Public recipes and prediction

```python
import numpy as np
from dryml.models.statsmodels import RegressionModel, ClassifierModel

x = np.array([[0.0], [1.0], [2.0], [3.0]])
y = np.array([0.4, 1.6, 2.3, 3.7])
model = RegressionModel(intercept=True).fit(x, y)
prediction = model.predict(x[:1])  # (1,), float64
results = model.results          # borrowed complete native Results
standard_errors = results.bse

binary_recipe = ClassifierModel(intercept=True, threshold=0.5,
                                fit_options={"maxiter": 100, "tol": 1e-8, "disp": False})
```

- `Model(algorithm, *, intercept=False, threshold=0.5, fit_options=None,
  output_spec=None)` accepts `algorithm="ols"` or `"logit"`.
- `RegressionModel(*, intercept=False, fit_options=None, output_spec=None)` fixes
  OLS. `ClassifierModel(*, intercept=False, threshold=0.5, fit_options=None,
  output_spec=None)` fixes binary Logit.
- `fit(x, y)` returns the wrapper. It constructs, fits, validates and encodes a
  private candidate before installing one accepted bundle. It does not invent
  Experiment accounting. There is no incremental/inside-fit continuation API.
- `results` (also `mdl`) borrows the native Results wrapper, or is `None` when
  unfitted. `n_features_in_` is the original fitted feature width, excluding the
  adapter's intercept. Logit `classes_` returns a detached int64 `[0, 1]` array;
  unavailable fitted facts raise `ValueError`.

Direct call, `predict` and classifier `predict_proba` require a **nonempty 2D
dense NumPy batch**, shape `(N, F)`, with the original fitted width. Training
requires matching nonempty `(N,)` targets. Integer and float32/float64 arrays are
copied to float64 without modifying or retaining caller arrays. Boolean, complex,
object, float16, masked, matrix, foreign arrays, NaN/infinity, wrong rank/width,
empty input and unfitted prediction fail explicitly. No implicit foreign
`__array__` handoff or row dropping is supported. Logit targets must contain
exactly 0 and 1, **both observed**; finite float targets with these values are valid.

| Operation | OLS | Binary Logit |
|---|---|---|
| `model(x)` | `(N,)` float64 | `(N, 2)` float64 probabilities |
| `predict(x)` | `(N,)` float64 | `(N,)` int64 labels |
| `predict_proba(x)` | Not supplied | `(N, 2)` float64, columns `[0, 1]` |
| Selected element | float64 scalar | `(2,)` float64 |

Selected element calls explicitly lift one row and squeeze one result; selected
batches do not add a second axis. `Map` and prepared calls use the same semantics.
With the default `default_batched=None`, `model.learn()` accepts the same 2D
NumPy batch as eager `model(x)` or `model(x=x)` and caches that call layout.
First-call learning normalizes physical `(N,F)` to logical sample `(F,)` with
batch `N`, retaining raw-call selection for unknown batch intent. Explicit input
specifications still require logical sample shape `(F,)`.
Pure `infer_output_spec` knows scalar OLS and `(2,)` Logit **before fitting**,
with float64 dtype and propagated input batching. It never constructs native
models, calls `predict`, or fabricates samples. Optional `output_spec` is an
unbatched dense NumPy float64 assertion with that exact sample shape, also
checked against actual candidate output.

Backend-local selection and learning guards refuse foreign/unknown producer
backends before generic Method conversion planning, including `Map.prepare()`.
An explicitly authored upstream conversion producing NumPy remains valid. Method
preparations are process-local. They read current accepted weights after a
same-width refit; changing fitted width requires explicit re-preparation and stale
invokers raise `MethodError`. This includes trait-only selection such as
`model.find_implementation(backend='numpy', batch_mode='batched').prepared_invoker()`.
Unfitted selections capture width lazily on their first fitted invocation.
Invokers hold only a weak model reference and raise `MethodError` after the owner
is collected. No Method preparation is persisted.

## Scientific policy

`intercept` must be an exact bool and defaults to **False**. `True` prepends
**exactly one** float64 ones column at fit and predict. There is no constant
detection: caller-authored constants remain, including duplicate constants. The
fitted feature-width contract always refers to the caller's original columns.

OLS uses native `method="pinv"` rank semantics. Finite rank-deficient solutions
are admitted; native rank warnings propagate. OLS `fit_options` must be absent
or empty. No regularization, weights, formula, robust-covariance or solver options
are admitted by this adapter.

Logit uses Newton, default `maxiter=100`, `tol=1e-8`, `disp=False`. Its closed
`fit_options` dict permits only positive exact-int `maxiter`, positive finite
Python-real `tol`, and exact-bool `disp`. Full convergence output is forced.
`raise_on_perfect_prediction=True` is set on the private native instance only.
Acceptance requires `mle_retvals["converged"] is True` and finite parameters;
missing/false/truthy substitutes do not count as convergence. This is **not** a
universal separation detector. Native exceptions and warnings are preserved;
there is no process-global warning filter or native configuration change.

Both families use finite float64 design/targets and native `missing="raise"`,
retain every submitted row, and validate finite parameters and actual prediction
shape/dtype/domain before acceptance. `missing="raise"` alone is insufficient:
native OLS can return nonfinite parameters for infinite targets. Logit probability
columns are `[1-p, p]`; `threshold` must be a finite Python real strictly in
`(0, 1)`, default `0.5`, and **equality selects label 1**.

## Dataset training and Experiment lifecycle

```python
from dryml.data import ArrayDataset, Batch, as_supervised
from dryml.models import Experiment
from dryml.models.statsmodels import BasicTraining, RegressionModel

data = as_supervised(ArrayDataset((x, y)), 0, 1)
exp = Experiment(RegressionModel(intercept=True), BasicTraining(),
                 train_data=Batch(data, 3))
native_results = exp.train_fn(exp)  # direct trainer invocation, no managed publication
```

`BasicTraining()` takes no constructor controls; solver options are Model
configuration and therefore stable CDef identity. It consumes only the canonical
finite dense NumPy Dataset through `materialize_supervised(strict_numpy=True)`.
Plain examples and authored batches retain exact order and actual N, including
short final batches. There is no implicit foreign-framework training handoff.

The shared private one-shot helper validates detached TrainState accounting before
native work. Preparation, native fit/validation, `prep_eval` cleanup and final
encoding all finish before installation. Every preacceptance failure, including
interruption, nonconvergence, cleanup or encoding failure, preserves the prior
accepted Results, fitted facts, exact bytes and successful counters. One success
records **one fit step, one completed epoch, N examples, next_batch zero**, no
training loss, native iteration telemetry or epoch postlude.

The trainer declares `supports_safe_points=False` and `supports_observers=False`.
Its `callbacks=()` / `observer_session=None` keywords reject intermediate controls
before source acquisition. `ManagedConfig.callbacks` remains the supported,
separate terminal checkpoint lane. Managed `exp.train(managed=...)` returns the
existing exact final StateRef, preserving Experiment-owned terminal Artifacts,
history, association and completed-fit replay. A failed attempt may mark phase
failed while retaining previous successful-fit facts. A postacceptance publication
failure retains accepted work and propagates without claiming a terminal receipt.

## Complete Results persistence: `statsmodels1`

Models inherit `Model` plus **Serializable, not Pickleable**. Core's additive MRO
hooks receive the unchanged `statsmodels1` codec and write only:

- `metadata.json`: closed canonical version-one envelope, fitted/unfitted marker,
  task/algorithm, original width, classes, output dtypes, intercept/design policy,
  threshold/options, writer versions, Python patch detail and SHA-256 payload binding.
- `results.pkl`: fitted-only complete native Results saved with
  `remove_data=False`. There is no automatic duplicate `heavy.pkl`.

Native readers/writers are initially bound to **statsmodels 0.15.0, Python 3.12
major/minor, NumPy 2.1.3, SciPy 1.18.0 and pandas 3.0.5**. Python patch detail is
recorded separately, not used to promise other major/minor compatibility. sklearn
and XGBoost are not dependencies of this codec. Wider reader pairs require future
evidence; pickle is not a portable cross-version interchange format.

Encoding occurs **once per accepted fit**, after cleanup. Every subsequent save
re-emits the immutable accepted payload and canonical metadata exactly. Fitted
restore validates envelope, recipe/version/digest binding, decodes a private native
candidate and validates task/classes/design/dtypes/finite parameters/convergence
before installing it. It never probes native prediction or fabricates restore
samples. The exact loaded payload/envelope are retained until another successful
fit/restore. Native save/load/save bytes are known to change, and read-only
diagnostics populate caches, so reserialization cannot substitute for replay.
Unfitted restore clears all fitted facts without importing statsmodels.

Corrupt/incompatible/legacy state fails explicitly with no migration or Store
rewrite. Core remains responsible for identity bindings, receipts, live-cache
reuse and failed-entered-restore graph invalidation. A failed hook does not rescue
prior graph validity; recover with a fresh exact Repo load (`reuse_live="never"`).

Complete Results retains `exog`, `endog`, original training arrays, parameters,
standard errors, covariance/confidence intervals, likelihood, observation count
and summary inspection. **Training data persists on disk**. Full native Results
and their retained encoded bytes both occupy workload-sized memory. Nothing calls
`remove_data` or strips diagnostics. Borrowed read-only inspection is supported,
including native diagnostic cache population; external Results/model mutation and
same-model concurrent fit/restore/inspection mutation are unsupported. Mutating a
borrowed object cannot update the retained authoritative bytes.

## Current qualification

Bounded local Linux Python 3.12 tests on the exact stack above cover OLS intercept
off/on, duplicated/rank-deficient columns, binary nonseparable Logit, native
coefficient/probability/prediction parity at `1e-8`, and useful held-out quality:
OLS MSE below half its constant baseline and Logit accuracy at least majority
baseline + 0.10. They also cover one-row semantics, strict admission, failure
preservation, Method/Map refits, complete inspection/exact byte replay, fresh Repo
restore, entered-hook invalidation and local managed terminal history. Native
references remain mandatory at explicit execution, with no pass-by-skip behavior.

U5 additionally exercises both OLS and Logit through real Experiment Artifacts and
terminal managed callbacks, then real same-host CPU Core subprocess fit/publication
and fresh exact caller reopen. Independent submissions use separate state/control
Stores and temporary-resource roots. Representative Logit recovery tests prove
one native fit, one occurrence/time and unchanged exposure after associated
checkpoint, Artifact compute/completion or history failure. Replay re-emits exact
retained Results pickle/metadata bytes with native `save` disabled and must match
the retained full checkpoint StateRef before continuing its row. Pre-association
failure keeps live accepted facts but permits refit from prior authority; an
already-trained live Experiment is not a terminal-staging repair guarantee.
DirStore and committed ZipStore fitted/unfitted aliased graphs obey Core's
entered-hook invalidation, receipt clearing, cache eviction and fresh exact
recovery. Real workers also exercise missing/incompatible dependency refusal and
independent workload/teardown failure evidence under the existing protocol.
See [Models recovery boundary](models.md#opaque-fit-recovery-and-worker-evidence-u5)
and `tests/models/test_estimator_experiments.py` for the evidence scope.

The local mixed environment does not qualify isolated statsmodels-only installs,
Windows, other native/Python reader pairs, external data or GPU execution.
The [native CI configuration](testing.md#native-estimator-qualification) requires
Ubuntu and Windows Python 3.12 statsmodels-only cells, with sklearn/XGBoost absent
and all 56 assigned cases executed with zero skips. Those installation/platform
cells remain unqualified until CI passes on the exact revision.
