# Dataset traversal implementation qualification

## Status

The traversal, continuation, Accumulate, accepted-training-state, and native
consumer implementation is present on `fix/real-usage-followups`. The plan is
**not complete**: comparative performance and updated-tree routine qualification
remain blocking. The complete routine gate passed at `238a2cbd`, but the later
`314c9057` run failed a stale collection inventory and exceeded 900 seconds.
The inventory is corrected below; passing focused checks does not waive either
completion gate.

Latest corrections cover retained native call preparation, JAX argument
commitment, and redundant explicit-TF scalar loss reduction. Earlier corrections
include the linear Array/Map-to-Batch kernel, the
[explicit TF/Keras recipe loss](#fresh-comparison-tfkeras-recipe-loss-correction),
and confirmation instrumentation identity.
Earlier exhausted reports remain **unverified**. The last successful complete
routine runner passed **5,149 tests with 25 skips in 898.23 seconds**. Current
evidence and the owner's continued-execution direction are recorded below.

The canonical decision artifact is
`docs/plans/2026-11-14-001-feat-dataset-traversal-continuation-plan.md`.
Its body was not changed during implementation. The annotated local tag
`pre-dataset-improvement` pins baseline
`9a681eabfe53b66f9d6d1459960ba10b95c1ff1d`.

## Continued optimization and current evidence

After the exhausted `e393202d` confirmation, the owner directed continued work
until interrupted. Subsequent comparison rounds retain the fixed profile and
1,200-second cumulative bound, including at most one confirmation each. Rounds
follow material corrections; historical results remain immutable, and unchanged
broad tests or comparisons are not repeated merely to seek favorable samples.

Implemented and focused-verified corrections:

- `9b26a325`: one real synthetic matrix publication per module, with immutable
  JSON seed and independent decoded trees for all corruption cases. Worker
  protocol-file runner observations fell from 34.23 to 4.96 seconds; coordinator
  protocol/runner verification passed 145 checks. No cases or assertions removed.
- `2c554147`: pure prepared exact-NumPy input predicates retain per-call metadata
  validation without reconstructing TensorSpec/MethodCallNode objects. Generic
  fallback, backend activation, raw-result validation and ambient delivery remain.
  Coordinator verified 129 Methods/Dataset checks.
- `3b636fa6`: bounded owned NumPy slices for pure Array-to-Batch inputs. Eleven
  rows/two leaves use six slice copies instead of 22 row copies; Map remains
  per-row. Coordinator verified 216 continuation/native-integration checks.
- `3d1df5e7`: routine representatives and executable collection proof; changed
  source metadata fallback matches generic values and failure frontiers.
- `238a2cbd`: remove immediately duplicated pre-reservation validation and retain
  eligible Torch raw-call selection per invocation. Fresh native-boundary checks,
  per-call signature delivery and mode/default fallback remain. Coordinator
  verified 96 transaction/Torch checks and 45 review-fix checks.
- `6b3701cc`: equivalent retained forward selection for TF explicit training,
  verified with 11 coordinator checks, including native SGD and collection.
- `ff9e72aa`: stable JAX argument commitment only with a committed single-device
  anchor, avoiding the second placement-only compilation. Uncommitted calls retain
  `jax.default_device` inference. Coordinator verified 32 checks before review
  and **14** final checks afterward, including two virtual CPU devices (the commit
  message's final count of 13 is a transcription error). All existing candidate
  validation, finite checks, synchronization and repair remain.
- `dd8e2fc6`: framework CI manifest includes both new TF parity cases; three
  coordinator checks passed without skips.

Three seven-lens local reviews completed under `/tmp/dryml/ce-code-review/`:
`hotpath-3b636fa6`, `training-hotpath-3d1df5e7`, and
`native-placement-6b3701cc`. Their independent validator batches confirmed and
subsequent commits corrected routine allowlist/tier omissions, CPU test controls,
TF CI-manifest omission, and all-uncommitted JAX default-device handling. No
cross-provider independence is claimed. A simplification pass made no changes.

One legacy Torch strict-observer-stop test expected save/retry after failure
without requesting a matching input record. A temporary committed-method control
reproduced the failure before the current optimization. The repaired test covers
both explicit checkpoint-before-error recovery and no-checkpoint fail-closed
behavior; model/optimizer, accepted progress and no-extra-update assertions remain.

### Complete routine pass at 238a2cbd

```sh
/usr/bin/time -o /tmp/dryml/dataset-routine-time-238a2cbd.txt -f 'exit=%x wall_seconds=%e' ./tests.sh good-enough --ignore tests/old --ignore tests/dev -x tests
```

CPU-only, single native-library thread controls, activated `big_env`:

| Phase | Passed | Skipped | Pytest elapsed |
|---|---:|---:|---:|
| Isolated phase | 265 | 0 | 46.04 s |
| Remaining routine selection | 4,884 | 25 | 845.99 s |
| Entire runner | **5,149** | **25** | **898.23 s full wall** |

Exit zero and strictly below 900 seconds: **R20 passed at this revision**.
Later TF/JAX changes require final-tree verification; this result is not erased
by older failing timings or silently extended to untested changes.

### Comparison round at 3d1df5e7 and confirmation at 238a2cbd

The complete fixed matrix at `3d1df5e7` finished all required timing cells without
failure in **914.6028693858534 seconds** (914.72 full wall). Its report is
`/tmp/dryml/u8-evidence/comparison-3d1df5e7.json`. Indexed steady new/old improved
to **0.824319**; Torch remained **1.152521/1.141811** at cadences 0/32, TF was
**0.972622/1.045081**, and Keras/JAX were noisy. All numerical work checks and
native reference modes remain represented; overall status was unverified.

The one confirmation after native-update corrections selected old/new training
workloads only, retaining the prior charge:

```sh
python -m tests.tools.benchmark_dataset_traversal --baseline-source /tmp/dryml/u8-baseline-f569da67 --baseline-revision 9a681eabfe53b66f9d6d1459960ba10b95c1ff1d --confirm /tmp/dryml/u8-evidence/comparison-3d1df5e7.json --output /tmp/dryml/u8-evidence/confirmation-238a2cbd.json --workload torch --workload jax --workload tf --workload keras --mode old --mode new
```

All selected timing cells completed; cumulative elapsed reached
**1200.1271082949825 seconds**, including final overhead, so the report remains
unverified. No missing cell was filled with stale evidence. Torch steady ratios
**0.900239/0.899465** resolved its measured regression. TF **1.050100/1.031355**
and JAX **1.052803/1.090686** still regressed. Keras cadence zero remained noisy
at **0.910618**; cadence 32 observed no regression at **0.867939**. This partial,
budget-exhausted confirmation does not establish whole-matrix completion.

### Comparison and confirmation at 3153003b

After the retained TF selection and JAX commitment corrections, the complete
fixed matrix consumed **953.0229556770064 seconds**. Its immutable report is
`/tmp/dryml/u8-evidence/comparison-3153003b.json`. Indexed implicit steady time
was **0.797490** times baseline; JAX cadences 0/32 were **0.828590/0.856285**, all
classified no-regression-observed. Torch cadence 32, both TF cadences and Keras
cadence zero remained noisy; the report was **unverified**.

The same-revision single confirmation selected old/new Torch, TF and Keras,
reusing only identity-matched instrumentation. It consumed a cumulative
**1074.811164646875 seconds**, `confirmation_count=1`; the report is
`/tmp/dryml/u8-evidence/confirmation-3153003b.json`.

| Steady work | New/pre-change | Decision |
|---|---:|---|
| Torch, cadence 0 / 32 | 0.922084 / 0.876633 | No regression observed |
| TF, cadence 0 | 0.881448 | No regression observed |
| TF, cadence 32 | **1.226797** | **Regression** |
| Keras, cadence 0 / 32 | 0.838015 / 0.823268 | No regression observed |

TF cadence 32 measured old/new medians **0.713653519/0.875508134 seconds**.
The **0.161854615-second** difference exceeds its three-max-MAD envelope of
**0.149415965 seconds**. This is a regression under the unchanged decision rule,
not a passing noise result. Keras cadence-32 full-invocation/end-to-end ratios
**1.121633/1.172677** remain separately visible informational regressions.
The partial confirmation has no automatic timing merge with the initial matrix
and remains unverified. Its unused **125.188835353125 seconds** do not permit a
second confirmation. Both reports are preserved.

### Explicit TensorFlow scalar-loss correction

The explicit training and validation loops inspect every actual loss result.
Exact native eager CPU float32/64 scalars use Identity instead of redundant Mean;
other types, ranks, dtypes, devices and symbolic/unknown-rank results retain
`tf.reduce_mean`. Training's default unit-seeded gradient is preserved. There is
no cached shape assumption, device readback, admission relaxation, or change to
compiled Keras training, native-owner reservation, acceptance or capture.

Proof-first native SGD parity passed before the new work assertion failed on
three scalar Mean operations over three updates. The isolated scalar/gradient
probe then reduced dispatched operations from **five to one**: Range, Mean,
Reshape, Tile and RealDiv become Identity. Signed zero, subnormals, finite extremes,
infinities and NaN retain bit-level values, unit-seeded gradients, dtype and device.
Per-call scalar/vector/scalar tests retain SGD and example-weighted accounting.
These are work/correctness proofs, not throughput evidence.

Coordinator verification passed **14 native correctness/recovery checks in
13.95 seconds**, then **5 checks in 6.98 seconds** after review fixes. Two
explicit GPU/cross-device cases skipped because no GPU was visible; that hardware
behavior remains unverified. Actual routine collection retained both SGD cases
and the scalar/fallback proof, excluding the two opt-in accelerator cases.
The six-lens local review and independent validator are retained at
`/tmp/dryml/ce-code-review/tf-scalar-loss-3153003b/`. Findings #1/#2 were corrected
by CPU-scoping native setup/execution and adding conditional GPU fallback tests.
Three simplification lenses requested no change. No cross-provider independence
is claimed. The subsequent fixed comparison and routine attempt are below.

### Scalar-loss comparison at 314c9057

With the same CPU/single-thread controls and `PYTHONDONTWRITEBYTECODE=1`:

```sh
/usr/bin/time -o /tmp/dryml/u8-evidence/comparison-314c9057.wall.txt -f 'exit=%x wall_seconds=%e' python -m tests.tools.benchmark_dataset_traversal --baseline-source /tmp/dryml/u8-baseline-f569da67 --baseline-revision 9a681eabfe53b66f9d6d1459960ba10b95c1ff1d --output /tmp/dryml/u8-evidence/comparison-314c9057.json --workload tf --workload keras --workload torch --workload jax --workload indexed-implicit --workload indexed-explicit --workload ordinary --workload direct --workload mixed --workload buffered
/usr/bin/time -o /tmp/dryml/u8-evidence/confirmation-314c9057.wall.txt -f 'exit=%x wall_seconds=%e' python -m tests.tools.benchmark_dataset_traversal --baseline-source /tmp/dryml/u8-baseline-f569da67 --baseline-revision 9a681eabfe53b66f9d6d1459960ba10b95c1ff1d --confirm /tmp/dryml/u8-evidence/comparison-314c9057.json --output /tmp/dryml/u8-evidence/confirmation-314c9057.json --workload torch --workload keras --workload jax --mode old --mode new
```

The initial complete matrix consumed **998.7260040468536 seconds** (998.84 full
wall). TF steady ratios **0.894977/0.918846** at cadences 0/32 both classify
no-regression-observed, resolving the prior TF slowdown in this round. Indexed
implicit was **0.489795**. Torch cadence zero, Keras cadence 32 and both JAX
cadences were noisy; no resolved old/new regression was observed.

The sole confirmation reached cumulative **1092.1976065519266 seconds**. It
reused identity-matched instrumentation, never timing windows:

| Work | Cadence 0 ratio / status | Cadence 32 ratio / status |
|---|---|---|
| Torch | 0.846131 / no regression observed | 0.862730 / noisy |
| Keras | 0.949693 / noisy | 0.912711 / no regression observed |
| JAX | 0.879115 / no regression observed | 0.843200 / no regression observed |

Although the initially noisy cells cleared, two other measured pairs became
noisy. Their favorable initial windows are not substituted to conceal that
uncertainty. The partial confirmation also omits 88 required matrix cells and
does not automatically combine timing evidence. **R28 remains unverified**.
The single confirmation allowance is spent despite approximately 107.8 unused
seconds; no unchanged repeat or second confirmation is implied.

### Routine failure and collection-inventory correction

```sh
/usr/bin/time -o /tmp/dryml/dataset-routine-time-314c9057.txt -f 'exit=%x wall_seconds=%e' ./tests.sh good-enough --ignore tests/old --ignore tests/dev -x tests
```

The isolated phase passed **265 tests in 47.69 seconds**. The ordinary phase
stopped with **4,835 passed, 25 skipped, one failed in 965.77 seconds**. Complete
runner wall time was **1019.93 seconds**, exit 1: both completion and strict
wall-time requirements failed. The stale exact inventory in
`test_good_enough_collects_native_regressions_without_eager_framework_imports`
expected 34 nodes instead of the correctly selected 35 after adding the scalar
loss proof. This is a test-inventory defect, not an excluded or disabled test.

Focused reproduction observed the same 35-versus-34 failure. The correction adds
the new node to the exact expected set and extends the executable `@manifest`
guard to collect it alongside both retained-call cases. Exact set/count/tier,
control filtering and zero-framework-import assertions remain. Coordinator
verification of the entire affected `tests/tools/test_runner.py` passed
**30 checks in 25.00 seconds**. This mechanical inventory synchronization does
not claim the full wall-time gate is repaired. The old untracked medium timing
file describes a different, smaller selection and cannot attribute this run's
elapsed time; it is preserved.

## Implemented boundaries

- Frozen, source-inert preparation retains operator topology, controls, specs,
  bindings, and implementation selections across independent sessions.
- DatasetTraversal owns continuous explicit sections and implicit finite passes,
  borrowed views, acceptance tokens, and boundary-before-reopen handoffs.
- DatasetState v1 owns detached snapshots and buffers, validates supplied targets,
  and publishes inventoried directories without clobbering cooperating writers.
- Source recovery is per occurrence: indexed, explicitly enabled replay, or direct
  RecoverableGenerator state. Retained notebook class source reconstructs in a
  clean process without requiring an importable user module.
- Accumulate owns post-transition carry; terminal Fold retains root-EOF authority.
- TrainFunction owns invocation input control. TrainState and Experiment v3
  encode one continuation, with input-unstarted, stale exact retry, provider
  retirement, reservation, and invalidation rules.
- Torch, explicit TensorFlow, and JAX loops use accepted input sessions. The Keras
  adapter retains a bounded unacknowledged window, validates consumed tokens
  before stateful forward effects, preserves compiled overlap, and owns source
  capture/cleanup on its producer thread.
- Keras callbacks retain one invocation lifecycle across fit segments. Known
  finite boundaries normalize before final-update callbacks; exact finite
  shortfalls fail instead of becoming successful epochs.

## Authoritative integrated verification

All runs used the existing `big_env` after `source ~/conda.sh`, with
`CUDA_VISIBLE_DEVICES=-1 JAX_PLATFORMS=cpu TMPDIR=/tmp/dryml`. Resource-intensive
runs were serial. Those historical runs installed no dependencies or environment
repairs; subsequent owner-authorized Optax setup is recorded separately below.

### Bounded affected regression

```sh
./tests.sh tests/data/test_dataset_cursor.py tests/data/test_generator_dataset.py tests/data/test_stream_graph.py tests/data/test_native_stream_operators.py tests/data/test_example_counts.py tests/data/test_native_training_inputs.py tests/methods/test_accumulator.py tests/artifacts/test_fold.py tests/artifacts/test_fold_resume.py tests/artifacts/test_fold_execution.py tests/models/test_dataset_training_integration.py --no-cov
```

Result: **272 passed, 9 skipped**, pytest elapsed 282.82 seconds. Skipped native
prerequisites remain unverified; they are not evidence of backend qualification.

### Complete routine gate

```sh
/usr/bin/time -o /tmp/dryml/dataset-routine-time-7b1698bd.txt -f 'exit=%x wall_seconds=%e' ./tests.sh good-enough --ignore tests/old --ignore tests/dev -x tests
```

At implementation revision `7b1698bd`, both interpreter phases completed with
exit zero:

| Phase | Passed | Skipped | Pytest elapsed |
|---|---:|---:|---:|
| Isolated session/runtime/dispatch phase | 265 | 0 | 46.75 s |
| Remaining representative maintained selection | 4,713 | 30 | 861.38 s |
| Complete runner | **4,978** | **30** | **914.70 s full wall time** |

**R20 fails.** The required limit is strictly less than 900 seconds, including
runner overhead, collection, setup, and both interpreter phases. An exit-zero run
at 914.70 seconds is not a passing timing gate. No assertions or representative
coverage were weakened, and the same expensive suite was not repeated merely to
seek a faster sample.

### Post-kernel complete routine gate

After material kernel, driver and evidence corrections at `e393202d`, the same
complete routine command ran with CPU/single-thread controls:

```sh
/usr/bin/time -o /tmp/dryml/dataset-routine-time-e393202d.txt -f 'exit=%x wall_seconds=%e' ./tests.sh good-enough --ignore tests/old --ignore tests/dev -x tests
```

| Phase | Passed | Skipped | Pytest elapsed |
|---|---:|---:|---:|
| Isolated session/runtime/dispatch phase | 265 | 0 | 47.68 s |
| Remaining representative maintained selection | 4,837 | 25 | 875.67 s |
| Complete runner | **5,102** | **25** | **929.86 s full wall time** |

Exit zero establishes correctness for the selected tests, but **R20 still
fails**. The 47 new linear-kernel cases and 115 backend-free benchmark checks
remain in the routine selection. Nine expanded real builder probes are opt-in;
named native training representatives remain selected. No assertions were
weakened. The runner was not repeated after this failed timing result.

A bounded diagnosis then ran only the two changed test files, retaining all
their selected cases:

```sh
./tests.sh tests/tools/test_benchmark_dataset_traversal.py tests/data/test_linear_batch_kernel.py --no-cov -m 'not exhaustive_only' --durations=15
```

Result: **162 passed, 9 deselected in 34.59 seconds**. Many protocol corruption
cases each rebuild and atomically publish a complete synthetic matrix, costing
about 1.2 seconds per case. Shared immutable setup with per-case deep copies is
a possible test-overhead correction, not an implemented improvement or a waiver
of the timing gate. Cold native imports also appear in the focused timing; they
must not be double-counted as independent routine costs.

### Additional focused evidence

- Four selected offline package checks passed, including wheel/sdist module
  inventory and an installed-package continuation suffix example.
- Fifty-two benchmark-tool/profile checks passed. These include synthetic output
  contention, interrupted atomic publication, required-evidence applicability,
  child reaping, and cumulative native Keras checkpoint-step tests; they do not
  execute the comparative workload.
- Twenty-two codec tests passed, including host-only decode and real CPU
  TensorFlow placement reconstruction/fallback rejection.
- Focused native tests qualified eager/compiled Keras queued-window restoration,
  token-before-forward ordering, callback lifecycle, overlap, and timeout cleanup
  on TensorFlow 2.19.0 / Keras 3.13.2 CPU.
- Fourteen deterministic installed-JAX transaction and repair checks passed.
  They do not replace real Optax training qualification.

### Subsequent JAX driver unit qualification

Owner-authorized setup added Optax 0.2.4 and Chex 0.1.89, retaining JAX/JAXlib
0.5.3 and NumPy 2.1.3. The coordinator reported **55 real JAX/native focused
checks passed, one optional NNX skip** before this driver work.

The comparative JAX driver now implements old/new functional DRYML training,
matched-native slice batching, and optimized-native JIT section scans. All use
the fixed dense/SGD/MSE recipe. Native modes use no DRYML wrappers. All required
JAX applicability cells remain required; missing JAX/Optax is explicitly
unavailable, not a qualification exemption.

Proof-first routing regression failed on the original unconditional
`unsupported` response. After implementation:

```sh
./tests.sh tests/tools/test_benchmark_dataset_traversal.py --no-cov
```

Result: **72 passed** (pytest elapsed 10.13 seconds). This comprises synthetic
protocol checks plus tiny real CPU builder probes (four rows/four SGD updates
per mode) and construction-only default-recipe/script-import checks. The old probe uses an
isolated archive of the actual pinned baseline; new's tiny checkpoint restores
and validates the remaining input suffix. Numerical NumPy protocol fixtures
check fixed 128-update checkpoint/count/weight associations at cadences 0/32;
those fixtures are not fake Optax qualification. No real comparison cell,
warmup, timed repetition, or throughput measurement ran during this driver work.

Those historical driver timing records included JIT compilation and normal final
input capture in the field then named `steady_seconds`. They were **not** isolated
steady measurements and cannot establish R28. They remain historical evidence;
the cost-domain protocol below supersedes their eligibility rules. See the
[Data measurement section](data.md#bounded-cpu-evidence-and-unresolved-gates)
for the retained historical failure and remaining comparison gate.

### U8 cost-domain fidelity correction (no fresh comparison)

At starting HEAD `fcb49ba4`, two proof-first synthetic regressions demonstrated
that the tool accepted invocation-only evidence as steady and returned a ratio
for equal performed work despite incompatible timed domains/counts. Both failed
before production changes. Verification uses `big_env`, CPU-only and one native
thread throughout, with `TMPDIR=/tmp/dryml`; no installation or broad test run is
part of this correction.

```sh
source ~/conda.sh
conda activate big_env
export CUDA_VISIBLE_DEVICES=-1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 TMPDIR=/tmp/dryml PYTHONDONTWRITEBYTECODE=1
```

The tool now emits `dryml-dataset-traversal-benchmark-v2` with measurement
protocol `dryml-dataset-traversal-cost-domains-v2`. The fixed public work profile
and applicability requirements are unchanged. Each metric carries a typed name,
unit, exact timed count and section count, checked in every window before ratios.
Training steady time sums accepted report intervals 1→32, 33→64, 65→96 and
97→128: **124 timed updates**, while all **128 updates/four sections** still run.
Throughput uses 124, never an inflated 128 numerator. Between-update source,
acceptance, synchronization, queue and reporting overhead remains inside those
intervals; no per-yield source/state/callable instrumentation or new device fence
was added. Initial setup/first report, section gaps and final postlude/capture
tail remain visible and reconcile with complete invocation time.
The report's `decision_domain="old-new-steady"` explicitly limits the meaning
of its overall steady regression verdict; full-cost signals keep their own status.

Full invocation and complete end-to-end 128-update domains have separate ratios.
Invocation includes normal final exact capture and cadence IO; end-to-end also
includes preparation/imports, reference math, validation, final saving and cleanup.
Checkpoint observations list 0 or 4 intermediate saves and exact accepted steps;
final save is separate. Independently unobservable import/open/compile/final-capture
durations remain unknown, with measured mixed boundary costs and complete totals
retained. A larger informational invocation ratio cannot waive a same-domain
steady regression, and an informational total-cost regression is not relabeled
as a steady regression merely because it includes cold or recovery work.

Optimized-native JAX scans and Keras 32-step execution remain grouped. Their four
section-end reports cannot isolate first-update-to-last-update intervals without
altering the optimized path. Their comparable steady domain/time/throughput and
first-update latency are explicitly unavailable, with a `steady_blocker`; full
invocation/end-to-end evidence remains required. Optimized NumPy likewise retains
its one whole-source transform and reports full vectorized delivery costs, not
fake post-first steady work. These unavailable native steady ratios remain named
follow-ups, not numerical parity or a waiver of old/new R28. Dataset post-first
counts are 1,016 implicit batches, 960 explicit batches, 480 sectioned source
outputs, and 511 buffered outputs; all full useful work and capture probes remain.
See [exact measurement semantics](data.md#measurement-domains-and-exact-timed-populations).

Proof-first command (before production edits):

```sh
./tests.sh tests/tools/test_benchmark_dataset_traversal.py -k 'invocation_only_training or phantom_128' --no-cov -o cache_dir=/tmp/dryml/u8-cost-domain-cache
```

Result: **2 failed as expected**: invocation-only evidence was accepted and the
incompatible-domain comparison returned ratio 1.0. The focused file then verifies
the corrected gate, exact numerators, boundary reconciliation, synthetic checkpoint
and Dataset scheduling, grouped-native refusal, preservation of informational full
cost signals, and rejection of historical confirmation. Tiny real numerical
checks remain four rows/four SGD updates per mode, including the actual archived
baseline; they do not invoke fixed workload windows.

```sh
./tests.sh tests/tools/test_benchmark_dataset_traversal.py --no-cov -o cache_dir=/tmp/dryml/u8-cost-domain-cache
```

Result: **96 passed**, pytest elapsed **30.38 seconds**. Final report-decision
labels and native-limitation reporting were then checked with:

```sh
./tests.sh tests/tools/test_benchmark_dataset_traversal.py -k 'complete_applicable_evidence or total_invocation_costs' --no-cov -o cache_dir=/tmp/dryml/u8-cost-domain-cache
```

Result: **2 passed, 94 deselected**, pytest elapsed **2.47 seconds**. All commands
were serial focused invocations; broad/full/heavy/exhaustive/profile/coverage
verification was not run. Local diff review and `git diff --check` covered the
four changed files. No plan, runtime source, Git index, branch or history changed.

The authorized fresh 1,200-second budget has **not started**. No actual comparison
cell, warmup, repetition, or timing run was executed. Historical JSON was not
modified; v1 reports cannot be confirmed or reused as v2 instrumentation evidence.
Correct measurement eligibility is not evidence that current R28 performance or
the historical regression signals are resolved.

### Delta-review #2: inclusive Dataset first-result latency

The validated delta-review #2 correction starts each new-mode Dataset first-result
clock before advancing the epoch enumerator, including acquisition/transformation
that stages the first value before returning its view. Initial enumerator setup
belongs to the first observation; preparation/open remain separately measured.
The cost-list and top-level first-result observations now share the same inclusive
boundary. Empty-view/terminal attempts without a delivery are separately measured
in `costs.epoch_enumeration_without_result_seconds`. Tail EOF/drain and final
capture retain their own boundaries, and the complete-window clock is unchanged.

Proof-first, with production code unchanged, the deterministic staging fixtures
returned **2 failed, 96 deselected** (pytest 0.57 seconds): the ordinary terminal
case reported 2 instead of 32 synthetic seconds, omitting 17 seconds of enumerator
setup and 13 of staging; the empty-view case raised `StopIteration` without
reporting its no-result cost.

```sh
./tests.sh tests/tools/test_benchmark_dataset_traversal.py -k 'first_result_includes_staging' --no-cov -o cache_dir=/tmp/dryml/delta-review-2-cache --basetemp=/tmp/dryml/delta-review-2-tests
```

After correction, both fixtures reconcile every synthetic second across preparation,
open, inclusive first results, post-first delivery/EOF, no-result enumeration,
final capture and cleanup. The related review testing gap #4 is covered by four
actual native NumPy producer-loop cases: indexed implicit/explicit crossed with
matched/optimized native, each executed once under a synthetic clock. Numerical
validation retains all 1,024 batches. Matched intervals retain 1,016/960 timed
batches and 8/64 first-result observations; optimized delivery retains the full
vectorized domain, null steady time/domain and an explicit blocker. Copies of
these actual producer results qualify through aggregation/protocol validation
without executing warmups or repetitions. Synthetic metadata comparison refuses
optimized steady ratios while permitting the complete-cost domain.

```sh
./tests.sh tests/tools/test_benchmark_dataset_traversal.py -k 'first_result_includes_staging or native_dataset_producer_cost_domains or dataset_window_cost_domains' --no-cov -o cache_dir=/tmp/dryml/delta-review-2-cache --basetemp=/tmp/dryml/delta-review-2-tests
./tests.sh tests/tools/test_benchmark_dataset_traversal.py --no-cov -o cache_dir=/tmp/dryml/delta-review-2-cache --basetemp=/tmp/dryml/delta-review-2-tests
```

Results: **13 passed, 89 deselected** (0.85 seconds), then **102 passed** (31.07
seconds: the existing 96 focused checks plus six synthetic-clock cases). Both
commands ran serially in source-activated `big_env` with the CPU/single-thread
controls above. The fixed profile and all four Dataset tail populations
(1,016/960/480/511), training's 124-update steady population, full useful-work
counts, capture probes and complete-cost visibility are preserved. No plan or
runtime source was edited, and no Git staging, commit, branch change or push was
performed. No broad/full/heavy/exhaustive/profile/coverage run or actual comparison
was started; the fresh 1,200-second budget remains unstarted. Review #1's pending
comparative/routine gate remains unchanged.

### Fresh-comparison TF/Keras recipe loss correction

The first fresh report, `/tmp/dryml/u8-evidence/comparison-fresh-fde25998.json`,
records HEAD `fde25998`, **894.4321190109476 seconds** charged, status
**unverified**, and `confirmation_count=0`. Old Keras failed at both cadences
because the recipe supplied an optimizer but no explicit compile loss. The pinned
baseline validates a fallback MSE without installing it in compile kwargs; this
is a benchmark recipe defect, not an unsupported baseline feature.

The shared untimed `_tf_training_recipe` now supplies
`Loss(tf.keras.losses.MeanSquaredError)` through the existing public API for both
old/new explicit TF and Keras. The canonical `_tf_layer_defs`, zero bias-free
16-to-1 dense kernel, SGD(.001), seed 1729, batch 32, 128 updates/four sections,
cadences 0/32, native baselines, loss reporting, and Torch/JAX recipes are retained.
Archived source is unchanged.

Proof-first, `test_tf_recipe_tiny_real_unit_qualification[old]` failed with
`KeyError: 'loss'` before fitting. The separate negative probe
`test_old_keras_original_recipe_missing_compile_loss` reproduces native
`ValueError: No loss to compute` with zero optimizer updates. After correction,
old/current subprocesses check compile kwargs before fitting and match reference
weights, per-update mean losses, four accepted callbacks, and two completed
sections using only four rows/four updates per trainer. The old subprocess uses
the existing `/tmp/dryml/u8-baseline-f569da67` archive, selected with
`DRYML_BENCHMARK_BASELINE_SOURCE`; DRYML is imported before TensorFlow.

In source-activated `big_env` with the CPU/single-thread controls above:

```sh
export DRYML_BENCHMARK_BASELINE_SOURCE=/tmp/dryml/u8-baseline-f569da67
./tests.sh tests/tools/test_benchmark_dataset_traversal.py::test_tf_recipe_tiny_real_unit_qualification tests/tools/test_benchmark_dataset_traversal.py::test_old_keras_original_recipe_missing_compile_loss tests/tools/test_benchmark_dataset_traversal.py::test_tf_layer_declarations_are_factories_for_the_sequential_owner tests/tools/test_benchmark_dataset_traversal.py::test_fixed_profile_cannot_silently_reduce_work_or_repeats tests/tools/test_benchmark_dataset_traversal.py::test_cli_profile_is_backend_free_and_rejects_profile_mutation 'tests/tools/test_benchmark_dataset_traversal.py::test_jax_driver_tiny_real_unit_qualification[new]' --no-cov -x -q
```

Result: **7 passed**, pytest elapsed **20.97 seconds**. This is functional unit
qualification, not measurement. No fixed comparison window, warmup, repetition,
confirmation, broad suite, or coverage ran. Comparative budget is **uncharged by
this correction**: **305.5678809890524 seconds remain**, with one confirmation
available. The first fresh report and its unresolved regression signals are
preserved. Subsequent coordinator verification is recorded below.

### Confirmation instrumentation identity

`--confirm` reuses only valid, uniquely keyed instrumentation whose own
`instrumentation_identity` exactly matches the selected runtime and current
benchmark driver. New reports record `instrumentation_identities` by mode and each
instrumentation cell records its identity, `reused_instrumentation`, and
`instrumentation_reuse_reason`. The driver SHA-256 covers the complete current
script, including shared recipes even when the worker imports archived source.
Old-mode runtime identity fingerprints the supplied archive's `src/dryml` file
paths and contents (excluding generated bytecode). Current/new and native modes
conservatively require the same clean source revision; staged, unstaged or
untracked source changes make that identity unknown. The unsupported
`src/dryml/graph` directory is outside the source contract. Inputs must remain
stable during the run; this identity does not lock external dependency versions.

Historical cells without identity remain readable, but cannot supply reusable
instrumentation, even if their report has a matching revision or report-level
identity. Unknown or changed identity forces fresh instrumentation within the
existing cell/overall deadlines, or leaves that required cell unverified if the
budget is unavailable. No identity is retrofitted onto historical cells. Timing
cells always run anew; there is no throughput merge or timing reuse. The fixed
profile, required cells, prior **894.4321190109476-second** charge and single
confirmation limit retain their existing meaning. This correction executes no
comparison and leaves **305.5678809890524 seconds and one confirmation** available.

### Linear Batch kernel and authoritative delta verification

Finite, unsectioned Array and Map(Array) inputs bind a specialized Batch loop at
execution construction. It retains each row before the next pull, invokes Map
and its validators per authored source item, and invokes the selected collator
once per delivered batch. Child source/Map facts publish once per Batch pull,
including failed pulls before parent cleanup; public capture and reentrant
advance remain forbidden while busy. Other topologies retain generic execution.

For eleven rows in batches 4/4/3, deterministic proofs eliminate eleven child
pull/step/delivery calls per source or Map occurrence and reduce child metadata
assignments from 88 to 18 per occurrence. Source reads and buffer retentions remain
eleven; Map calls remain eleven when present; collations remain three. These work
counts do not establish measured throughput improvement. The 47 kernel cases
compare generic values/frontiers, bidirectional restoration, independent
occurrences, failure frontiers, close-once cleanup, stale-record refusal,
readonly foreign backing, per-row validation and native non-scalarization.

Coordinator verification, serially in `big_env` with CPU/single-thread controls:

```sh
./tests.sh tests/data/test_linear_batch_kernel.py tests/data/test_traversal_preparation.py tests/data/test_dataset_traversal.py tests/data/test_dataset_state.py tests/data/test_native_training_inputs.py tests/tools/test_benchmark_dataset_traversal.py --no-cov
./tests.sh tests/tools/test_benchmark_dataset_traversal.py --no-cov -m 'not exhaustive_only'
```

Results: **269 passed in 79.09 seconds**, including the three opt-in tiny TF
recipe probes with `DRYML_BENCHMARK_BASELINE_SOURCE` set; then **115 passed,
9 deselected in 26.92 seconds** after the instrumentation identity correction.
The worker observed eight expected red stale-identity failures before that
correction. Synthetic confirmation checks preserve the prior charge, reject
historical/changed/unknown identities, remeasure timing even on an identity match,
and leave missing instrumentation unverified when the residual budget is spent.

Delta review artifacts are under
`/tmp/dryml/ce-code-review/20261119-122051-fde25998-delta/`. Specialist and validator
routes were unavailable; the disclosed static local fallback identified the
stale-instrumentation P1, now corrected and covered by focused tests. No
independent reviewer or cross-model validation is claimed for this delta.

### Single post-kernel confirmation: budget exhausted, regressions remain

At clean runtime revision `e393202d`, the one authorized confirmation used the
unchanged profile, all required modes/cells, and a workload order prioritizing
indexed, Torch, JAX and then corrected Keras evidence. The prior report's
894.4321190109476-second charge remained binding:

```sh
/usr/bin/time -o /tmp/dryml/u8-evidence/confirmation-e393202d.wall.txt -f 'exit=%x wall_seconds=%e' python -m tests.tools.benchmark_dataset_traversal --baseline-source /tmp/dryml/u8-baseline-f569da67 --baseline-revision 9a681eabfe53b66f9d6d1459960ba10b95c1ff1d --confirm /tmp/dryml/u8-evidence/comparison-fresh-fde25998.json --output /tmp/dryml/u8-evidence/confirmation-e393202d.json --workload indexed-implicit --workload torch --workload jax --workload keras --workload tf --workload indexed-explicit --workload ordinary --workload direct --workload mixed --workload buffered
```

Exit **1**, confirmation wall **306.87 seconds**, cumulative tool elapsed
**1201.2203454473056 seconds**, `confirmation_count=1`, status **unverified**.
Termination and aggregation overhead are charged, not hidden. Historical
instrumentation lacks a cell identity and was not reused. No throughput from the
first report was merged into the confirmation to replace unfinished cells.

Completed comparable steady windows (seconds, median ± MAD):

| Work | Pre-change | New | New/pre-change | Three-max-MAD envelope |
|---|---:|---:|---:|---:|
| Indexed implicit, 1,016 post-first batches | 3.714223 ± 0.017297 | 3.974633 ± 0.035538 | 1.070112 | 0.106613 |
| Torch, 124 timed updates, cadence 0 | 0.126583 ± 0.001491 | 0.161739 ± 0.000332 | 1.277731 | 0.004472 |
| Torch, 124 timed updates, cadence 32 | 0.129622 ± 0.000182 | 0.162725 ± 0.001355 | 1.255380 | 0.004066 |
| JAX, 124 timed updates, cadence 0 | 0.149058 ± 0.002093 | 0.178595 ± 0.002101 | 1.198159 | 0.006304 |

Each new-minus-old delta exceeds its noise envelope. **R28 remains failing**;
the work-count reduction did not resolve these measured regressions. Complete
training still performs 128 updates/four sections; the steady numerator is 124.
JAX cadence-zero full invocation and end-to-end totals improved relative to old,
but those informational domains do not waive its isolated steady regression.

The deadline interrupted old JAX cadence 32; its remaining cells and subsequent
Keras, TF, explicit-indexed and source/recovery cells are budget-exhausted. The
corrected Keras loss recipe therefore still lacks post-correction fixed-profile
evidence, despite its real tiny functional checks. Native gaps and unavailable
grouped steady domains remain explicitly labeled in JSON. Neither a second
confirmation nor another independent comparison budget is authorized.

## Review corrections

Ten local review lenses completed; one independent validation batch confirmed
fourteen actionable findings. Thirteen implementation/test findings were fixed
in subsequent commits:

- Exact finite shortfalls and final generator/composite boundary propagation.
- Private DatasetState representation and valid public target annotations.
- Python-compatible secondary error notes and primary-error preservation.
- Readonly NumPy views over mutable foreign backing in stream/native buffers.
- JAX candidate reservation before native-state reads, recovery callback capture,
  and stronger multi-owner repair checks.
- Invocation-wide Keras callback lifecycle and History.
- TensorFlow required-placement verification after reconstruction.
- Exclusive atomic benchmark evidence publication, applicability-aware verdicts,
  and cumulative native checkpoint numbering.

The remaining P1 finding is incomplete qualification, not an accepted waiver.
Review and validation artifacts are under
`/tmp/dryml/ce-code-review/20261119-001028-b9f6bcd4/`; the original review report
describes the pre-fix findings, and this document records their application.
Cross-model review could not attest the host; local review does not claim
different-model corroboration.

## Unresolved qualification

1. **Final-tree routine verification:** `238a2cbd` passed in **898.23 seconds**;
   the subsequent `314c9057` run failed at **1019.93 seconds** with a stale test
   inventory. The inventory correction passes its affected subsystem; a complete
   successful run strictly below 900 seconds is still required. Historical
   929.86 and 914.70-second failures remain recorded.
2. **Comparative performance:** the historical fixed 1,200-second comparison
   budget was exhausted. Earlier indexed and Torch regression signals are preserved in
   [the Data measurement section](data.md#bounded-cpu-evidence-and-unresolved-gates).
   A later retained-owner fix eliminated repeated graph preparation in a
   deterministic proof. The owner-authorized fresh report at `fde25998` consumed
   894.4321190109476 seconds and observed indexed, Torch, explicit-TF and JAX
   steady regressions beyond measured noise. Old Keras failed because of the
   recipe loss omission. The subsequent linear kernel and corrected TF/Keras
   recipe were then subject to the one authorized confirmation at `e393202d`.
   It exhausted the cumulative budget at 1201.2203454473056 seconds, retained
   indexed/Torch/JAX steady regressions, and left later required cells incomplete.
    Both reports remain unverified. The subsequent owner-directed round at
   `3d1df5e7` and its confirmation at `238a2cbd` are recorded above; their total
    budget is also spent. The subsequent `3153003b` matrix and single confirmation
    clear several steady comparisons but retain the explicit-TF cadence-32
    regression recorded above. The `314c9057` scalar-loss round resolves that
    regression, but its single confirmation still has Torch/Keras noise and
    partial matrix coverage. Both reports remain unverified.
3. **Comparative JAX evidence:** real Optax is now available and the four-mode
   driver has tiny numerical unit qualification and real fixed-profile evidence
    at `fde25998` and later revisions. The `3153003b` complete matrix measures
    no old/new JAX steady regression at either cadence after the placement-only
    recompilation correction. This does not waive the remaining whole-matrix gate
    or qualify unmeasured accelerator/distributed behavior.
4. **Remaining evidence limits:** comparative copy/compile/pending-window metrics
   are incomplete. Python 3.10 failure-note behavior was emulated on Python 3.12
   with `add_note` unavailable; no real Python 3.10 run occurred. Accelerator
   throughput and distributed TensorFlow placement were not qualified.

Repository-wide Flake8 is not clean; this work does not claim a clean lint gate.
No configured typecheck was found. Exhaustive, broad heavy/profile, and coverage
suites were not run.

The implementation and focused fixes are committed locally. The parent workspace
gitlink and remote shipping are separate from this framework implementation.
