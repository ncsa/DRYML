# Dataset traversal implementation qualification

## Status

The traversal, continuation, Accumulate, accepted-training-state, and native
consumer implementation is present on `fix/real-usage-followups`. The plan is
**not complete**: required comparative and routine performance qualification
remain blocking. Passing correctness checks do not waive those requirements.

The canonical decision artifact is
`docs/plans/2026-11-14-001-feat-dataset-traversal-continuation-plan.md`.
Its body was not changed during implementation. The annotated local tag
`pre-dataset-improvement` pins baseline
`9a681eabfe53b66f9d6d1459960ba10b95c1ff1d`.

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

1. **Routine performance:** reduce complete runner wall time below 900 seconds
   without weakening representative tests. The current complete result is
   914.70 seconds.
2. **Comparative performance:** the fixed 1,200-second comparison budget was
   exhausted. Earlier indexed and Torch regression signals are preserved in
   [the Data measurement section](data.md#bounded-cpu-evidence-and-unresolved-gates).
   A later retained-owner fix eliminated repeated graph preparation in a
   deterministic proof, but no post-fix comparison establishes its throughput
   effect. Corrected TF/Keras benchmark recipes lack fresh comparative evidence.
    The owner has authorized one fresh fixed 1,200-second comparison after runtime
    optimization. This driver work has not started it; the exhausted historical
    result remains unverified and must not be relabeled.
3. **Comparative JAX evidence:** real Optax is now available and the four-mode
    driver has tiny numerical unit qualification, but its fixed-profile cells
    still have no real evidence. They block the report rather than being exempted;
    coordinator-owned comparison and final verification remain outstanding.
4. **Remaining evidence limits:** comparative copy/compile/pending-window metrics
   are incomplete. Python 3.10 failure-note behavior was emulated on Python 3.12
   with `add_note` unavailable; no real Python 3.10 run occurred. Accelerator
   throughput and distributed TensorFlow placement were not qualified.

Repository-wide Flake8 is not clean; this work does not claim a clean lint gate.
No configured typecheck was found. Exhaustive, broad heavy/profile, and coverage
suites were not run.

The implementation and focused fixes are committed locally. The parent workspace
gitlink and remote shipping are separate from this framework implementation.
