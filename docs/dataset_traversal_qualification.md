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

JAX timing records explicitly include JIT compilation and normal final input
capture in invocation time, distinguish setup from invocation and final save,
and leave unisolated cold-import/compile/capture-only costs unknown. See the
[Data measurement section](data.md#bounded-cpu-evidence-and-unresolved-gates)
for the retained historical failure and remaining comparison gate.

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
