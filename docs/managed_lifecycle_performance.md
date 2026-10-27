# Managed Lifecycle Performance

## Fresh-cache benchmark

Run the opt-in developer benchmark in `big_env`:

```bash
source ~/conda.sh
conda activate big_env
TMPDIR=/tmp/dryml CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu \
  python -m tests.tools.benchmark_cached_dataset --repeats 3
TMPDIR=/tmp/dryml CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu \
  python -m tests.tools.benchmark_cached_dataset --depth 5 --repeats 3
```

Each sample computes one one-element NumPy array into a fresh Store with default
SQLite indexing and real environment observation. Depth 5 wraps the same source
in five `Take(..., 1)` operators: data work stays constant while recipe metadata
grows. Checkpointing, final state publication, and managed completion are timed;
build is separate. The saved result is freshly restored and its content verified
after timing. Temporary Stores are removed on exit. `--elements` supports a
larger finite input, without touching user datasets or Stores.

The same script can run against an archived pre-Node source using its `src/` and
repository root on `PYTHONPATH`. Its old-build/return-value adapters exist only for
comparative benchmarking, not as product compatibility APIs. Use fresh Python
processes and the same environment for each implementation; compare multiple
samples and CPU time as well as wall time. These measurements are opt-in, not
flaky automated time cutoffs.

## 2026-10-26 comparison on schumann

Medians of three fresh-Store samples per row, using the same script and `big_env`:

| Implementation | Depth 0 wall / CPU | Depth 5 wall / CPU |
| --- | --- | --- |
| Before the Node lifecycle refactor (`f996ea43`) | 6.58s / 6.32s | 7.97s / 7.59s |
| Before this performance pass (`11354d8a`) | 24.14s / 22.89s | 48.93s / 46.68s |
| Performance branch | 5.09s / 4.70s | 7.99s / 7.41s |

This recovers approximately **4.7× / 6.1×** wall-time speedups over the regressed
implementation. The simple case is about **23% faster** than the pre-refactor
baseline; the deeper recipe reaches roughly wall-time parity, with slightly lower
CPU time. The deeper wall-time difference is below the variation among samples,
so it is not evidence of an additional speedup over the old implementation.
Earlier measurements under lower host load had smaller absolute times; the
comparison above reran all implementations together. It does not measure the
user's full dataset, GPU work, or Dispatch startup/transport.

### Notebook follow-up

The user also reported the following timings for the actual dispatched cache
workload, before and after this performance pass:

| Stage | Before | After |
| --- | --- | --- |
| Concretize | 0.046s | 0.042s |
| Build | 0.023s | 0.047s |
| Compute | 124.857s | 83.695s |
| Outside worker | 5.745s | 3.600s |
| Total cell | 130.671s | 87.383s |

This is approximately **33% less compute and total wall time** (about 1.5×
end-to-end speedup). These are user-reported single runs of the real workload,
not controlled benchmark medians; they confirm an application-level improvement
without attributing all remaining time or run-to-run variation to the framework.

## Work removed and correctness boundaries

- Observe the environment only once per executing invocation that actually
  captures state. Replay and capture-free explicit-final selection do not scan it.
- Reuse bounded interpretations of identical immutable control/Node/metadata bytes
  within the same process/thread/task admission. Every file read, pending check,
  owner/generation fence, selected-state reader and publication acknowledgement
  remains fresh. Scope exit, including cancellation, clears all entries.
- Retain complete canonical Node comparison evidence within the preparation's
  existing LRU budget. Equality still compares exact evidence under hash ties.
- Validate and detach encoder-ready JSON in one traversal rather than freezing
  and thawing the same tree. Byte output and closed grammar bounds are retained.
- Infer standard distribution locations without statting all subsequent installed
  files. All RECORD row/hash/size syntax is still checked, and later observations
  read the inventory afresh.

Routine regression tests guard work counts and failure behavior, not elapsed
seconds: stable safe points still read authority without redecoding graphs;
changed bytes and wrong operation IDs fail; cancellation clears facts; copied
caller contexts cannot borrow them; eviction affects work only; replay skips
environment observation; and environment failures remain unavailable evidence.

## Concurrent-history follow-up: performance recheck

On 2026-10-26, the query/history concurrency follow-up was compared with its
immediate baseline, `bb890dd3`, using serial fresh Python processes in `big_env`.
The baseline was exported with `git archive`; the active checkout and user Stores
were not changed for the comparison. Baseline/current order alternated to reduce
host-order bias. Current results describe the follow-up worktree, not `bb890dd3`.

The existing cache benchmark used three fresh-Store samples per implementation
and depth, including real environment observation and verified result restoration:

| Cache recipe | `bb890dd3` median wall / CPU | Follow-up median wall / CPU |
| --- | --- | --- |
| One-item source | 2.690s / 2.610s | 2.552s / 2.528s |
| Same source plus five Take wrappers | 4.087s / 4.052s | 3.991s / 3.960s |

There was no measured cache-compute regression. These absolute times reflect a
different host-load window from the earlier comparison; use the paired results,
not an across-window speedup claim.

History discovery was also timed against identical ready Store bytes: eight
Experiments had an initial history plus two published row revisions each, and a
ninth Experiment had no history. Three samples per history case were taken in
each fresh process, without writers or retries. Ordinary query figures below are
medians of ten samples from the same paired run:

| Query | `bb890dd3` median | Follow-up median | Candidate rows / decoded graphs, before → after |
| --- | --- | --- | --- |
| Matching history | 3.344s | 0.277s | 51 / 43 → 3 / 3 |
| Missing history | 3.421s | 0.0081s | 51 / 43 → 0 / 0 |
| Ordinary concrete model hit | 0.0797s | 0.0715s | 1 / 1 → 1 / 1 |
| Ordinary concrete model miss | 0.00584s | 0.00559s | 0 / 0 → 0 / 0 |
| Exact Experiment hit | 0.0887s | 0.0832s | 1 / 1 → 1 / 1 |

The history improvement comes from excluding unrelated classes/quoted subjects
before decoding, not from cached Store authority. Otherwise supported selective
posting plans retain their existing SQL and avoid these additional facet filters.
Unquoted recipes do not encode constructor values to look for quoted pins; a
regression test guards that work boundary, including array-valued configuration.
Quoted-history planning adds roughly 0.23ms in this fixture while removing seconds
of unrelated graph work. These filters still require scan admission when no
supported selective posting anchor exists; no general constant-time SQL claim is
made for arbitrary Store size.

For repeatable fresh-Store history and ordinary-query measurements:

```bash
TMPDIR=/tmp/dryml CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu \
  python -m tests.tools.benchmark_experiment_history \
  --histories 8 --checkpoints 2 --repeats 3
```

The benchmark uses lightweight CounterModel/Experiment fixtures, synthetic
environment evidence during untimed preparation, real SQLite reads, and no model
training. It reports wall/CPU time and candidate/decode counts and checks matching
cardinality. Vary `--histories` to inspect growth. Like the cache benchmark, it is
opt-in and imposes no flaky automated timing threshold.

The reduced TensorFlow/MAE Dispatch grid also completed all four trials with two
concurrent subprocess workers; a follow-up run verified one completed terminal
history row and a restored finite MAE result for each trial. This exercises the
shared-Store concurrency path, but is not a measurement of the full GPU workload.
Contention can still add bounded retry waits; these quiescent timings measure the
normal path, not a guarantee of unchanged latency under concurrent mutation.

### Follow-up verification

- Timed representative `good-enough` gate: **4,489 passed, 29 skipped**, **13m17s**
  complete local wall time across both interpreter phases. Test-selection policy
  was not weakened.
- Focused state/retention/query/history checks and concurrent Experiment workers
  passed, including metadata-only retries, spent budgets, iterator no-replay,
  quotation correlation/opacity, and constructor-encoding work guards.
- Changed Python F/E9 lint and whitespace checks passed.
- An additional targeted heavy native recovery run encountered six XGBoost
  prerequisite failures: `big_env` contains 3.4.2 while the framework requires
  3.4.1. Dependency pins were preserved; these cases are not qualified by this run.
- Exhaustive, installed-package, real Ray and full GPU-workload suites were not run.

## 2026-10-27 sustained-contention follow-up

The larger notebook subsequently exhausted retries in checkpoint capture/lineage
validation and managed preparation publication. Global epoch invalidation was
discarding unchanged independent metadata. Store-private publication and selected
capture/lineage validation now revalidate their exact byte/directory dependencies
and may rebase onto the actual newer cut when all remain unchanged. Dependent CAS
conflicts, explicit epoch dependencies, public prepared mutations, exact replay
and query continuation cuts remain strict. Reused state-source facts contribute
their complete read witnesses, and rebased journals retain the actual commit base
and all failure/recovery evidence. See [formats](formats.md) and [locking](locking.md).

The same serial alternating cache comparison against `bb890dd3` was rerun after
these changes, with three fresh-process samples per implementation/depth:

| Cache recipe | `bb890dd3` median wall / CPU | Sustained-contention follow-up wall / CPU |
| --- | --- | --- |
| One-item source | 2.689s / 2.619s | 2.576s / 2.553s |
| Same source plus five Take wrappers | 4.112s / 4.081s | 4.033s / 3.997s |

These samples show no normal-path regression against the committed baseline.
Deterministic tests inject independent publication during both preparation and
writer acquisition and assert one preparation, fresh commit-epoch evidence, and
no lost CAS, hidden dependency, or duplicate mutation after failure/interruption.

Two CPU Dispatch stress runs completed widths 32/64/128/256 with two concurrent
workers, 512-example batches and eight train/evaluation batches. To exercise
repeated Fold progress publication without the full dataset, the disposable
reproduction forced the validated checkpoint cadence to two batches. It verified
completed terminal histories and restored finite MAE values for all four trials.
This is reduced-data concurrency evidence, not a full GPU-training qualification.

The sustained-contention final representative gate passed **4,498 tests**, with
**29 skipped**, in **13m48s** across both interpreter phases. Changed Python F/E9
lint and whitespace checks passed. Three subprocess readiness cases in an earlier
targeted run exceeded their existing 20-second deadline during heavy host IO;
all five startup cases then passed in 2.6s, the complete 61-case journal/read-set
group passed, and the final routine gate passed without changing timeouts or
selection policy. Full GPU, exhaustive and installed-package verification was
not run.

## Original performance-pass verification

- Affected subsystem selection: **702 passed** under the representative policy.
- Final routine gate: **4,447 passed, 29 skipped**, **23m47s** complete local wall
  time across both interpreter phases on the loaded host. Test selection was not
  weakened or expanded to exhaustive coverage.
- Changed Python F/E9 lint and diff whitespace checks passed.
- Exhaustive, installed-package and real Ray/GPU integration suites were not run.
