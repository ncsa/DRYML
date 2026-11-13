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

## Snapshot-heavy cached-object discovery

A subsequent owner-reported cache lookup spent more than four minutes before its
first result message. The supplied expression ended in `object_projection()`, an
eager collect-and-project terminal; the later fixed-set `count()` calls were not
rerunning the query. Its default domain included recipe and saved-state histories.

Read-only diagnosis of the supplied Store found 192 saved roots, 32 recipe roots,
10,049 discovery contributions, current schema-11/coverage-3 certification, and no
pending mutations. The selected cache appeared in 140 containing snapshot blocks.
SQL candidate selection took about 0.13s; repeated graph hydration and validation
dominated. Almost all first-page blocks exceeded the existing decoded-owner cache
entry budget, so increasing the number of retries or reusing a query variable
would not solve the work amplification.

Identified-object discovery now addresses authoritative declaration projections
using optional derived identity hints. Acceptance still checks source membership,
declaration contributions, exact projection digests, all identified dependencies,
cross-Store consistency, fresh cuts/bytes and residual predicates. Stateless
objects and source-root-dependent restrictions retain ordinary graph discovery.
The selected object no longer requires reconstructing every unrelated containing
model snapshot. The decoder also proves already-identical typed syntax directly,
using a DAG-aware comparison and canonical fallback for reordered representations.

For a root-identity-pinned Artifact recipe, completion lookup first seeks one
matching independently stored snapshot and reads that receiver's unique managed
`compute` record. Embedded-only saved knowledge still has its complete fallback;
unpinned recipes retain full ambiguity checking.

The explicit object-identity form is:

```python
matches = repo.query().object_refs().sel(cached_test_ds_def).collect()
if matches.count() == 1:
    completed = CachedDataset.find_completed_state_ref(matches.one(), repo=repo)
```

An isolated `9f1336d2` source export and the patched worktree queried the same
unchanged ready Store using the cache recipe reconstructed from its metadata.
The comparison below uses the object-only expression for
both implementations, so it excludes the original expression's extra recipe and
saved-domain work. Fresh-process wall measurements used warm filesystem metadata:

| Operation | `9f1336d2` | Patched | Decoded blocks before → after |
| --- | --- | --- | --- |
| Cached object lookup | 78.076s | 2.414s | 206 → 1 |
| Pinned completion lookup | 80.199s | 0.420s | 205 → 2 |
| Combined | 158.275s | 2.834s | — |

An earlier patched lookup took 29s during filesystem/host contention, so the warm
timings are not a cold-IO or service-latency guarantee. The stable improvement is
the eliminated graph work: the same 159 object contributions are validated while
decoding one canonical object projection. No private dataset payload was opened,
and no Store migration, rebuild or authoritative mutation was performed.

Use this credential-free opt-in reproduction with synthetic containing snapshots:

```bash
TMPDIR=/tmp/dryml CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu \
  python -m tests.tools.benchmark_cached_lookup \
  --snapshots 20 --padding 32 --repeats 3
```

It builds one NumPy cache item, publishes multiple containing snapshots, reopens a
fresh Repo per sample, and verifies the recovered completed state. Timings and
candidate/decode counts are emitted as JSON. Routine tests enforce work counts and
authority/fallback behavior rather than machine-dependent timing thresholds.

The original cache-compute benchmark was also rerun with three fresh processes
per implementation/depth, alternating baseline/current order. Median compute wall
and CPU times remained lower than `9f1336d2`:

| Cache recipe | `9f1336d2` wall / CPU | Patched wall / CPU |
| --- | --- | --- |
| One-item source | 2.646s / 2.623s | 2.506s / 2.484s |
| Same source plus five Take wrappers | 4.083s / 4.051s | 3.826s / 3.795s |

Snapshot-query verification: **4,510 passed, 29 skipped** in the final
`good-enough` run, **12m09s** complete wall time across both interpreter phases.
Focused codec/query/artifact checks, the synthetic benchmark, changed Python
F/E9 lint and whitespace checks passed. Existing shared-root traversal tests
caught redundant identity enumeration during development; page-local memoization
and the shared-root fallback retain those work bounds. No test-selection or
timeout policy was weakened. Exhaustive, installed-package and full GPU workloads
were not run for this metadata-only change.

### Follow-up: the unchanged projection expression

The owner retried the original `.sel(recipe).object_projection()` expression and
still observed about three minutes. The preceding measurements had exercised a
rewritten object-only query; they did not establish that the original default-
domain terminal was fast. The terminal now conservatively proves when discovery
and projection can be fused: an ordinary unbounded Repo/Store selection by plain
state-independent Node chains, with no branching or repeated Node children,
quotations, matchers, callbacks, symbolic adapters or other restrictions. The
proof inspects at most 1,024 constructor values. Unproved patterns, state pins,
explicit domains, bounds and composed sources retain collect-then-project.
Regression tests compare results and source evidence to that unfused operation
and retain state-sensitive, quoted, stateless, embedded and bounded behavior.

The **unchanged** expression was compared against the isolated `9f1336d2` export
on the same ready user Store, reconstructing the selected cached test recipe from
authority (Take count 20,000; source seed 42):

| Original default-domain projection | `9f1336d2` | Patched |
| --- | --- | --- |
| Wall time | 230.958s | 2.170s |
| Candidate contributions | 478 | 159 |
| Decoded graph blocks | 617 | 1 |
| Returned object identities | 1 | 1 |

In that paired run, fresh managed completion lookup added 0.404s after the patched
projection. Another original-expression run took 2.318s plus 0.449s for completion.
These are warm-filesystem fresh-process measurements, not cold-IO latency bounds.
The test Store remained at epoch 3076, without pending mutation or derived-index
rebuild. The synthetic lookup benchmark now defaults to this original expression;
pass `--expression object-domain` to measure the explicit domain instead.

Original-expression verification: **293 affected query checks passed**, followed
by **4,514 passed, 29 skipped** in the final representative `good-enough` gate,
**12m39s** complete wall time across both interpreter phases. Changed Python F/E9
lint and whitespace checks passed. The original-expression synthetic benchmark
also returned the expected completed state with one lookup graph decode.

## Large checkpoint association metadata

Repeated receiver/checkpoint/final recipes can exceed the 1 MiB control limit
even when each graph packet fits individually. Control encoding now pools exact
repeated Node value blocks within a record, preserving separate graph tables,
views, bindings and exact previous/proposed source-byte witnesses. It does not
raise the stored-byte limit or open private state.

Two deterministic synthetic regressions exercise this boundary:

| Case | Expanded logical JSON | Stored wire bytes |
| --- | ---: | ---: |
| Seven graph packets in a 43-row final-association preparation | 1,601,539 | 553,387 |
| Largest preparation in an authored checkpoint/completion workflow | 1,636,419 | 214,555 |

The authored workflow completes with one body, two captures and zero restores,
then rebuilds SQLite projections and replays completion with private-state reads
forbidden. Separate tests cover exact legacy/noncanonical source bytes,
owner-exit recovery, distinct saved states, malformed pools and expansion bounds.
These are local synthetic measurements, not a rerun of the user's notebook or a
latency guarantee. Expanded control metadata has its own 8 MiB limit; unique wire
metadata must still fit 1 MiB. See [formats](formats.md#managed-control-formats).

Final verification: **194 affected checks passed**, including 18 sharing/size/
recovery regressions, and **8 offline source/wheel artifact checks passed**.
The representative `good-enough` gate passed **4,834 tests, 29 skipped**, in
**885.09s (14m45s)** complete wall time across both interpreter phases. An earlier
run reached its 890-second cutoff; removing duplicate preparation graph encoding
and reusing already validated detached metadata brought the final run within the
15-minute gate without changing test selection or validation bounds.

## Original performance-pass verification

- Affected subsystem selection: **702 passed** under the representative policy.
- Final routine gate: **4,447 passed, 29 skipped**, **23m47s** complete local wall
  time across both interpreter phases on the loaded host. Test selection was not
  weakened or expanded to exhaustive coverage.
- Changed Python F/E9 lint and diff whitespace checks passed.
- Exhaustive, installed-package and real Ray/GPU integration suites were not run.
