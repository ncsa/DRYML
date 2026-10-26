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

## Verification

- Affected subsystem selection: **702 passed** under the representative policy.
- Final routine gate: **4,447 passed, 29 skipped**, **23m47s** complete local wall
  time across both interpreter phases on the loaded host. Test selection was not
  weakened or expanded to exhaustive coverage.
- Changed Python F/E9 lint and diff whitespace checks passed.
- Exhaustive, installed-package and real Ray/GPU integration suites were not run.
