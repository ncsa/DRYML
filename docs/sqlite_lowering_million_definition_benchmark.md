# SQLite Lowering Million-Definition Benchmark

This benchmark recipe is the sprint closeout contract for broad SQLite lowering behavior. It is intentionally documented outside the untracked local `benchmarks/` directory so it remains versioned with the query-index docs.

## Dataset

Create 1,000,000 stored root definitions with:

- one stable root class;
- a categorical `name` or `bucket` feature distributed across at least 1,000 values;
- one nested child definition on a fixed path for 10% of roots;
- at least one broad unindexed selector case.

## Measurements

For each query, capture `query.explain(analyze=True)` and record:

- `candidate_rows_read`;
- `cdef_blobs_decoded`;
- `python_verifications`;
- `lowering_strategy`;
- `scan_required` and `scan_reason`;
- wall-clock time for `exists()`, `one()`, `count()`, and `defs().first()`.

## Required Contracts

- Exact-root and selective posting queries report `lowering_strategy == "sqlite-lowered"`.
- Exact-root explain plans use the stable-hash index.
- Selective posting explain plans use the postings index or primary-key lookup on `postings`.
- Broad `exists()` decodes fewer CDefs than total candidates when an early verified match exists.
- Broad query-backed `defs()` fetches only the first candidate page when only the first result is consumed.
- Broad unindexed selectors record `scan_required == True` unless `require_indexed()` rejects them.
- `count()` streams verification and does not construct a `DefinitionResultSet`.

## Recommended Command Shape

Use the sprint environment and keep the benchmark separate from default tests:

```bash
source ~/conda.sh && conda activate big_env
python -m dryml.devtools.sqlite_lowering_benchmark --roots 1000000 --store /tmp/dryml-sqlite-lowering-bench
```

The `dryml.devtools.sqlite_lowering_benchmark` entry point is a recommended follow-up CLI wrapper around this recipe; the current sprint implementation has the runtime counters needed for it in `QueryExplanation`.
