"""Opt-in fresh-Store cache benchmark; emits JSON lines and restores every result.

Run with ``python -m tests.tools.benchmark_cached_dataset --depth 5 --repeats 3``
in the development environment. Each sample owns a disposable TemporaryDirectory
under TMPDIR, uses real environment observation and SQLite publication, and times
build and compute separately. Imports, restore verification and cleanup are outside
the compute measurement. This is not a timing assertion or an automated test.
"""

import argparse
import json
from tempfile import TemporaryDirectory
from time import perf_counter, process_time

import numpy as np

from dryml.artifacts import CachedDataset
from dryml.core import Definition, Ref, Repo
from dryml.core.store.dir import DirStore
from dryml.data.structural import Take
from dryml.managed import ManagedConfig
from tests.artifacts.test_cached_dataset import CountingDataset


def main():
    """Benchmark CLI-selected recipe depth/repeats; reject invalid counts via argparse.

    Reads --depth (nonnegative int) and --repeats/--elements (positive ints) from
    argv. Returns None after emitting one JSON object per sample. Framework/IO
    failures propagate with cleanup; source traversal and restored content are
    verified rather than reporting a failed/incomplete invocation as a timing.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--elements", type=int, default=1)
    args = parser.parse_args()
    if args.depth < 0 or args.repeats < 1 or args.elements < 1:
        parser.error("depth must be nonnegative; repeats and elements must be positive")
    for sample in range(args.repeats):
        source = Definition(CountingDataset, [np.asarray([1], dtype=np.int32)] * args.elements, args.elements)
        for _ in range(args.depth):
            source = Definition(Take, source, args.elements)
        recipe = Definition(CachedDataset, Ref(source.concretize()))
        with TemporaryDirectory(prefix="dryml-cache-benchmark-") as root:
            repo = Repo(DirStore(root + "/store"))
            try:
                started = perf_counter()
                concrete = recipe.concretize()
                cache = (repo.build(concrete) if hasattr(repo, "build") else
                         repo._load_structural(concrete, require_store=False))
                built = perf_counter()
                cpu_started = process_time()
                result = cache.compute(codec="numpy", managed=ManagedConfig(state_repo=repo))
                cpu_seconds = process_time() - cpu_started
                completed = perf_counter()
                saved = getattr(result, "final_state_ref", result)
                restored = repo.load_state_ref(saved, reuse_live="never")
                assert restored.ready and len(restored) == args.elements and restored.peek().tolist() == [1]
                print(json.dumps({"sample": sample, "depth": args.depth, "elements": args.elements,
                    "build_seconds": built - started, "compute_seconds": completed - built,
                    "compute_cpu_seconds": cpu_seconds}), flush=True)
            finally:
                repo.close(flush=False)


if __name__ == "__main__":
    main()
