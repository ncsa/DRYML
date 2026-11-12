"""Opt-in cached-object lookup benchmark with many containing saved snapshots.

Run ``python -m tests.tools.benchmark_cached_lookup`` in big_env. Fixture creation
uses a deterministic environment, one NumPy cache item and disposable TMPDIR
storage. Timed queries use real SQLite, graph validation and completion records.
"""

import argparse
import json
from tempfile import TemporaryDirectory
from time import perf_counter, process_time

import numpy as np

from dryml.artifacts import CachedDataset
from dryml.core import Definition, Ref, Repo
from dryml.core.store.dir import DirStore
from dryml.environments import introspection
from dryml.managed import ManagedConfig
from tests.artifacts.test_cached_dataset import CountingDataset
from tests.conftest import _synthetic_environment_record
from tests.core.test_query_decode_work import DecodeState, DecodeValue


def main():
    """Measure object lookup and pinned completion after untimed fixture creation.

    Reads positive --snapshots/--repeats and nonnegative --padding counts from
    argv. --expression selects the original projection terminal (default) or an
    explicit object domain. Prints JSON samples and returns None; invalid counts fail
    through argparse. Runtime/authority failures propagate after temporary Store
    cleanup and restoration of the process's original environment observer.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", type=int, default=20)
    parser.add_argument("--padding", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--expression", choices=("projection", "object-domain"), default="projection")
    args = parser.parse_args()
    if args.snapshots < 1 or args.repeats < 1 or args.padding < 0:
        parser.error("snapshots/repeats must be positive; padding must be nonnegative")
    original = introspection.inspect_current
    introspection.inspect_current = _synthetic_environment_record
    try:
        with TemporaryDirectory(prefix="dryml-cached-lookup-") as root:
            path = root + "/store"
            repo = Repo(DirStore(path))
            try:
                source = Definition(CountingDataset, [np.asarray([1], dtype=np.int32)], 1).concretize()
                recipe = Definition(CachedDataset, Ref(source))
                cache = repo.build(recipe.concretize())
                completed = cache.compute(codec="numpy", managed=ManagedConfig(state_repo=repo))
                padding = [DecodeValue("padding-" + str(index), repo=repo) for index in range(args.padding)]
                parent = DecodeState([cache, *padding], repo=repo)
                for index in range(args.snapshots):
                    parent.progress = index
                    repo.save(parent)
            finally:
                repo.close(flush=False)
            for sample in range(args.repeats):
                repo = Repo(DirStore.open_existing(path))
                try:
                    started, cpu = perf_counter(), process_time()
                    query = repo.query().sel(recipe)
                    matches = (query.object_projection() if args.expression == "projection" else
                               query.object_refs().collect())
                    found = perf_counter()
                    selected = CachedDataset.find_completed_state_ref(matches.one(), repo=repo)
                    finished = perf_counter()
                    cpu_seconds = process_time() - cpu
                    assert selected.graph_equal(completed.final_state_ref)
                    stats = matches.diagnostic()
                    print(json.dumps({"sample": sample, "snapshots": args.snapshots,
                        "expression": args.expression,
                        "padding": args.padding, "lookup_seconds": found-started,
                        "completion_seconds": finished-found, "cpu_seconds": cpu_seconds,
                        "candidate_rows_read": stats.candidate_rows_read,
                        "cdef_blobs_decoded": stats.cdef_blobs_decoded}), flush=True)
                finally:
                    repo.close(flush=False)
    finally:
        introspection.inspect_current = original


if __name__ == "__main__":
    main()
