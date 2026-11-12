"""Opt-in fresh-Store history benchmark with JSON timings and decode counts.

Run in big_env with ``python -m tests.tools.benchmark_experiment_history``.
Preparation uses deterministic lightweight test models and a synthetic environment;
timed queries use real Store authority and SQLite. Temporary Stores live under
TMPDIR and are removed on exit. No training or optional ML backend is needed.
"""

import argparse
import json
from tempfile import TemporaryDirectory
from time import perf_counter, process_time

from dryml.core import Definition, Repo
from dryml.core.store.dir import DirStore
from dryml.environments import introspection
from dryml.models import Experiment, ExperimentData
from tests.conftest import _synthetic_environment_record
from tests.models.test_experiment_artifacts import CounterModel, TerminalOnly


def main():
    """Measure validated hit/miss query work after untimed history preparation.

    Reads positive --histories/--repeats and nonnegative --checkpoints integers
    from argv. Emits one JSON object per case/sample and returns None. Invalid
    counts fail through argparse; Store/query failures propagate with cleanup.
    This opt-in CLI temporarily substitutes deterministic environment observation
    during fixture preparation, restoring the original observer on exit.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--histories", type=int, default=8)
    parser.add_argument("--checkpoints", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.histories < 1 or args.checkpoints < 0 or args.repeats < 1:
        parser.error("histories/repeats must be positive; checkpoints must be nonnegative")
    observer = introspection.inspect_current
    introspection.inspect_current = _synthetic_environment_record
    try:
        with TemporaryDirectory(prefix="dryml-history-benchmark-") as root:
            repo = Repo(DirStore(root + "/store"))
            try:
                subjects = []
                for index in range(args.histories + 1):
                    saved = repo.save(Experiment(CounterModel(index), TerminalOnly(), repo=repo))
                    subjects.append(saved)
                    if index == args.histories:
                        continue
                    history = ExperimentData.get_or_create(saved.object_view(), repo=repo)
                    for checkpoint in range(args.checkpoints):
                        history.add_row(expected_artifacts=(), state_ref=saved,
                            prev_state_ref=None, time=checkpoint, examples_seen=checkpoint)
                        history.publish(repo=repo)
                store = repo.default_store
                cases = (
                    ("history-hit", ExperimentData._definition(subjects[0].object_view()), args.checkpoints + 1),
                    ("history-miss", ExperimentData._definition(subjects[-1].object_view()), 0),
                    ("ordinary-model-hit", Definition(CounterModel, 0).concretize(), 1),
                    ("ordinary-model-miss", Definition(CounterModel, -1).concretize(), 0),
                    ("ordinary-experiment-hit", subjects[-1], 1),
                )
                for label, selector, expected in cases:
                    query = store.query().saved().sel(selector).refresh(False)
                    for sample in range(args.repeats):
                        started, cpu = perf_counter(), process_time()
                        result = query.collect()
                        wall_seconds, cpu_seconds = perf_counter() - started, process_time() - cpu
                        assert len(result) == expected
                        stats = result.diagnostic()
                        print(json.dumps({"case": label, "sample": sample,
                            "histories": args.histories, "checkpoints": args.checkpoints,
                            "wall_seconds": wall_seconds, "cpu_seconds": cpu_seconds,
                            "candidate_rows_read": stats.candidate_rows_read,
                            "cdef_blobs_decoded": stats.cdef_blobs_decoded}), flush=True)
            finally:
                repo.close(flush=False)
    finally:
        introspection.inspect_current = observer


if __name__ == "__main__":
    main()
