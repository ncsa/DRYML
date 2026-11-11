"""Synthetic notebook-scale Store and fresh-process query latency scenarios.

The fixture publishes real metadata through DirStore, then builds its real SQLite
index once. It contains no private payloads, training, optional ML backend, user
data or network input. Saved selections have valid creation evidence and synthetic
content addresses: these tests exercise metadata queries, not payload restoration.
"""

from dataclasses import dataclass
from collections.abc import Mapping
from hashlib import sha256
import argparse
import json
from pathlib import Path
import sys
import time
from uuid import NAMESPACE_URL, uuid5

from dryml.core import Definition, F, Node, Object, ObjectId, Par, Ref, Repo, Serializable, Template
from dryml.core.state_values import LocalState, StateKey
from dryml.core.store.dir import DirStore
from dryml.core.utils.graph.path import GraphPath, Parameter


WIDTHS = (32, 64, 128, 256, 512, 1024, 2048)
COPIES = 5
COHORTS = ((10000, 2000), (100000, 20000))
HISTORY_PER_EXPERIMENT = 2
DECOYS = 256
INSTANCES = len(WIDTHS) * COPIES * len(COHORTS)
HISTORIES = len(WIDTHS) * len(COHORTS) * HISTORY_PER_EXPERIMENT
BUILD_TIMEOUT_SECONDS = 90
QUERY_TIMEOUT_SECONDS = 8.0
SCENARIO_TIMEOUT_BUDGET_SECONDS = 60.0
ROOT = GraphPath()
MODEL = GraphPath((Parameter("model"),))


class _Sequence(Object):
    """Retain inert layer factories without loading a backend."""

    def __init__(self, layer_defs):
        raise AssertionError("query workload constructed a sequence")


class _Model(Serializable):
    """Model declaration with independent saved identities and shared recipes."""

    def __init__(self, encoder, decoder):
        raise AssertionError("query workload constructed a model")


class _Dataset(Object):
    """Fixed data size/seed declaration; no actual dataset is allocated."""

    def __init__(self, n, seed):
        raise AssertionError("query workload constructed a dataset")


class _CachedDataset(Serializable):
    """State-pinned source whose quoted recipe remains exact constructor data."""

    def __init__(self, src: Ref[Node]):
        raise AssertionError("query workload constructed a cached dataset")


class _Metric(Object):
    """Quoted metric recipe carrying literal references to its experiment."""

    def __init__(self, test_ds: Ref[Node], model: Ref[Node]):
        raise AssertionError("query workload constructed a metric")


class _Experiment(Serializable):
    """Backend-free analogue of a model/data/artifact experiment graph."""

    def __init__(self, model, train_data, test_data, artifacts: Mapping[str, Template]):
        raise AssertionError("query workload constructed an experiment")


class _History(Object):
    """Repeated ancestor with two paths to the same model occurrence."""

    def __init__(self, experiment, model, step):
        raise AssertionError("query workload constructed a history object")


class _Unrelated(Object):
    """Unrelated indexed roots keep the fixture from being an all-match toy."""

    def __init__(self, number, payload):
        raise AssertionError("query workload constructed an unrelated object")


@dataclass(frozen=True)
class Scenario:
    """One named query contract, its independent expected count and hard wall budget.

    The budget covers a fresh worker's startup, Store reopen, selector construction,
    cold and repeated query, authority checks, and shutdown. Fixture construction
    is separately bounded once for the entire test module.
    """

    name: str
    expected: int
    timeout_seconds: float = QUERY_TIMEOUT_SECONDS


SCENARIOS = (
    Scenario("model_count", len(WIDTHS) + 2 * INSTANCES),
    Scenario("experiment_count", len(WIDTHS) + INSTANCES // len(COHORTS)),
    # Each identified model contributes one association path in each of its
    # saved/object views; each experiment view contributes two model paths.
    # The two unbound recipe views each add one path per width/cohort. Each
    # history has saved/object views with four model paths apiece.
    Scenario("nested_model_count", 6 * INSTANCES + 2 * len(WIDTHS) * len(COHORTS)
             + 8 * HISTORIES),
    Scenario("saved_experiments", INSTANCES // len(COHORTS)),
    Scenario("width_refinement", COPIES),
    Scenario("model_objects", INSTANCES),
    Scenario("fixed_dataset_miss", 0),
)


def _identity(label):
    return ObjectId.from_data({"namespace": ["queryperf"], "nonce": str(uuid5(NAMESPACE_URL, label))})


def _state(creation, label):
    return LocalState(StateKey(creation.creation_digest(), "perf-" + sha256(label.encode()).hexdigest()), creation)


def _model(width):
    def sequence(output):
        return Definition(_Sequence, layer_defs=[F("Dense", units=width, activation="relu"),
                                                F("Dense", units=output, activation="linear")])
    return Definition(_Model, encoder=sequence(2), decoder=sequence(3))


def _experiment(width, train_size, test_data):
    metric = Definition(_Metric, test_ds=Par("this.test_data"), model=Par("this.model"))
    return Definition(_Experiment, model=_model(width),
                      train_data=Definition(_Dataset, train_size, seed=10), test_data=test_data,
                      artifacts={"mae": metric.quote()})


def build_store(path: Path) -> dict:
    """Publish the synthetic graph once and certify its real derived index.

    Args:
        path: New disposable Store directory beneath the test's temporary root.
    Returns:
        JSON-compatible fixture dimensions, projection census, and setup timing.
    Raises:
        ValueError: If the destination exists. Publication/index errors propagate.
    Side Effects:
        Creates metadata and derived SQLite only beneath path; never invokes an
        Object constructor, creates payloads, observes an environment or trains.
    """
    if path.exists():
        raise ValueError("performance fixture requires a new disposable Store")
    started = time.monotonic()
    store = DirStore(path, query_index="none")
    datasets, publications = {}, 0
    try:
        for train_size, test_size in COHORTS:
            creation = Definition(_CachedDataset, Definition(_Dataset, test_size, seed=42).concretize()).concretize()
            label = "cached-" + str(test_size)
            saved = creation.with_realization(identities={ROOT: _identity(label)}, states={ROOT: _state(creation, label)})
            record = store.publish_node(saved, kind="saved")
            datasets[str(train_size)] = record.digest
            publications += 1
            for width in WIDTHS:
                template = _experiment(width, train_size, saved).concretize()
                model_creation = template.parameters["model"].creation_view()
                store.publish_node(template)
                publications += 1
                for copy in range(COPIES):
                    label = f"experiment-{train_size}-{width}-{copy}"
                    selected = template.with_realization(
                        identities={ROOT: _identity(label), MODEL: _identity("model-" + label)},
                        states={ROOT: _state(template, label), MODEL: _state(model_creation, "model-" + label)})
                    store.publish_node(selected, kind="saved")
                    publications += 1
                    for step in range(HISTORY_PER_EXPERIMENT if copy == 0 else 0):
                        history = Definition(_History, experiment=selected, model=selected.parameters["model"], step=step).concretize()
                        store.publish_node(history)
                        publications += 1
            print(f"fixture cohort {train_size}: {publications} publications in {time.monotonic() - started:.2f}s", file=sys.stderr, flush=True)
        for number in range(DECOYS):
            store.publish_node(Definition(_Unrelated, number, {"enabled": True, "values": list(range(8))}).concretize())
            publications += 1
    finally:
        store.close()
    publish_seconds = time.monotonic() - started
    print(f"fixture publication complete: {publications} roots in {publish_seconds:.2f}s; indexing", file=sys.stderr, flush=True)
    store = DirStore.open_existing(path)
    try:
        index = store.open_query_index()
        index.maintain_store()
        assert index.status().state == "ready"
        con = index._connections.connection(readonly=True)
        families = dict(con.execute("SELECT family,count(*) FROM store_projection_rows GROUP BY family"))
    finally:
        store.close()
    summary = {"version": 1, "widths": list(WIDTHS), "copies": COPIES, "cohorts": list(COHORTS),
               "experiments": INSTANCES, "histories": HISTORIES,
               "decoys": DECOYS, "publications": publications, "dataset_roots": datasets,
               "families": families, "publish_seconds": publish_seconds,
               "build_seconds": time.monotonic() - started}
    (path.parent / "query-fixture.json").write_text(json.dumps(summary), encoding="utf-8")
    return summary


def _query(repo, selector, template, name):
    if name == "model_count":
        return repo.query().sel(selector.model)
    if name == "experiment_count":
        return repo.query().sel(selector)
    if name == "nested_model_count":
        return repo.query().nested().sel(selector.model)
    if name == "saved_experiments":
        return repo.query().saved().sel(selector)
    if name == "width_refinement":
        return repo.query().saved().sel(selector).sel(template.sub(width=WIDTHS[0]))
    if name == "model_objects":
        return repo.query().object_refs().sel(selector.model)
    if name == "fixed_dataset_miss":
        other = template.with_kwargs(train_data=Definition(_Dataset, 123456, seed=10)).loose_selector()
        return repo.query().sel(other)
    raise ValueError("unknown performance scenario")


def run_scenario(path: Path, name: str, *, legacy: bool = False) -> dict:
    """Execute a real public query cold and repeated against an unchanged Store.

    Args:
        path: Prepared fixture Store. name: A registered SCENARIOS name.
        legacy: Test-only sensitivity probe disabling the optimized count hooks
            in this disposable worker; never enabled by ordinary tests.
    Returns:
        Counts and elapsed call times for both evaluations.
    Raises:
        AssertionError: For wrong answers, mutation, materialization, or backend
            imports. The parent test enforces the process's hard wall deadline.
    Side Effects:
        Reads fixture metadata and its index. No payload read or Store mutation.
    """
    case = next(case for case in SCENARIOS if case.name == name)
    if legacy:
        from dryml.core.query import indexed_count
        indexed_count._try_identity_count = lambda *a, **k: None
        indexed_count._try_count = lambda *a, **k: None
    manifest = json.loads((path.parent / "query-fixture.json").read_text(encoding="utf-8"))
    store = DirStore.open_existing(path)
    repo = Repo(store)
    try:
        before = store.authority_cut()
        index_before = store.open_query_index().status()
        assert index_before.state == "ready"
        cached = store.read_node(manifest["dataset_roots"][str(COHORTS[-1][0])], kind="saved")
        template = _experiment(Par("width"), COHORTS[-1][0], cached)
        selector = template.loose_selector()
        results = []
        for temperature in ("cold", "repeated"):
            started = time.monotonic()
            count = _query(repo, selector, template, name).count()
            elapsed = time.monotonic() - started
            assert count == case.expected, f"{name}: expected {case.expected}, got {count}"
            results.append({"phase": temperature, "count": count, "seconds": elapsed})
        assert store.authority_cut() == before
        index_after = store.open_query_index().status()
        assert index_after.state == "ready" and index_after.generation == index_before.generation
        assert index_after.diagnostics == index_before.diagnostics
        assert repo._num_constructions == 0
        assert not {"tensorflow", "torch", "jax", "ray"}.intersection(sys.modules)
        return {"scenario": name, "results": results}
    finally:
        repo.close(flush=False)


def main(argv=None):
    """Run one build/query worker; emit JSON or propagate an actionable failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", *(case.name for case in SCENARIOS)))
    parser.add_argument("store", type=Path)
    parser.add_argument("--legacy", action="store_true", help="test-only fallback sensitivity probe")
    args = parser.parse_args(argv)
    result = build_store(args.store) if args.action == "build" else run_scenario(args.store, args.action, legacy=args.legacy)
    print(json.dumps(result), flush=True)
