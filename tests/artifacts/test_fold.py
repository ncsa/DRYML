"""U5 deferred Fold traversal and managed-lifecycle coverage."""

from __future__ import annotations

import importlib.util
import os
import numpy as np
import pytest
from pathlib import Path
import subprocess
import sys
from threading import Event, Thread

from dryml.core import Repo
from dryml.core.store.dir import DirStore
from dryml.core.tensor_spec import Dynamic, TensorSpec
from dryml.data import Dataset, StreamDataset
from dryml.managed import ManagedConfig, ManagedConflictError, ManagedInterrupted
from dryml.methods import Accumulator, AccumulatorGroup, ImplementationSelectionError, IteratorPort, Method, StreamNode, traits


OBSERVATION_SPEC = TensorSpec("float32", shape=(Dynamic,), batch=Dynamic, backend="numpy")
CARRY_SPEC = TensorSpec("float64", shape=(), backend="numpy")


class CountingDataset(Dataset):
    """Re-iterable batched source that records traversal and iterator cleanup."""

    iterations = 0
    yields = 0
    closed = 0

    def __init__(self, batches, *, spec=OBSERVATION_SPEC, fail_after=None):
        self.batches = tuple(batches)
        self.fail_after = fail_after
        super().__init__(spec=spec)

    @classmethod
    def reset(cls):
        """Clear process-local instrumentation shared by reconstructed sources."""

        cls.iterations = cls.yields = cls.closed = 0

    def __iter__(self):
        """Yield configured batches once per fresh iterator and record closure."""

        type(self).iterations += 1
        try:
            for index, batch in enumerate(self.batches):
                if self.fail_after == index:
                    raise OSError("source failure")
                type(self).yields += 1
                yield batch
        finally:
            type(self).closed += 1


class InitialTotal(Method):
    """Allocate a scalar float64 carry from the first real source batch."""

    def __init__(self):
        self.calls = 0
        self.inferred = 0

    @traits(backend="numpy", batch_mode="batched")
    def numpy(self, observation):
        """Return an independent zero-valued carry without consuming the batch."""

        self.calls += 1
        return np.array(0.0, dtype=np.float64)

    def infer_output_spec(self, observation_spec):
        """Declare a scalar NumPy carry without executing allocation."""

        self.inferred += 1
        return CARRY_SPEC


class Adding(Accumulator):
    """Accumulate the sum of every element in an uneven batch stream."""

    def __init__(self, *, interrupt_after=None, fail=False):
        self.calls = 0
        self.selections = 0
        self.interrupt_after = interrupt_after
        self.fail = fail

    def find_implementation(self, *args, **kwargs):
        """Count Fold's one-time selected carrier binding."""

        self.selections += 1
        return super().find_implementation(*args, **kwargs)

    @traits(backend="numpy", batch_mode="batched")
    def numpy(self, observation, state):
        """Return a fresh scalar carry after one batch transition."""

        self.calls += 1
        if self.interrupt_after == self.calls:
            raise KeyboardInterrupt()
        if self.fail:
            raise ValueError("transition failure")
        return state + observation.sum(dtype=np.float64)

    def infer_output_spec(self, observation_spec, state_spec):
        """Preserve the independent scalar carry specification."""

        return state_spec


class FinishTotal(Method):
    """Convert the final scalar carry to a host scalar result."""

    fail_globally = False

    def __init__(self, *, fail=False):
        self.calls = 0
        self.fail = fail

    @traits(backend="numpy")
    def numpy(self, state):
        """Return the terminal result or raise the configured branch failure."""

        self.calls += 1
        if self.fail or type(self).fail_globally:
            raise ValueError("finalizer failure")
        return state

    def infer_output_spec(self, state_spec):
        """Declare a scalar float64 terminal result without invoking the Method."""

        return TensorSpec("float64", shape=(), backend="numpy")


def _fold(source, *, accumulator=None, finalizer=None):
    """Build one declared Fold with reusable test Method dependencies."""

    from dryml.artifacts import Fold

    return Fold(
        source,
        initial_state=InitialTotal(),
        accumulator=Adding() if accumulator is None else accumulator,
        finalize=FinishTotal() if finalizer is None else finalizer,
    )


def _run_native_fold_test_in_subprocess(request, backend):
    """Run one native Fold assertion in a child without loading its framework here.

    Args:
        request: Pytest request for the selected native test.
        backend: Optional native backend required by the test.

    Returns:
        ``True`` when the parent ran the selected assertion in a child and should
        return; ``False`` when the guarded child must execute the assertion body.

    Raises:
        AssertionError: If the child test fails or imports a native framework
            into the parent process.
    """
    framework = {"torch": "torch", "tf": "tensorflow"}[backend]
    if importlib.util.find_spec(framework) is None:
        pytest.skip(f"optional {framework} backend is not installed")
    if os.environ.get("DRYML_FOLD_NATIVE_SUBPROCESS") == "1":
        return False

    loaded_frameworks = {name for name in ("torch", "tensorflow") if name in sys.modules}
    env = os.environ.copy()
    env["DRYML_FOLD_NATIVE_SUBPROCESS"] = "1"
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "--no-cov", request.node.nodeid],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, (
        f"native Fold child failed:\n{completed.stdout}\n{completed.stderr}"
    )
    assert {name for name in ("torch", "tensorflow") if name in sys.modules} == loaded_frameworks
    return True


def test_fold_declaration_is_inert_and_persists_without_source_or_method_calls(tmp_path):
    """A declared Fold retains source/Methods without traversal or helper construction."""

    CountingDataset.reset()
    source = CountingDataset([np.ones((2, 3), dtype=np.float32)])
    initializer = InitialTotal()
    accumulator = Adding()
    finalizer = FinishTotal()
    from dryml.artifacts import Fold

    fold = Fold(source, initial_state=initializer, accumulator=accumulator, finalize=finalizer)
    repo = Repo(DirStore(tmp_path / "store"))
    state = repo.save_object(fold)
    restored = Repo(DirStore(tmp_path / "store")).load_state_ref(state, reuse_live="never")

    assert (CountingDataset.iterations, CountingDataset.yields, CountingDataset.closed) == (0, 0, 0)
    assert (initializer.calls, accumulator.calls, finalizer.calls) == (0, 0, 0)
    assert fold.ready is False
    assert restored.ready is False
    assert state is not None


def test_fold_consumes_one_iterator_once_and_reuses_selected_dynamic_batch_carriers(tmp_path):
    """The first batch initializes once then transitions exactly once in one traversal."""

    CountingDataset.reset()
    source = CountingDataset([
        np.ones((2, 3), dtype=np.float32),
        np.full((1, 3), 2, dtype=np.float32),
    ])
    accumulator = Adding()
    fold = _fold(source, accumulator=accumulator)
    store = DirStore(tmp_path / "store")

    assert fold.compute(managed=ManagedConfig(state_repo=store)) == fold.last_state_ref
    assert fold.value() == 12.0
    assert (CountingDataset.iterations, CountingDataset.yields, CountingDataset.closed) == (1, 2, 1)
    assert accumulator.calls == 2
    assert accumulator.selections == 1


@pytest.mark.parametrize("grouped", (False, True))
def test_fold_retains_role_invokers_without_per_transition_compilation(tmp_path, monkeypatch, grouped):
    """Five and one hundred observations add no compilation or role reselection."""
    from dryml.artifacts import Fold
    from dryml.core import signatures
    compiled = []
    original_compile = signatures.compile_signature
    original_select = Fold._select_followups
    monkeypatch.setattr(signatures, "compile_signature",
                        lambda target, **kw: (compiled.append(target), original_compile(target, **kw))[1])

    def select(self, *specs):
        transition, finalizer = original_select(self, *specs)
        def invoke(observation, state):
            before = len(compiled)
            result = transition(observation, state)
            assert len(compiled) == before
            return result
        return invoke, finalizer

    monkeypatch.setattr(Fold, "_select_followups", select)
    for length in (5, 100):
        child = Adding()
        source = CountingDataset([np.ones((1, 2), dtype=np.float32)] * length)
        if grouped:
            fold = Fold(source, initial_state=MethodGroupInitializer(InitialTotal()),
                        accumulator=AccumulatorGroup({"left": child, "right": child}))
        else:
            fold = _fold(source, accumulator=child)
        fold.compute(managed=ManagedConfig(state_repo=DirStore(tmp_path / str(length))))
        assert child.calls == length * (2 if grouped else 1)
        assert child.selections == (2 if grouped else 1)


def test_fold_group_shares_one_stream_and_keeps_independent_branch_carries(tmp_path):
    """A grouped accumulator receives each observation once with separate carry slots."""

    CountingDataset.reset()
    source = CountingDataset([np.ones((1, 2), dtype=np.float32)] * 3)
    left, right = Adding(), Adding()
    group = AccumulatorGroup({"left": left, "right": right})
    initializer = InitialTotal()
    from dryml.artifacts import Fold

    fold = Fold(
        source,
        initial_state=MethodGroupInitializer(initializer),
        accumulator=group,
        finalize=None,
    )
    store = DirStore(tmp_path / "store")

    fold.compute(managed=ManagedConfig(state_repo=store))
    assert fold.value() == {"left": np.array(6.0), "right": np.array(6.0)}
    assert (CountingDataset.iterations, CountingDataset.yields) == (1, 3)
    assert (left.calls, right.calls) == (3, 3)


def test_fold_failures_and_interruptions_leave_no_partial_payload_then_rerun_fresh(tmp_path):
    """Pre-install failures preserve absence and a rerun gets a new traversal/carry."""

    CountingDataset.reset()
    source = CountingDataset([np.ones((1, 2), dtype=np.float32)] * 3)
    accumulator = Adding(interrupt_after=2)
    fold = _fold(source, accumulator=accumulator)
    store = DirStore(tmp_path / "store")

    with pytest.raises(ManagedInterrupted) as caught:
        fold.compute(managed=ManagedConfig(state_repo=store))
    assert type(caught.value.__cause__) is KeyboardInterrupt
    assert fold.ready is False
    assert CountingDataset.closed == 1

    accumulator.interrupt_after = None
    fold.compute(managed=ManagedConfig(state_repo=store, rerun=True))
    assert fold.value() == 6.0
    assert CountingDataset.iterations == 2


@pytest.mark.usefixtures("fixed_managed_snapshot_environment")
def test_fold_rejects_empty_and_source_failures_without_replacing_old_result(tmp_path):
    """Bad source input and pre-install errors retain a prior complete Value payload."""

    store = DirStore(tmp_path / "store")
    CountingDataset.reset()
    FinishTotal.fail_globally = False
    source = CountingDataset([np.ones((1, 1), dtype=np.float32)])
    fold = _fold(source)
    fold.compute(managed=ManagedConfig(state_repo=store))
    assert fold.value() == 1.0

    FinishTotal.fail_globally = True
    try:
        with pytest.raises(ValueError, match="finalizer failure"):
            fold.compute(managed=ManagedConfig(state_repo=store, rerun=True))
        assert fold.value() == 1.0
        checkpoint = fold.compute.status(state_repo=store).checkpoint_state_ref
        saved = Repo(store).load_state_ref(checkpoint, reuse_live="never")
        assert saved._fold_progress_version == 2
        assert saved._fold_progress["eof"] == "root-output-eof"
    finally:
        FinishTotal.fail_globally = False

    assert fold.compute(managed=ManagedConfig(state_repo=store)) == fold.last_state_ref
    assert fold.value() == 1.0
    assert CountingDataset.iterations == 2

    empty = _fold(CountingDataset([]))
    with pytest.raises(ValueError, match="empty"):
        empty.compute(managed=ManagedConfig(state_repo=DirStore(tmp_path / "empty")))
    assert empty.ready is False

    broken = _fold(CountingDataset([np.ones((1, 1), dtype=np.float32)], fail_after=0))
    with pytest.raises(OSError, match="source failure"):
        broken.compute(managed=ManagedConfig(state_repo=DirStore(tmp_path / "broken")))
    assert broken.ready is False


def test_fold_keeps_complete_live_result_when_final_managed_publication_fails(tmp_path, monkeypatch):
    """Payload installation precedes final state/control publication without false success."""

    CountingDataset.reset()
    source = CountingDataset([np.ones((1, 2), dtype=np.float32)])
    fold = _fold(source)
    store = DirStore(tmp_path / "store")
    save_object = Repo.save_object

    def fail_final_save(repo, obj, *args, **kwargs):
        if obj.ready:
            raise OSError("state publication failed")
        return save_object(repo, obj, *args, **kwargs)

    monkeypatch.setattr(Repo, "save_object", fail_final_save)
    with pytest.raises(OSError, match="state publication failed"):
        fold.compute(managed=ManagedConfig(state_repo=store))
    assert fold.ready is True
    assert fold.value() == 2.0
    status = fold.compute.status(state_repo=store)
    assert (status.state, status.failure_code, status.final_state_ref) == (
        "failed", "publication_error", None,
    )
    checkpoint = status.checkpoint_state_ref
    saved = Repo(store).load_state_ref(checkpoint, reuse_live="never")
    assert saved._fold_progress_version == 2
    assert saved._fold_progress["eof"] == "root-output-eof"
    monkeypatch.setattr(Repo, "save_object", save_object)
    from dryml.artifacts import Fold
    monkeypatch.setattr(Fold, "_load_source", lambda *_: pytest.fail("publication retry reopened source"))
    receipt = fold.compute(managed=ManagedConfig(state_repo=store))
    assert receipt == fold.last_state_ref == fold.compute.status(state_repo=store).final_state_ref
    assert fold.value() == 2.0


@pytest.mark.usefixtures("fixed_managed_snapshot_environment")
def test_fold_rejects_missing_or_ambiguous_specs_and_invalid_role_results(tmp_path):
    """Input, carry, and finalizer contract errors cannot install a partial Value."""

    from dryml.artifacts import Fold

    missing = _fold(CountingDataset([np.ones((1, 1), dtype=np.float32)], spec=None))
    with pytest.raises(ValueError, match="spec"):
        missing.compute(managed=ManagedConfig(state_repo=DirStore(tmp_path / "missing")))
    assert missing.ready is False

    ambiguous = _fold(CountingDataset(
        [{"left": np.ones((1, 1), dtype=np.float32), "right": np.ones((1,), dtype=np.float32)}],
        spec={
            "left": TensorSpec("float32", shape=(1,), batch=Dynamic, backend="numpy"),
            "right": TensorSpec("float32", shape=(1,), backend="numpy"),
        },
    ))
    with pytest.raises(ValueError, match="batch"):
        ambiguous.compute(managed=ManagedConfig(state_repo=DirStore(tmp_path / "ambiguous")))
    assert ambiguous.ready is False

    zero = _fold(CountingDataset([np.empty((0, 1), dtype=np.float32)]))
    with pytest.raises(ValueError, match="zero size"):
        zero.compute(managed=ManagedConfig(state_repo=DirStore(tmp_path / "zero")))
    assert zero.ready is False

    for name, initializer, accumulator, finalizer in (
        ("initial", BadInitial(), Adding(), FinishTotal()),
        ("carry", InitialTotal(), BadCarry(), FinishTotal()),
        ("final", InitialTotal(), Adding(), BadFinalizer()),
    ):
        fold = Fold(
            CountingDataset([np.ones((1, 1), dtype=np.float32)]),
            initial_state=initializer,
            accumulator=accumulator,
            finalize=finalizer,
        )
        with pytest.raises(Exception):
            fold.compute(managed=ManagedConfig(state_repo=DirStore(tmp_path / name)))
        assert fold.ready is False


def test_fold_rejects_mixed_initializer_carry_before_the_first_transition(tmp_path, request):
    """A NumPy observation cannot select a transition with a Torch initializer carry."""

    if _run_native_fold_test_in_subprocess(request, "torch"):
        return

    import dryml.torch
    import torch
    from dryml.artifacts import Fold

    class TorchCarryInitial(Method):
        """Declare a Torch carry while remaining selectable from a NumPy observation."""

        @traits(backend="numpy", batch_mode="batched")
        def numpy(self, observation):
            """Return the incompatible native carry for the transition check."""

            return torch.zeros((), dtype=torch.float64)

        def infer_output_spec(self, observation_spec):
            """Declare the deliberately incompatible carry backend."""

            return TensorSpec("float64", shape=(), backend="torch")

    class MustNotTransition(Accumulator):
        """Record any transition that runs after an invalid carry selection."""

        def __init__(self):
            self.calls = 0

        @traits(backend="numpy", batch_mode="batched")
        def numpy(self, observation, state):
            """Fail if selection did not reject the mixed backend first."""

            self.calls += 1
            raise AssertionError("mixed carry reached transition")

        def infer_output_spec(self, observation_spec, state_spec):
            """Retain the initializer carry contract for the selected transition."""

            return state_spec

    accumulator = MustNotTransition()
    fold = Fold(
        CountingDataset([np.ones((1, 1), dtype=np.float32)]),
        initial_state=TorchCarryInitial(),
        accumulator=accumulator,
    )

    with pytest.raises(ImplementationSelectionError) as error:
        fold.compute(managed=ManagedConfig(state_repo=DirStore(tmp_path / "store")))
    assert error.value.reason == "conflict"
    assert accumulator.calls == 0
    assert fold.ready is False


def test_fold_freezes_dynamic_coordinate_shape_from_first_observation(tmp_path):
    """A dynamic source coordinate is concrete carry state after the prototype."""

    from dryml.artifacts import Fold

    fold = Fold(
        CountingDataset([
            np.ones((1, 3), dtype=np.float32),
            np.ones((1, 4), dtype=np.float32),
        ]),
        initial_state=DynamicWidthInitial(),
        accumulator=DynamicWidthAccumulator(),
    )
    fold._install_value_payload({
        "format": "dryml.artifacts.value",
        "version": 1,
        "present": True,
        "result": np.array([9.0]),
    })

    with pytest.raises(Exception):
        fold.compute(managed=ManagedConfig(state_repo=DirStore(tmp_path / "store")))

    assert np.array_equal(fold.value(), np.array([9.0]))


def test_fold_validates_first_observation_against_declared_source_spec(tmp_path):
    """An actual prototype cannot weaken a caller's fixed coordinate contract."""

    from dryml.artifacts import Fold

    initializer = InitialTotal()
    fold = Fold(
        CountingDataset(
            [np.ones((1, 4), dtype=np.float32)],
            spec=TensorSpec("float32", shape=(3,), batch=Dynamic, backend="numpy"),
        ),
        initial_state=initializer,
        accumulator=Adding(),
        finalize=FinishTotal(),
    )

    with pytest.raises(ValueError, match="source observation"):
        fold.compute(managed=ManagedConfig(state_repo=DirStore(tmp_path / "store")))

    assert initializer.calls == 0
    assert fold.ready is False


@pytest.mark.parametrize("backend", ("numpy", "torch", "tf"))
def test_fold_rejects_zero_sized_native_observation_before_initializer(tmp_path, backend, request):
    """Native zero-sized inputs fail before allocation or selected Method calls."""

    if backend != "numpy" and _run_native_fold_test_in_subprocess(request, backend):
        return

    initializer = NativeInitial()
    fold = _native_fold(backend, (0, 1), initializer=initializer)

    with pytest.raises(ValueError, match="zero size"):
        fold.compute(managed=ManagedConfig(state_repo=DirStore(tmp_path / backend)))

    assert initializer.calls == 0
    assert fold.ready is False


@pytest.mark.parametrize("backend", ("torch", "tf"))
def test_fold_normalizes_native_terminal_result_only_after_completion(tmp_path, backend, request):
    """Native carries remain native until Fold converts the final result to host data."""

    if _run_native_fold_test_in_subprocess(request, backend):
        return

    initializer = NativeInitial()
    fold = _native_fold(backend, (1, 3), fill=1, initializer=initializer)

    fold.compute(managed=ManagedConfig(state_repo=DirStore(tmp_path / backend)))

    assert fold.value() == 3.0
    assert initializer.calls == 1


def test_fold_named_mean_program_keeps_the_first_batch_and_uneven_weighting(tmp_path):
    """U6's declared mean program uses Fold's one-pass first-item semantics."""

    from dryml.artifacts import mean

    CountingDataset.reset()
    source = CountingDataset([
        np.array([[1.0, 10.0], [3.0, 30.0]], dtype=np.float64),
        np.array([[9.0, 90.0]], dtype=np.float64),
    ], spec=TensorSpec("float64", shape=(2,), batch=Dynamic, backend="numpy"))
    fold = mean(source, mode="coordinate")

    fold.compute(managed=ManagedConfig(state_repo=DirStore(tmp_path / "mean")))

    assert np.allclose(fold.value(), [13 / 3, 130 / 3])
    assert (CountingDataset.iterations, CountingDataset.yields) == (1, 2)


@pytest.mark.usefixtures("fixed_managed_snapshot_environment")
def test_mean_fold_definition_read_characterization(tmp_path, record_testsuite_property):
    """Ordinary exact Fold validation reaches independent simple Method records."""
    from collections import Counter
    from dryml.artifacts import mean
    from dryml.core.materialization import build_exact_state_load_plan
    from dryml.core.symbol import ImportRef
    from dryml.core.cdef_codec import _eligible_graph_size
    from tests.core.test_exact_load_metadata_evidence import exact_work_counts

    store = DirStore(tmp_path / "mean", query_index="none")
    writer = Repo(store)
    fold = mean(CountingDataset([np.ones((1, 2), dtype=np.float32)]), mode="coordinate")
    state = writer.save_object(fold)
    reader = Repo(store)
    ledgers = []
    for _ in range(2):
        counts = Counter()
        records = []
        method_counts = Counter()
        def observe(frame, event, arg):
            if event == "call":
                name = frame.f_code.co_name
                if name == "decode" and getattr(frame.f_locals.get("record_type"), "__name__", None) == "DefinitionRecord":
                    records.append(frame.f_locals["payload"]["definition"])
                if name in {"_graph_and_labels", "_role_paths", "_build_graph_nodes", "_realize_prepared_graph"}:
                    owner = frame.f_back
                    while owner is not None:
                        if owner.f_code.co_name == "decode" and "record_type" in owner.f_locals:
                            payload = owner.f_locals["payload"]
                            if payload.get("schema") == "definition":
                                label = payload["definition"]["nodes"][0]["cls"].qualname
                                if label in {"MeanInitial", "MeanUpdate", "MeanFinalize"}:
                                    method_counts[name] += 1
                            break
                        owner = owner.f_back
            if previous is not None:
                previous(frame, event, arg)
        previous = sys.getprofile()
        with exact_work_counts() as counts:
            previous = sys.getprofile()
            sys.setprofile(observe)
            try:
                plan = build_exact_state_load_plan(reader, state)
            finally:
                sys.setprofile(previous)
        assert store._can_reuse_exact_validation()
        names = [graph["nodes"][0]["cls"].qualname for graph in records]
        for name in ("MeanInitial", "MeanUpdate", "MeanFinalize"):
            graph = next(graph for graph in records if graph["nodes"][0]["cls"].qualname == name)
            assert len(graph["nodes"]) == 1
            assert type(graph["nodes"][0]["cls"]) is ImportRef
            assert graph["nodes"][0]["parameters"]["kind"] == "dict"
            assert _eligible_graph_size(graph) > 0
        assert "Fold" in names
        fold_graph = next(graph for graph in records if graph["nodes"][0]["cls"].qualname == "Fold")
        source = dict(fold_graph["nodes"][0]["parameters"]["items"])["src"]
        assert source["kind"] == "link" and source["edge_kind"] == "ref"
        with pytest.raises(TypeError):
            _eligible_graph_size(fold_graph)
        assert counts["dill_load"] == 14
        assert method_counts["_build_graph_nodes"] == 3
        if ledgers:
            assert method_counts["_graph_and_labels"] == method_counts["_role_paths"] == 0
            assert method_counts["_realize_prepared_graph"] == 3
        else:
            assert method_counts["_graph_and_labels"] > 0
            assert method_counts["_role_paths"] > 0
        assert plan.state_ref == state
        ledgers.append(dict(counts))
    record_testsuite_property("u6_mean_ledgers", repr(ledgers))
    assert ledgers[1]["_graph_and_labels"] < ledgers[0]["_graph_and_labels"]
    assert ledgers[1]["decode_cdef_graph"] == ledgers[0]["decode_cdef_graph"] - 3


@pytest.mark.usefixtures("fixed_managed_snapshot_environment")
def test_mean_fold_embedded_source_reference_uses_full_graph_fallback(tmp_path):
    """An exact saved source remains inert, excluded data in the Fold root."""
    from dryml.artifacts import mean
    from dryml.core.materialization import build_exact_state_load_plan
    from dryml.core.cdef_codec import _eligible_graph_size, encode_cdef_graph
    from tests.core.test_exact_load_metadata_evidence import exact_work_counts
    store = DirStore(tmp_path / "store", query_index="none")
    writer = Repo(store)
    source = writer.save_object(CountingDataset([np.ones((1, 2), dtype=np.float32)]))
    fold = mean(source, mode="coordinate")
    state = writer.save_object(fold)
    graph = encode_cdef_graph(state.definition)
    src = dict(graph["nodes"][0]["parameters"]["items"])["src"]
    assert src["kind"] == "link" and src["target"]["kind"] == "state_ref"
    with pytest.raises(TypeError):
        _eligible_graph_size(graph)
    reader = Repo(store)
    build_exact_state_load_plan(reader, state)
    with exact_work_counts() as counts:
        plan = build_exact_state_load_plan(reader, state)
    assert counts["decode_cdef_graph"] > 0
    assert counts["_realize_prepared_graph"] == 3
    assert len(plan.actions) == 1


@pytest.mark.parametrize("backend", ("torch", "tf"))
def test_fold_rejects_restored_native_tensor_payloads(tmp_path, backend, request):
    """The result-only persistence boundary never admits a heavyweight tensor."""

    if _run_native_fold_test_in_subprocess(request, backend):
        return

    fold = _fold(CountingDataset([np.ones((1, 1), dtype=np.float32)]))

    with pytest.raises(ValueError, match="unsupported"):
        fold._install_value_payload({
            "format": "dryml.artifacts.value",
            "version": 1,
            "present": True,
            "result": _native_tensor(backend, (1,)),
        })


@pytest.mark.parametrize("result", (
    np.array([object()], dtype=object),
    np.array([np.inf], dtype=np.float64),
))
def test_fold_rejects_non_lightweight_or_nonfinite_restored_results(result):
    """Restoration accepts neither object arrays nor non-finite numeric results."""

    fold = _fold(CountingDataset([np.ones((1, 1), dtype=np.float32)]))

    with pytest.raises(ValueError):
        fold._install_value_payload({
            "format": "dryml.artifacts.value",
            "version": 1,
            "present": True,
            "result": result,
        })


@pytest.mark.usefixtures("fixed_managed_snapshot_environment")
def test_fold_iterator_close_failure_is_not_completed(tmp_path, monkeypatch):
    """A close error after terminal payload installation remains a failed attempt."""

    ClosingDataset.closed = 0
    source = ClosingDataset([np.ones((1, 1), dtype=np.float32)])
    fold = _fold(source)
    store = DirStore(tmp_path / "store")

    with pytest.raises(OSError, match="iterator close failure"):
        fold.compute(managed=ManagedConfig(state_repo=store))

    assert ClosingDataset.closed == 1
    assert fold.ready is True
    assert fold.compute.status(state_repo=store).state == "failed"
    status = fold.compute.status(state_repo=store)
    assert status.final_state_ref is None
    saved = Repo(store).load_state_ref(status.checkpoint_state_ref, reuse_live="never")
    assert saved._fold_progress_version == 2
    assert saved._fold_progress["eof"] == "root-output-eof"
    assert not saved.ready
    from dryml.artifacts import Fold
    monkeypatch.setattr(Fold, "_load_source", lambda *_: pytest.fail("EOF cleanup retry reopened source"))
    receipt = fold.compute(managed=ManagedConfig(state_repo=store))
    assert receipt == fold.last_state_ref == fold.compute.status(state_repo=store).final_state_ref
    assert fold.value() == 1.0


@pytest.mark.usefixtures("fixed_managed_snapshot_environment")
@pytest.mark.parametrize("prior_value", (False, True))
@pytest.mark.parametrize("failure", ("initializer", "accumulator", "observation", "interrupt", "map", "stream", "acquisition"))
def test_fold_consumer_failure_survives_cleanup_without_replacing_value(tmp_path, monkeypatch, failure, prior_value):
    """Cleanup cannot mask workload errors or authorize a partial completed result."""

    from dryml.artifacts import Fold
    from dryml.data import Map

    first = np.ones((1, 1), dtype=np.float32)
    second = np.full((1, 1), 2, dtype=np.float32)
    source = ClosingDataset((first, second))
    if failure == "map":
        source = Map(source, FailingMap())
    elif failure == "stream":
        source = FailingStream(source)
    fold = _fold(source)
    if prior_value:
        fold._install_value_payload({
            "format": "dryml.artifacts.value", "version": 1,
            "present": True, "result": 99.0,
        })
    if failure == "initializer":
        original = Fold._select_initializer

        def select(self, spec):
            carry_spec, _ = original(self, spec)
            def fail(_observation):
                raise ValueError("initializer primary")
            return carry_spec, fail
        monkeypatch.setattr(Fold, "_select_initializer", select)
    elif failure in {"accumulator", "interrupt"}:
        original = Fold._select_followups

        def select(self, *specs):
            transition, finalizer = original(self, *specs)
            def fail(observation, carry):
                if observation.sum() == 2:
                    if failure == "interrupt":
                        raise KeyboardInterrupt("interrupt primary")
                    raise ValueError("accumulator primary")
                return transition(observation, carry)
            return fail, finalizer
        monkeypatch.setattr(Fold, "_select_followups", select)
    elif failure == "observation":
        source.values = (first, np.empty((0, 1), dtype=np.float32))
        monkeypatch.setattr(Fold, "_load_source", lambda *_: source)
    elif failure == "acquisition":
        from dryml.data import DatasetCursor

        class AcquisitionCursor(DatasetCursor):
            def _read(self):
                raise ValueError("acquisition primary")

        class Owner:
            def close(self):
                raise OSError("iterator close failure")

        def iterator(_self):
            cursor = AcquisitionCursor(iter(()))
            cursor._own(Owner())
            return cursor
        monkeypatch.setattr(ClosingDataset, "iterator", iterator)

    store = DirStore(tmp_path / "store")
    error = ManagedInterrupted if failure == "interrupt" else ValueError
    match = "zero size" if failure == "observation" else (None if failure == "interrupt" else f"{failure} primary")
    with pytest.raises(error, match=match) as caught:
        fold.compute(checkpoint_every=1, managed=ManagedConfig(state_repo=store))
    primary = caught.value.__cause__ if failure == "interrupt" else caught.value
    assert any("iterator close failure" in note for note in getattr(primary, "__notes__", ()))
    status = fold.compute.status(state_repo=store)
    assert status.state in {"failed", "interrupted"}
    assert status.final_state_ref is None
    if failure not in {"initializer", "acquisition"}:
        saved = Repo(store).load_state_ref(status.checkpoint_state_ref, reuse_live="never")
        assert saved._fold_progress["state"] == "active"
        assert saved.processed_count == 1
        assert saved.ready == prior_value
        if prior_value:
            assert saved.value() == 99.0
    assert fold.ready == prior_value
    if prior_value:
        assert fold.value() == 99.0


class FailingMap(Method):
    """Fail on the second observation inside a prepared Map cursor."""

    @traits(backend="numpy", batch_mode="batched")
    def numpy(self, observation):
        """Preserve the first value and raise the primary transform failure later."""
        if observation.sum() == 2:
            raise ValueError("map primary")
        return observation

    def infer_output_spec(self, spec):
        """Preserve the input contract without executing the transform."""
        return spec


def _fail_stream(source):
    """Deliver one root output, then fail without reporting root EOF."""
    yield next(source)
    raise ValueError("stream primary")


class FailingStream(StreamDataset):
    """Exercise Fold failure evidence on an owned custom stream graph root."""

    node = StreamNode(
        name="failing-fold-stream",
        inputs=(IteratorPort(OBSERVATION_SPEC),),
        output=IteratorPort(OBSERVATION_SPEC),
        element_spec_transform=lambda specs: specs[0],
        cardinality_transform=lambda cardinalities: cardinalities[0],
        pull_policy=("one_then_fail",), max_buffered_items=1,
        implementation=_fail_stream,
    )


@pytest.mark.usefixtures("fixed_managed_snapshot_environment")
def test_fold_uses_one_managed_owner_and_services_interrupt_at_checkpoint(tmp_path):
    """A concurrent owner conflicts and a request stops at Fold's next safe point."""

    CountingDataset.reset()
    BlockingAccumulator.entered, BlockingAccumulator.release = Event(), Event()
    fold = _fold(
        CountingDataset([np.ones((1, 1), dtype=np.float32)]),
        accumulator=BlockingAccumulator(),
    )
    store = DirStore(tmp_path / "store")
    outcome = []

    def invoke():
        """Run the only active Fold attempt until the test deliberately releases it."""

        try:
            outcome.append(fold.compute(managed=ManagedConfig(state_repo=store)))
        except BaseException as error:  # pragma: no cover - asserted below.
            outcome.append(error)

    worker = Thread(target=invoke)
    worker.start()
    try:
        assert BlockingAccumulator.entered.wait(10)
        with pytest.raises(ManagedConflictError):
            fold.compute(managed=ManagedConfig(state_repo=store))
        assert fold.compute.request_interrupt(state_repo=store).outcome == "requested"
        assert worker.is_alive()
        BlockingAccumulator.release.set()
        worker.join(10)
        assert not worker.is_alive()
    finally:
        BlockingAccumulator.release.set()
        worker.join(10)

    assert len(outcome) == 1
    assert isinstance(outcome[0], ManagedInterrupted)
    status = fold.compute.status(state_repo=store)
    assert status.state == "interrupted"
    assert status.attempt_id is not None
    assert status.checkpoint_state_ref is not None
    assert fold.ready is False

    assert fold.compute(managed=ManagedConfig(state_repo=store)) == fold.last_state_ref
    assert fold.value() == 1.0
    assert CountingDataset.iterations == 1


class MethodGroupInitializer(Method):
    """Build two independent carries for the mapping-shaped accumulator group."""

    def __init__(self, initializer):
        self.initializer = initializer

    @traits(backend="numpy", batch_mode="batched")
    def numpy(self, observation):
        """Invoke the declared initializer twice without retaining either carry."""

        return {"left": np.array(0.0, dtype=np.float64), "right": np.array(0.0, dtype=np.float64)}

    def infer_output_spec(self, observation_spec):
        """Declare the mapping-shaped independent carry contract."""

        return {"left": CARRY_SPEC, "right": CARRY_SPEC}


class BadInitial(InitialTotal):
    """Return an invalid initial carry to exercise selected output validation."""

    @traits(backend="numpy", batch_mode="batched")
    def numpy(self, observation):
        """Deliberately violate the float64 carry specification."""

        return np.array(0.0, dtype=np.float32)


class BadCarry(Adding):
    """Return an invalid next carry while retaining the declared carry spec."""

    @traits(backend="numpy", batch_mode="batched")
    def numpy(self, observation, state):
        """Deliberately change carry dtype after one transition."""

        return np.array(0.0, dtype=np.float32)


class BadFinalizer(FinishTotal):
    """Return an invalid terminal result after a valid complete carry."""

    @traits(backend="numpy")
    def numpy(self, state):
        """Deliberately violate the inferred float64 final result spec."""

        return np.array(0.0, dtype=np.float32)


class BlockingAccumulator(Adding):
    """Block a Fold transition to expose managed ownership and request behavior."""

    entered = Event()
    release = Event()

    @traits(backend="numpy", batch_mode="batched")
    def numpy(self, observation, state):
        """Wait without a managed checkpoint, then return one valid next carry."""

        type(self).entered.set()
        assert type(self).release.wait(10)
        return state + observation.sum(dtype=np.float64)


class DynamicWidthInitial(Method):
    """Allocate coordinate carry using only the first runtime observation width."""

    @traits(backend="numpy", batch_mode="batched")
    def numpy(self, observation):
        """Return a carry whose coordinate shape follows the allocation prototype."""

        return np.zeros(observation.shape[1:], dtype=np.float32)

    def infer_output_spec(self, observation_spec):
        """Keep the pre-prototype carry coordinate unknown during pure inference."""

        return TensorSpec("float32", shape=(Dynamic,), backend="numpy")


class DynamicWidthAccumulator(Accumulator):
    """Expose a transition that would otherwise let coordinate width drift."""

    @traits(backend="numpy", batch_mode="batched")
    def numpy(self, observation, state):
        """Return a fresh carry sized from the current observation."""

        return np.zeros(observation.shape[1:], dtype=np.float32)

    def infer_output_spec(self, observation_spec, state_spec):
        """Declare that transitions retain the initializer carry slot."""

        return state_spec


class NativeInitial(Method):
    """Create native scalar carry while recording whether initialization ran."""

    def __init__(self):
        self.calls = 0

    @traits(batch_mode="batched")
    def native(self, observation):
        """Create a zero scalar in the observation's selected backend."""

        self.calls += 1
        if type(observation).__module__.startswith("tensorflow"):
            import tensorflow as tf

            with tf.device("/CPU:0"):
                return _native_sum(observation) * 0
        return _native_sum(observation) * 0

    def infer_output_spec(self, observation_spec):
        """Declare a native scalar carry without allocating it during inference."""

        return TensorSpec("float32", shape=(), backend=observation_spec.backend)


class NativeAccumulator(Accumulator):
    """Advance the native scalar carry without converting observations to host data."""

    @traits(batch_mode="batched")
    def native(self, observation, state):
        """Add the native observation sum to the native carry."""

        if type(observation).__module__.startswith("tensorflow"):
            import tensorflow as tf

            with tf.device("/CPU:0"):
                return state + _native_sum(observation)
        return state + _native_sum(observation)

    def infer_output_spec(self, observation_spec, state_spec):
        """Retain the scalar carry specification."""

        return state_spec


class ClosingDataset(Dataset):
    """Dataset whose owned iterator reports a close failure after normal exhaustion."""

    closed = 0

    def __init__(self, values):
        self.values = tuple(values)
        super().__init__(spec=OBSERVATION_SPEC)

    def __iter__(self):
        """Return one explicitly closable iterator owned by this Fold invocation."""

        dataset = self

        class Iterator:
            """Iterate values and fail only when Fold releases the owned resource."""

            def __init__(self):
                self.index = 0

            def __iter__(self):
                return self

            def __next__(self):
                if self.index == len(dataset.values):
                    raise StopIteration
                value = dataset.values[self.index]
                self.index += 1
                return value

            def close(self):
                type(dataset).closed += 1
                raise OSError("iterator close failure")

        return Iterator()


def _native_tensor(backend, shape, *, fill=0):
    """Create one CPU-native float32 tensor after explicitly loading its plugin."""

    if backend == "numpy":
        return np.full(shape, fill, dtype=np.float32)
    if backend == "torch":
        import dryml.torch
        import torch

        return torch.full(shape, fill, dtype=torch.float32)
    if backend == "tf":
        import dryml.tf
        import tensorflow as tf

        with tf.device("/CPU:0"):
            return tf.fill(shape, tf.cast(fill, tf.float32))
    raise AssertionError(f"unexpected backend {backend!r}")


def _native_sum(value):
    """Sum one native tensor without converting it to a host representation."""

    if type(value).__module__.startswith("tensorflow"):
        import tensorflow as tf

        return tf.reduce_sum(value)
    return value.sum()


class NativeDataset(Dataset):
    """Create one backend-native observation only when Fold starts its traversal."""

    def __init__(self, backend, shape, *, fill=0):
        self.backend = backend
        self.shape = tuple(shape)
        self.fill = fill
        super().__init__(spec=TensorSpec(
            "float32", shape=(Dynamic,), batch=Dynamic, backend=backend,
        ))

    def __iter__(self):
        """Yield the declared CPU-native observation without storing it in the CDef."""

        yield _native_tensor(self.backend, self.shape, fill=self.fill)


def _native_fold(backend, shape, *, fill=0, initializer):
    """Build a Fold over one backend-native batched observation."""

    from dryml.artifacts import Fold

    return Fold(
        NativeDataset(backend, shape, fill=fill),
        initial_state=initializer,
        accumulator=NativeAccumulator(),
    )
