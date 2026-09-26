"""Focused resumable Fold progress and native-carry coverage."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from dryml.core import Repo
from dryml.core.store.dir import DirStore
from dryml.core.tensor_spec import Dynamic, TensorSpec
from dryml.core.utils.general import pickle_load
from dryml.data import Dataset, DatasetExhaustedError
from dryml.managed import ManagedConfig, ManagedRerunRequiredError
from dryml.methods import Accumulator, Method, traits


_SPEC = TensorSpec("float64", shape=(), batch=Dynamic, backend="numpy")


class ResumeDataset(Dataset):
    """Re-iterable scalar batches with process-local traversal evidence."""

    iterations = 0
    yields = 0
    limit = None

    def __init__(self, values=(1.0, 2.0, 3.0)):
        self.values = tuple(values)
        super().__init__(spec=_SPEC)

    def __iter__(self):
        """Yield one NumPy batch per declared value."""

        type(self).iterations += 1
        for index, value in enumerate(self.values):
            if type(self).limit is not None and index == type(self).limit:
                return
            type(self).yields += 1
            yield np.asarray([value], dtype=np.float64)


class ResumeInitial(Method):
    """Allocate scalar native carry and count initializer executions."""

    calls = 0

    @traits(backend="numpy", batch_mode="batched")
    def numpy(self, observation):
        """Return one fresh float64 zero scalar."""

        type(self).calls += 1
        return np.asarray(0.0, dtype=np.float64)

    def infer_output_spec(self, observation_spec):
        """Declare fixed NumPy scalar carry."""

        return TensorSpec("float64", shape=(), backend="numpy")


class ResumeSum(Accumulator):
    """Add each scalar batch to native carry."""

    @traits(backend="numpy", batch_mode="batched")
    def numpy(self, observation, state):
        """Return the next scalar sum."""

        return state + observation.sum(dtype=np.float64)

    def infer_output_spec(self, observation_spec, state_spec):
        """Preserve fixed carry specification."""

        return state_spec


class ResumeFinish(Method):
    """Return terminal carry while counting finalization."""

    calls = 0

    @traits(backend="numpy")
    def numpy(self, state):
        """Return the completed scalar carry."""

        type(self).calls += 1
        return state

    def infer_output_spec(self, state_spec):
        """Preserve fixed terminal specification."""

        return state_spec


class NativeResumeDataset(Dataset):
    """Create selected CPU-native observations only during traversal."""

    def __init__(self, backend):
        self.backend = backend
        super().__init__(TensorSpec("float64", shape=(), batch=Dynamic, backend=backend))

    def __iter__(self):
        """Yield three one-element CPU batches."""

        if self.backend == "torch":
            import dryml.torch
            import torch

            for value in (1.0, 2.0, 3.0):
                yield torch.tensor([value], dtype=torch.float64)
        else:
            import dryml.tf
            import tensorflow as tf

            with tf.device("/CPU:0"):
                for value in (1.0, 2.0, 3.0):
                    yield tf.constant([value], dtype=tf.float64)


class ConfusionDataset(Dataset):
    """Yield fixed prediction/target label batches for confusion carry recovery."""

    def __init__(self):
        super().__init__({
            "prediction": TensorSpec("int64", shape=(), batch=Dynamic, backend="numpy"),
            "target": TensorSpec("int64", shape=(), batch=Dynamic, backend="numpy"),
        })

    def __iter__(self):
        """Yield three uneven fixed-domain label batches."""

        for prediction, target in (
                ([0, 1], [0, 0]), ([1], [1]), ([0, 1, 1], [1, 1, 0])):
            yield {
                "prediction": np.asarray(prediction, dtype=np.int64),
                "target": np.asarray(target, dtype=np.int64),
            }


def _fold(values=(1.0, 2.0, 3.0)):
    """Build one inert representative resumable Fold."""

    from dryml.artifacts import Fold

    return Fold(
        ResumeDataset(values),
        initial_state=ResumeInitial(),
        accumulator=ResumeSum(),
        finalize=ResumeFinish(),
    )


def _stop_after_checkpoint(_obj, _context):
    """Leave the just-associated checkpoint as failed resumable authority."""

    raise RuntimeError("stop after checkpoint")


def _contains_forbidden_runtime(value) -> bool:
    """Report live callables, iterators, or managed contexts in a progress tree."""

    from dryml.managed import ManagedContext

    if callable(value) or isinstance(value, ManagedContext) or hasattr(value, "__next__"):
        return True
    if isinstance(value, dict):
        return any(_contains_forbidden_runtime(key) or _contains_forbidden_runtime(item) for key, item in value.items())
    if isinstance(value, (tuple, list)):
        return any(_contains_forbidden_runtime(item) for item in value)
    return False


def test_fold_resumes_saved_carry_and_position_without_rerunning_initializer(tmp_path):
    """Compatible resume restores carry, skips yields, and keeps one initializer call."""

    ResumeDataset.iterations = ResumeDataset.yields = ResumeInitial.calls = ResumeFinish.calls = 0
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    fold = _fold()

    with pytest.raises(RuntimeError, match="stop after checkpoint"):
        fold.compute(checkpoint_every=2, managed=ManagedConfig(
            state_repo=repo, callbacks=[_stop_after_checkpoint],
        ))
    checkpoint = fold.compute.status(state_repo=repo).checkpoint_state_ref
    assert checkpoint is not None
    assert fold.processed_count == 2
    assert ResumeInitial.calls == 1
    assert ResumeFinish.calls == 0

    restored = repo.load_state_ref(checkpoint, reuse_live="never")
    payload_dir = tmp_path / "payload"
    payload_dir.mkdir()
    restored.save_state_to_dir(str(payload_dir), codec=restored.state_codec)
    payload = pickle_load(payload_dir / "fold-progress.pkl")
    assert payload["processed_count"] == 2
    assert payload["progress"]["state"] == "active"
    assert not _contains_forbidden_runtime(payload)

    with pytest.raises(ManagedRerunRequiredError, match="rerun_required"):
        fold.compute(checkpoint_every=3, managed=ManagedConfig(state_repo=repo))
    assert ResumeInitial.calls == 1

    assert fold.compute(checkpoint_every=2, managed=ManagedConfig(state_repo=repo)) is None
    assert fold.value() == 6.0
    assert fold.processed_count == 3
    assert ResumeInitial.calls == 1
    assert ResumeFinish.calls == 1
    assert (ResumeDataset.iterations, ResumeDataset.yields) == (2, 5)


def test_fold_eof_checkpoint_retries_finalization_without_loading_source(tmp_path, monkeypatch):
    """Exhausted progress can finalize after failure without another source traversal."""

    from dryml.artifacts import Fold

    ResumeDataset.iterations = ResumeDataset.yields = ResumeInitial.calls = ResumeFinish.calls = 0
    store = DirStore(tmp_path / "store")
    fold = _fold((2.0, 4.0))

    with pytest.raises(RuntimeError, match="stop after checkpoint"):
        fold.compute(checkpoint_every=100, managed=ManagedConfig(
            state_repo=store, callbacks=[_stop_after_checkpoint],
        ))
    assert ResumeFinish.calls == 0
    assert fold._fold_progress["state"] == "exhausted"

    def fail_source_load(*_args, **_kwargs):
        raise AssertionError("exhausted resume loaded its source")

    monkeypatch.setattr(Fold, "_load_source", fail_source_load)
    assert fold.compute(checkpoint_every=100, managed=ManagedConfig(state_repo=store)) is None
    assert fold.value() == 6.0
    assert fold.processed_count == 2
    assert ResumeFinish.calls == 1
    assert ResumeDataset.iterations == 1


def test_fold_rejects_non_positive_or_non_exact_checkpoint_intervals(tmp_path):
    """Cadence validation rejects booleans, non-integers, zero, and negatives."""

    initial_iterations = ResumeDataset.iterations
    for index, value in enumerate((True, 1.0, 0, -1)):
        fold = _fold((1.0,))
        error = TypeError if value in (True, 1.0) else ValueError
        with pytest.raises(error, match="checkpoint_every"):
            fold.compute(
                checkpoint_every=value,
                managed=ManagedConfig(state_repo=DirStore(tmp_path / str(index))),
            )
        assert fold.processed_count == 0
        assert ResumeDataset.iterations == initial_iterations


def test_fold_store_override_controls_checkpoint_and_final_publication(tmp_path):
    """One declared Store override governs both progress and completed state."""

    routed = DirStore(tmp_path / "routed")
    override = DirStore(tmp_path / "override")
    control = DirStore(tmp_path / "control")
    repo = Repo(routed)
    fold = _fold()

    with pytest.raises(RuntimeError, match="stop after checkpoint"):
        fold.compute(
            checkpoint_every=2,
            store=override,
            managed=ManagedConfig(
                state_repo=repo, control_store=control,
                callbacks=[_stop_after_checkpoint],
            ),
        )
    checkpoint = fold.compute.status(
        state_repo=override, control_store=control,
    ).checkpoint_state_ref
    assert override.read_state_ref_record(checkpoint.digest()).state_ref == checkpoint
    assert routed.read_state_ref_record(checkpoint.digest()) is None

    fold.compute(
        checkpoint_every=2,
        store=override,
        managed=ManagedConfig(state_repo=repo, control_store=control),
    )
    final = fold.compute.status(
        state_repo=override, control_store=control,
    ).final_state_ref
    assert override.read_state_ref_record(final.digest()).state_ref == final
    assert routed.read_state_ref_record(final.digest()) is None


def test_fold_resume_fails_if_fresh_source_exhausts_during_saved_skip(tmp_path):
    """A shortened replay source cannot silently combine its suffix with saved carry."""

    ResumeDataset.limit = None
    store = DirStore(tmp_path / "store")
    fold = _fold()
    with pytest.raises(RuntimeError, match="stop after checkpoint"):
        fold.compute(checkpoint_every=2, managed=ManagedConfig(
            state_repo=store, callbacks=[_stop_after_checkpoint],
        ))

    ResumeDataset.limit = 1
    try:
        with pytest.raises(DatasetExhaustedError) as caught:
            fold.compute(checkpoint_every=2, managed=ManagedConfig(state_repo=store))
        assert (caught.value.requested, caught.value.yielded) == (2, 1)
        assert fold.ready is False
        assert fold.processed_count == 2
    finally:
        ResumeDataset.limit = None

    fold.compute(checkpoint_every=2, managed=ManagedConfig(state_repo=store))
    assert fold.value() == 6.0


def test_fold_resume_crosses_a_fresh_process_without_cursor_transport(tmp_path):
    """A second interpreter restores checkpoint carry and opens its own cursor."""

    store_path = tmp_path / "store"
    interrupt_script = """
import sys
from tests.artifacts.test_fold_resume import _fold, _stop_after_checkpoint
from dryml.core import Repo
from dryml.core.store.dir import DirStore
from dryml.managed import ManagedConfig
store = DirStore(sys.argv[1])
repo = Repo(store)
fold = _fold()
try:
    fold.compute(checkpoint_every=2, managed=ManagedConfig(
        state_repo=repo, callbacks=[_stop_after_checkpoint],
    ))
except RuntimeError as error:
    assert str(error) == 'stop after checkpoint'
else:
    raise AssertionError('Fold did not stop after its checkpoint')
checkpoint = fold.compute.status(state_repo=repo).checkpoint_state_ref
assert checkpoint is not None
print(checkpoint.digest())
"""
    interrupted = subprocess.run(
        [sys.executable, "-c", interrupt_script, str(store_path)],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True, text=True, check=False,
    )
    assert interrupted.returncode == 0, interrupted.stderr
    checkpoint_digest = interrupted.stdout.strip().splitlines()[-1]

    resume_script = """
import sys
from dryml.core import Repo
from dryml.core.store.dir import DirStore
from dryml.managed import ManagedConfig
store = DirStore.open_existing(sys.argv[1])
record = store.read_state_ref_record(sys.argv[2])
repo = Repo._for_state_io((store,))
fold = repo.load_state_ref(record.state_ref, reuse_live='never')
assert fold.compute(checkpoint_every=2, managed=ManagedConfig(state_repo=repo)) is None
assert fold.value() == 6.0
assert fold.processed_count == 3
assert fold.compute.status(state_repo=repo).state == 'completed'
"""
    resumed = subprocess.run(
        [sys.executable, "-c", resume_script, str(store_path), checkpoint_digest],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True, text=True, check=False,
    )
    assert resumed.returncode == 0, resumed.stderr


def test_fold_resumes_reservoir_and_confusion_carry_without_losing_components(tmp_path):
    """Reservoir counters/key/storage and confusion counts survive checkpoints."""

    from dryml.artifacts import Fold, quantile
    from dryml.metrics import ConfusionCounts, ConfusionInitial

    resumed_quantile = quantile(
        ResumeDataset((1.0, 9.0, 3.0, 7.0, 5.0)), 0.5,
        mode="global", capacity=2, seed=17,
    )
    direct_quantile = quantile(
        ResumeDataset((1.0, 9.0, 3.0, 7.0, 5.0)), 0.5,
        mode="global", capacity=2, seed=17,
    )
    resumed_store = DirStore(tmp_path / "quantile-resumed")
    with pytest.raises(RuntimeError, match="stop after checkpoint"):
        resumed_quantile.compute(checkpoint_every=2, managed=ManagedConfig(
            state_repo=resumed_store, callbacks=[_stop_after_checkpoint],
        ))
    resumed_quantile.compute(
        checkpoint_every=2, managed=ManagedConfig(state_repo=resumed_store),
    )
    direct_quantile.compute(
        checkpoint_every=100, managed=ManagedConfig(
            state_repo=DirStore(tmp_path / "quantile-direct"),
        ),
    )
    np.testing.assert_array_equal(resumed_quantile.value(), direct_quantile.value())

    confusion = Fold(
        ConfusionDataset(),
        initial_state=ConfusionInitial((0, 1)),
        accumulator=ConfusionCounts((0, 1)),
    )
    confusion_store = DirStore(tmp_path / "confusion")
    with pytest.raises(RuntimeError, match="stop after checkpoint"):
        confusion.compute(checkpoint_every=2, managed=ManagedConfig(
            state_repo=confusion_store, callbacks=[_stop_after_checkpoint],
        ))
    confusion.compute(
        checkpoint_every=2, managed=ManagedConfig(state_repo=confusion_store),
    )
    np.testing.assert_array_equal(confusion.value(), np.asarray([[1, 2], [1, 2]]))
    assert confusion.processed_count == 3


def _run_optional_backend_test_in_subprocess(request, backend):
    """Run optional native carry coverage without importing frameworks in parent."""

    framework = {"torch": "torch", "tf": "tensorflow"}[backend]
    if importlib.util.find_spec(framework) is None:
        pytest.skip(f"optional {framework} backend is not installed")
    if os.environ.get("DRYML_FOLD_RESUME_NATIVE_SUBPROCESS") == "1":
        return False
    loaded = {name for name in ("torch", "tensorflow") if name in sys.modules}
    env = os.environ.copy()
    env["DRYML_FOLD_RESUME_NATIVE_SUBPROCESS"] = "1"
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "--no-cov", request.node.nodeid],
        cwd=Path(__file__).resolve().parents[2], env=env,
        capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert {name for name in ("torch", "tensorflow") if name in sys.modules} == loaded
    return True


@pytest.mark.parametrize("backend", ("torch", "tf"))
def test_fold_resume_round_trips_optional_cpu_native_carry(tmp_path, request, backend):
    """Torch and TensorFlow CPU carry survive a real serialized checkpoint boundary."""

    if _run_optional_backend_test_in_subprocess(request, backend):
        return

    from dryml.artifacts import mean

    fold = mean(NativeResumeDataset(backend), mode="global")
    store = DirStore(tmp_path / backend)
    with pytest.raises(RuntimeError, match="stop after checkpoint"):
        fold.compute(checkpoint_every=2, managed=ManagedConfig(
            state_repo=store, callbacks=[_stop_after_checkpoint],
        ))
    fold.compute(checkpoint_every=2, managed=ManagedConfig(state_repo=store))

    assert fold.value() == 2.0
    assert fold.processed_count == 3
