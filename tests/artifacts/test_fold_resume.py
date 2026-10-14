"""Focused resumable Fold progress and native-carry coverage."""

from __future__ import annotations

import importlib.util
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from dryml.core import Repo
from dryml.core.store.dir import DirStore
from dryml.core.tensor_spec import Dynamic, TensorSpec
from dryml.core.utils.general import pickle_load, pickle_save, unpickler
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


class RepairableDataset(ResumeDataset):
    """Fail between two observations until the caller repairs the source."""

    broken = True

    def __iter__(self):
        """Yield one value, then fail or yield the repaired suffix."""

        yield np.asarray([1.0], dtype=np.float64)
        if type(self).broken:
            raise OSError("source failed before two")
        yield np.asarray([2.0], dtype=np.float64)


class FixtureDataset(ResumeDataset):
    """Replay the frozen fixture's explicitly unnamed batch-axis contract."""

    def __init__(self, values=(1.0, 2.0)):
        self.values = tuple(values)
        Dataset.__init__(self, spec=TensorSpec(
            "float64", shape=(), batch=Dynamic, backend="numpy", batch_axis_name=None,
        ))

    def __iter__(self):
        """Replay NumPy observations with the fixture's absent device metadata."""
        for value in self.values:
            yield np.asarray([value], dtype=np.float64).view(FixtureArray)


class FixtureArray(np.ndarray):
    """Synthetic NumPy observation matching legacy absent device evidence."""

    @property
    def device(self):
        """Return the explicitly absent device fact recorded in frozen progress."""
        return None


@pytest.mark.usefixtures("fixed_managed_snapshot_environment")
def test_fold_failed_read_retains_active_authority_before_repaired_resume(tmp_path, monkeypatch):
    """A failed pull cannot certify EOF or turn recovery into a truncated Value."""

    from dryml.artifacts import Fold

    monkeypatch.setattr(RepairableDataset, "broken", True)
    fold = Fold(RepairableDataset(), initial_state=ResumeInitial(), accumulator=ResumeSum())
    repo = Repo(DirStore(tmp_path / "store"))
    with pytest.raises(OSError, match="source failed before two"):
        fold.compute(checkpoint_every=1, managed=ManagedConfig(state_repo=repo))
    status = fold.compute.status(state_repo=repo)
    assert status.state == "failed"
    assert status.final_state_ref is None
    assert not fold.ready
    checkpoint = status.checkpoint_state_ref
    restored = repo.load_state_ref(checkpoint, reuse_live="never")
    payload_dir = tmp_path / "payload"
    payload_dir.mkdir()
    restored.save_state_to_dir(str(payload_dir), codec=restored.state_codec)
    payload = pickle_load(payload_dir / "fold-progress.pkl")
    assert payload["version"] == 2
    assert payload["processed_count"] == 1
    assert payload["progress"]["state"] == "active"
    assert "eof" not in payload["progress"]
    assert not restored.ready
    monkeypatch.setattr(RepairableDataset, "broken", False)
    receipt = fold.compute(checkpoint_every=1, managed=ManagedConfig(state_repo=repo))
    assert receipt == fold.last_state_ref == fold.compute.status(state_repo=repo).final_state_ref
    assert fold.value() == 3.0
    assert fold.processed_count == 2


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


_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "fold_progress"


def _fixture_bytes(name):
    """Read frozen bytes and verify both checkout and authoritative payload hashes."""

    manifest = json.loads((_FIXTURES / "manifest.json").read_text())
    record = manifest["files"][name]
    encoded = (_FIXTURES / name).read_bytes()
    assert hashlib.sha256(encoded).hexdigest() == record["encoded_sha256"]
    data = base64.b64decode(encoded)
    assert len(data) == record["size"]
    assert hashlib.sha256(data).hexdigest() == record["sha256"]
    return data


def _restore_fixture(fold, name, directory):
    """Restore committed progress bytes through the real Fold state hook."""

    directory.mkdir()
    (directory / "fold-progress.pkl").write_bytes(_fixture_bytes(name))
    fold.restore_state_from_dir_imp(str(directory), codec=fold.state_codec)


def _seed_fixture_checkpoint(fold, repo, tmp_path, name, *, incompatible=None):
    """Associate fixture carry/state with only synthetic invocation identities rebound."""

    from dryml.artifacts import Fold
    from dryml.artifacts.fold import _source_binding_evidence
    from dryml.managed.control import ControlSnapshot, ManagedControlStore
    from dryml.managed.identity import argument_digest, operation_digest
    from dryml.managed.storage import ownership_evidence

    _restore_fixture(fold, name, tmp_path / "fixture")
    progress = fold._fold_progress
    operation_id = operation_digest(fold.object_ref, "compute")
    attempt_id = "a" * 32
    progress.update(source_binding=_source_binding_evidence(fold.src),
                    program=fold._program_evidence(), operation_id=operation_id, attempt_id=attempt_id)
    if incompatible:
        if incompatible == "program":
            progress["program"]["initializer"] = "different-program"
        elif incompatible == "source_binding":
            progress["source_binding"]["digest"] = "different-source"
        else:
            progress[incompatible] = "different-attempt"
    checkpoint = repo.save_object(fold, deep_capture=True)
    arguments = argument_digest(Fold.compute, fold, (), {"checkpoint_every": 100, "managed": None})
    ManagedControlStore(repo.stores[0], repo).create_initial(ControlSnapshot(
        operation_id, fold.object_ref.digest(), arguments, "compute", attempt_id, None,
        1, "failed", None, checkpoint.digest(), None, "method_error",
        ownership_evidence(repo, fold.object_ref.objects.values()),
    ))
    return checkpoint


@pytest.mark.usefixtures("fixed_managed_snapshot_environment")
@pytest.mark.parametrize("name", ("v1-active", "v2-active", "v2-exhausted"))
def test_fold_frozen_progress_compatible_recovery(tmp_path, monkeypatch, name):
    """Frozen active carries resume; trusted v2 exhaustion finalizes source-free."""

    from dryml.artifacts import Fold

    fold = Fold(FixtureDataset(), initial_state=ResumeInitial(), accumulator=ResumeSum(), finalize=ResumeFinish())
    repo = Repo(DirStore(tmp_path / "store"))
    _seed_fixture_checkpoint(fold, repo, tmp_path, name + ".pkl.b64")
    calls = ResumeInitial.calls
    if name == "v2-exhausted":
        monkeypatch.setattr(Fold, "_load_source", lambda *_: pytest.fail("trusted EOF loaded source"))
    receipt = fold.compute(checkpoint_every=100, managed=ManagedConfig(state_repo=repo))
    assert receipt == fold.last_state_ref == fold.compute.status(state_repo=repo).final_state_ref
    assert fold.value() == (1.0 if name == "v2-exhausted" else 3.0)
    assert ResumeInitial.calls == calls


@pytest.mark.usefixtures("fixed_managed_snapshot_environment")
def test_fold_legacy_exhausted_admission_preserves_valid_receiver_and_provenance(tmp_path, monkeypatch):
    """Ambiguous v1 restores, fails in BODY before effects, and accepts explicit rerun."""

    from dryml.artifacts import Fold
    import dryml.artifacts.fold as fold_module

    fold = Fold(FixtureDataset(), initial_state=ResumeInitial(), accumulator=ResumeSum(), finalize=ResumeFinish())
    repo = Repo(DirStore(tmp_path / "store"))
    checkpoint = _seed_fixture_checkpoint(fold, repo, tmp_path, "v1-exhausted.pkl.b64")
    snapshot = Path(repo.stores[0].get_snapshot_directory(checkpoint))
    before = {str(path.relative_to(snapshot)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in snapshot.rglob("*") if path.is_file()}
    restore_carry = fold_module._restore_carry

    def no_materialization(*args, **kwargs):
        assert kwargs.get("materialize") is False, "legacy admission materialized carry"
        return restore_carry(*args, **kwargs)

    with monkeypatch.context() as guard:
        guard.setattr(fold_module, "_restore_carry", no_materialization)
        for method in ("_select_initializer", "_select_followups", "_load_source", "_install_terminal"):
            guard.setattr(Fold, method, lambda *_: pytest.fail("legacy admission invoked a role/source"))
        with pytest.raises(ManagedRerunRequiredError, match="legacy.*rerun=True"):
            fold.compute(checkpoint_every=100, managed=ManagedConfig(state_repo=repo))
    status = fold.compute.status(state_repo=repo)
    assert (status.state, status.failure_code, status.checkpoint_state_ref) == ("failed", "method_error", checkpoint)
    assert status.final_state_ref is None
    assert not fold.ready
    saved = repo.save_object(fold, deep_capture=True)
    restored = repo.load_state_ref(saved, reuse_live="never")
    directory = tmp_path / "roundtrip"
    directory.mkdir()
    restored.save_state_to_dir(str(directory), codec=restored.state_codec)
    payload = pickle_load(directory / "fold-progress.pkl")
    assert payload["version"] == 1
    assert payload["progress"]["state"] == "exhausted"
    assert "eof" not in payload["progress"]
    with pytest.raises(ManagedRerunRequiredError, match="legacy"):
        fold.compute(checkpoint_every=100, managed=ManagedConfig(state_repo=repo))
    receipt = fold.compute(checkpoint_every=100, managed=ManagedConfig(state_repo=repo, rerun=True))
    assert receipt == fold.last_state_ref
    assert fold.value() == 3.0
    assert fold.compute.status(state_repo=repo).attempt_id != status.attempt_id
    assert before == {str(path.relative_to(snapshot)): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in snapshot.rglob("*") if path.is_file()}
    _fixture_bytes("v1-exhausted.pkl.b64")


@pytest.mark.parametrize("name", ("v1-active", "v1-exhausted", "v1-completed", "v2-active", "v2-exhausted"))
def test_fold_frozen_progress_roundtrip_and_downgrade_gate(tmp_path, monkeypatch, name):
    """Frozen bytes restore/roundtrip without blessing old EOF; old readers reject v2."""

    spec = importlib.util.spec_from_file_location("fold_legacy_reader", _FIXTURES / "legacy_reader.py")
    legacy_reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy_reader)

    fold = _fold((1.0, 2.0))
    _restore_fixture(fold, name + ".pkl.b64", tmp_path / "fixture")
    if name == "v1-completed":
        from dryml.artifacts import Value
        (tmp_path / "fixture" / "value.pkl").write_bytes(_fixture_bytes("v1-completed-value.pkl.b64"))
        Value.restore_state_from_dir_imp(fold, str(tmp_path / "fixture"), codec=fold.state_codec)
        assert fold.ready and fold.value() == 3.0
    directory = tmp_path / "saved"
    directory.mkdir()
    fold.save_state_to_dir(str(directory), codec=fold.state_codec)
    payload = pickle_load(directory / "fold-progress.pkl")
    assert payload == unpickler(_fixture_bytes(name + ".pkl.b64"))
    if name == "v1-completed":
        again = _fold()
        again.restore_state_from_dir(str(directory), codec=again.state_codec)
        assert again.value() == 3.0 and again.processed_count == 2
    assert payload["version"] == int(name[1])
    assert payload["processed_count"] == (2 if name.endswith("completed") else 1)
    assert (payload["progress"] is None) == name.endswith("completed")
    if name.startswith("v2"):
        monkeypatch.setattr(legacy_reader, "_restore_carry", lambda *_args, **_kwargs: pytest.fail("old reader reached carry"))
        with pytest.raises(ValueError, match="version is unsupported"):
            legacy_reader._validate_progress_payload(payload)
    else:
        assert legacy_reader._validate_progress_payload(payload)[0] == payload["processed_count"]
    _fixture_bytes(name + ".pkl.b64")


@pytest.mark.parametrize("damage", ("version", "extra", "missing_eof", "active_eof", "unknown_eof", "carry", "count"))
def test_fold_malformed_progress_does_not_partially_install(tmp_path, damage):
    """Closed grammar failures leave the prior in-memory progress and Value intact."""

    payload = unpickler(_fixture_bytes("v2-exhausted.pkl.b64"))
    if damage == "version":
        payload["version"] = 99
    elif damage == "extra":
        payload["progress"]["extra"] = True
    elif damage == "missing_eof":
        del payload["progress"]["eof"]
    elif damage == "active_eof":
        payload["progress"]["state"] = "active"
    elif damage == "unknown_eof":
        payload["progress"]["eof"] = "closed"
    elif damage == "carry":
        payload["progress"]["carry"]["data"] = np.asarray(1, dtype=np.int32)
    else:
        payload["processed_count"] = 2
    fold = _fold()
    _restore_fixture(fold, "v1-active.pkl.b64", tmp_path / "prior")
    prior = fold._fold_progress
    fold._install_value_payload({"format": "dryml.artifacts.value", "version": 1, "present": True, "result": 99.0})
    damaged = tmp_path / "damaged"
    damaged.mkdir()
    pickle_save(payload, damaged / "fold-progress.pkl")
    with pytest.raises(ValueError):
        fold.restore_state_from_dir_imp(str(damaged), codec=fold.state_codec)
    assert fold._fold_progress is prior
    assert fold.processed_count == 1
    assert fold._fold_progress_version == 1
    assert fold.value() == 99.0


@pytest.mark.usefixtures("fixed_managed_snapshot_environment")
@pytest.mark.parametrize("incompatible", ("source_binding", "program", "operation_id", "attempt_id"))
def test_fold_incompatible_progress_rejected_before_runtime_effects(tmp_path, monkeypatch, incompatible):
    """Saved identity mismatches cannot invoke Methods, restore native carry, or complete."""

    import dryml.artifacts.fold as fold_module
    from dryml.artifacts import Fold

    fold = Fold(FixtureDataset(), initial_state=ResumeInitial(), accumulator=ResumeSum(), finalize=ResumeFinish())
    repo = Repo(DirStore(tmp_path / "store"))
    checkpoint = _seed_fixture_checkpoint(fold, repo, tmp_path, "v1-active.pkl.b64", incompatible=incompatible)
    original = fold_module._restore_carry
    def no_materialization(*args, **kwargs):
        assert kwargs.get("materialize") is False
        return original(*args, **kwargs)
    monkeypatch.setattr(fold_module, "_restore_carry", no_materialization)
    with pytest.raises(ValueError, match="incompatible"):
        fold.compute(checkpoint_every=100, managed=ManagedConfig(state_repo=repo))
    assert not fold.ready
    assert fold.compute.status(state_repo=repo).checkpoint_state_ref == checkpoint


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


@pytest.mark.usefixtures("fixed_managed_snapshot_environment")
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
    assert payload["version"] == 2
    assert payload["progress"]["state"] == "active"
    assert "eof" not in payload["progress"]
    assert not _contains_forbidden_runtime(payload)

    with pytest.raises(ManagedRerunRequiredError, match="rerun_required"):
        fold.compute(checkpoint_every=3, managed=ManagedConfig(state_repo=repo))
    assert ResumeInitial.calls == 1

    assert fold.compute(checkpoint_every=2, managed=ManagedConfig(state_repo=repo)) == fold.last_state_ref
    assert fold.value() == 6.0
    assert fold.processed_count == 3
    assert ResumeInitial.calls == 1
    assert ResumeFinish.calls == 1
    assert (ResumeDataset.iterations, ResumeDataset.yields) == (2, 5)


@pytest.mark.usefixtures("fixed_managed_snapshot_environment")
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
    checkpoint = fold.compute.status(state_repo=store).checkpoint_state_ref
    saved = Repo(store).load_state_ref(checkpoint, reuse_live="never")
    assert saved._fold_progress_version == 2
    assert saved._fold_progress["eof"] == "root-output-eof"
    assert not saved.ready

    def fail_source_load(*_args, **_kwargs):
        raise AssertionError("exhausted resume loaded its source")

    monkeypatch.setattr(Fold, "_load_source", fail_source_load)
    assert fold.compute(checkpoint_every=100, managed=ManagedConfig(state_repo=store)) == fold.last_state_ref
    assert fold.value() == 6.0
    assert fold.processed_count == 2
    assert ResumeFinish.calls == 1
    assert ResumeDataset.iterations == 1


@pytest.mark.usefixtures("fixed_managed_snapshot_environment")
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


@pytest.mark.usefixtures("fixed_managed_snapshot_environment")
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


@pytest.mark.usefixtures("fixed_managed_snapshot_environment")
def test_fold_resume_fails_if_fresh_source_exhausts_during_saved_skip(tmp_path, monkeypatch):
    """A shortened replay source cannot silently combine its suffix with saved carry."""

    ResumeDataset.limit = None
    store = DirStore(tmp_path / "store")
    fold = _fold()
    with pytest.raises(RuntimeError, match="stop after checkpoint"):
        fold.compute(checkpoint_every=2, managed=ManagedConfig(
            state_repo=store, callbacks=[_stop_after_checkpoint],
        ))

    checkpoint = fold.compute.status(state_repo=store).checkpoint_state_ref
    original_iterator = ResumeDataset.iterator

    class CleanupFailure:
        def close(self):
            raise OSError("short replay cleanup")

    def iterator(self):
        cursor = original_iterator(self)
        if type(self).limit is not None:
            cursor._own(CleanupFailure())
        return cursor

    monkeypatch.setattr(ResumeDataset, "iterator", iterator)
    ResumeDataset.limit = 1
    try:
        with pytest.raises(DatasetExhaustedError) as caught:
            fold.compute(checkpoint_every=2, managed=ManagedConfig(state_repo=store))
        assert (caught.value.requested, caught.value.yielded) == (2, 1)
        assert any("short replay cleanup" in note for note in caught.value.__notes__)
        assert fold.ready is False
        assert fold.processed_count == 2
        assert fold.compute.status(state_repo=store).checkpoint_state_ref == checkpoint
        assert Repo(store).load_state_ref(checkpoint, reuse_live="never")._fold_progress["state"] == "active"
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
assert fold.compute(checkpoint_every=2, managed=ManagedConfig(state_repo=repo)) == fold.last_state_ref
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


@pytest.mark.usefixtures("fixed_managed_snapshot_environment")
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
    checkpoint = fold.compute.status(state_repo=store).checkpoint_state_ref
    saved = Repo(store).load_state_ref(checkpoint, reuse_live="never")
    assert saved._fold_progress_version == 2
    assert all(item[:2] == (backend, "cpu") for item in saved._fold_progress["backend_devices"])
    receipt = fold.compute(checkpoint_every=2, managed=ManagedConfig(state_repo=store))
    assert receipt == fold.last_state_ref == fold.compute.status(state_repo=store).final_state_ref

    assert fold.value() == 2.0
    assert fold.processed_count == 3
