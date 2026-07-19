from __future__ import annotations

import json
import pickle
import tempfile
from pathlib import Path

from dryml.core2.methods import Method
from dryml.managed import ManagedOutput, ManagedOutputs, current_operation_context

from .train_spec import (
    TRAIN_CHECKPOINT_SCHEMA,
    TrainCapability,
)


class TrainFunction(Method):
    """Training procedure and deterministic managed output contract.

    Trainers are non-resumable unless they explicitly advertise an exact
    capability. Exact trainers may use :meth:`checkpoint` at safe points; the
    Experiment restores that checkpoint into a fresh model before re-entry.
    """

    __dryml_managed_outputs__ = ManagedOutputs(
        ManagedOutput(
            "model",
            primary=True,
            kind="stored_state",
            subject_path=("model",),
            representations=("dryml.object_state",),
        )
    )
    __dryml_train_capability__ = TrainCapability.none()

    @classmethod
    def resume_capability(cls, definition=None) -> TrainCapability:
        """Return the immutable capability for this trainer definition."""

        del definition
        capability = cls.__dryml_train_capability__
        if not isinstance(capability, TrainCapability):
            raise TypeError("__dryml_train_capability__ must be a TrainCapability")
        return capability

    def checkpoint(self, exp, *, payload=None) -> str:
        """Commit model, TrainState, and trainer-owned payload atomically.

        Args:
            exp: The invocation-local training view supplied to ``__call__``.
            payload: Optional trusted pickle-compatible trainer cursor/state.

        Returns:
            The immutable managed checkpoint ID.
        """

        context = current_operation_context()
        if context.checkpoint_schema != TRAIN_CHECKPOINT_SCHEMA:
            raise RuntimeError("trainer did not advertise the generic train checkpoint schema")
        with tempfile.TemporaryDirectory(prefix="dryml-train-checkpoint-") as temp:
            root = Path(temp)
            model_root = root / "model"
            model_root.mkdir()
            exp.model.save_state_to_dir(str(model_root))
            (root / "train-state.pkl").write_bytes(pickle.dumps(exp.state, protocol=5))
            (root / "trainer-payload.pkl").write_bytes(pickle.dumps(payload, protocol=5))
            (root / "train.json").write_text(
                json.dumps(
                    {
                        "schema": TRAIN_CHECKPOINT_SCHEMA,
                        "schema_version": 1,
                        "model_cdef_id": exp.model.definition.stable_hash(),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                encoding="utf-8",
            )
            for path in sorted(item for item in root.rglob("*") if item.is_file()):
                context.write_checkpoint(path.relative_to(root).as_posix(), _file_chunks(path))
        return context.commit_checkpoint(
            metadata={"schema": TRAIN_CHECKPOINT_SCHEMA, "schema_version": 1}
        )

    def restore_checkpoint(self, exp, checkpoint_root) -> object:
        """Restore a verified generic train checkpoint into a fresh model."""

        root = Path(checkpoint_root)
        try:
            descriptor = json.loads((root / "train.json").read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError("training checkpoint descriptor is unreadable") from exc
        expected = {
            "schema": TRAIN_CHECKPOINT_SCHEMA,
            "schema_version": 1,
            "model_cdef_id": exp.model.definition.stable_hash(),
        }
        if descriptor != expected:
            raise RuntimeError("training checkpoint is incompatible with the selected model")
        exp.model.restore_state_from_dir(str(root / "model"))
        try:
            exp.state = pickle.loads((root / "train-state.pkl").read_bytes())
            return pickle.loads((root / "trainer-payload.pkl").read_bytes())
        except Exception as exc:
            raise RuntimeError("training checkpoint payload is unreadable") from exc

    def __call__(self, exp):
        raise NotImplementedError("TrainFunction subclasses must implement __call__(exp).")


def _file_chunks(path: Path, size: int = 1024 * 1024):
    with path.open("rb") as handle:
        while chunk := handle.read(size):
            yield chunk


__all__ = ["TrainFunction"]
