from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import ClassVar


TRAIN_CHECKPOINT_SCHEMA = "dryml.experiment-train.v1"


class TrainResumeMode(Enum):
    """Trainer resume guarantee advertised before managed execution."""

    NONE = "none"
    EXACT = "exact"


@dataclass(frozen=True, slots=True)
class TrainCapability:
    """Definition-derived whole-trainer checkpoint capability.

    Exact capability means the trainer can checkpoint every mutable component
    it owns. Completed cached inputs are immutable and are pinned separately by
    the managed runtime.
    """

    mode: TrainResumeMode
    diagnostic: str
    checkpoint_schema: str | None = None
    early_completion: bool = False

    def __post_init__(self):
        if not isinstance(self.mode, TrainResumeMode):
            raise TypeError("train capability mode must be a TrainResumeMode")
        if not isinstance(self.diagnostic, str) or not self.diagnostic:
            raise ValueError("train capability diagnostic must be non-empty")
        expected = TRAIN_CHECKPOINT_SCHEMA if self.mode is TrainResumeMode.EXACT else None
        if self.checkpoint_schema != expected:
            raise ValueError("train checkpoint schema does not match resume mode")
        if not isinstance(self.early_completion, bool):
            raise TypeError("train early_completion capability must be a bool")

    @classmethod
    def none(cls, diagnostic: str = "trainer does not advertise exact resume"):
        """Return an explicit non-resumable trainer capability."""

        return cls(TrainResumeMode.NONE, diagnostic)

    @classmethod
    def exact(cls, diagnostic: str, *, early_completion: bool = False):
        """Return an exact operation-checkpoint capability."""

        return cls(
            TrainResumeMode.EXACT,
            diagnostic,
            TRAIN_CHECKPOINT_SCHEMA,
            early_completion,
        )


@dataclass(slots=True)
class TrainState:
    initial: ClassVar[str | None] = None
    training: ClassVar[str] = "training"
    trained: ClassVar[str] = "trained"
    failed: ClassVar[str] = "failed"

    epoch: int = 0
    step: int = 0
    phase: str | None = initial

    @property
    def is_initial(self) -> bool:
        return self.phase == self.initial

    @property
    def is_training(self) -> bool:
        return self.phase == self.training

    @property
    def is_trained(self) -> bool:
        return self.phase == self.trained

    @property
    def is_failed(self) -> bool:
        return self.phase == self.failed

    def __eq__(self, other):
        if isinstance(other, TrainState):
            return (
                self.epoch == other.epoch
                and self.step == other.step
                and self.phase == other.phase
            )
        if isinstance(other, str) or other is None:
            return self.phase == other
        return NotImplemented

    def advance_epoch(self, n: int = 1):
        self.epoch += n

    def advance_step(self, n: int = 1):
        self.step += n


__all__ = [
    "TRAIN_CHECKPOINT_SCHEMA",
    "TrainCapability",
    "TrainResumeMode",
    "TrainState",
]
