from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


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


__all__ = ["TrainState"]
