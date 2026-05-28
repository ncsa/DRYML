from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TrainState:
    epoch: int = 0
    step: int = 0
    phase: str | None = None

    def advance_epoch(self, n: int = 1):
        self.epoch += n

    def advance_step(self, n: int = 1):
        self.step += n


__all__ = ["TrainState"]
