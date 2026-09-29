from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def metric_value(value: Any) -> Any:
    """Convert common tensor/scalar metric values into display-friendly values."""
    if hasattr(value, "detach"):
        value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _format_metric(value: Any) -> str:
    value = metric_value(value)
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def format_metrics(metrics: Mapping[str, Any]) -> str:
    return " - ".join(f"{name}: {_format_metric(value)}" for name, value in metrics.items())


class TrainingProgress:
    def __init__(self, *, total: int | None = None, verbose: int = 1, desc: str = "Training"):
        self.total = total
        self.verbose = int(verbose)
        self.desc = desc
        self._bar = None

        if self.verbose == 1:
            try:
                from tqdm.auto import tqdm
            except Exception:
                tqdm = None
            if tqdm is not None:
                self._bar = tqdm(total=total, desc=desc, leave=True)

    def update(self, n: int = 1, metrics: Mapping[str, Any] | None = None) -> None:
        if self.verbose <= 0:
            return
        if self._bar is None:
            return
        if metrics:
            self._bar.set_postfix({name: _format_metric(value) for name, value in metrics.items()})
        self._bar.update(n)

    def epoch_end(
            self,
            epoch: int,
            *,
            epochs: int | None = None,
            metrics: Mapping[str, Any] | None = None) -> None:
        if self.verbose <= 0 or self._bar is not None:
            return
        label = f"Epoch {epoch}/{epochs}" if epochs is not None else f"Epoch {epoch}"
        if metrics:
            print(f"{label} - {format_metrics(metrics)}")
        else:
            print(label)

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None
