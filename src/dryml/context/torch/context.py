from __future__ import annotations

import os
import sys

from ..context_tracker import ContextBootstrapError
from ..plain.context import PlainComputeContext
from dryml.core.utils.general import module_is_available


class TorchComputeContext(PlainComputeContext):
    name = "torch"

    def check_compatible_env(self):
        if not module_is_available('torch'):
            raise ContextBootstrapError("Torch not available")

    def bootstrap_env(self) -> dict[str, str]:
        env = super().bootstrap_env()

        if self.allocation is None:
            raise ContextBootstrapError("No allocation available")

        if self.allocation.gpu_ids:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.allocation.gpu_ids))

        return env

    def validate_current(self) -> None:
        super().validate_current()

        if self.allocation is None:
            raise ContextBootstrapError("No allocation available")

        if "torch" not in sys.modules:
            return

        env = self.bootstrap_env()

        if "CUDA_VISIBLE_DEVICES" in env:
            if os.environ.get("CUDA_VISIBLE_DEVICES") != env["CUDA_VISIBLE_DEVICES"]:
                raise ContextBootstrapError(
                    "Cannot change CUDA_VISIBLE_DEVICES after torch is imported"
                )

    def apply_current(self) -> None:
        if self.allocation is None:
            raise ContextBootstrapError("No allocation available")
        if self._applied:
            return

        super().apply_current()
        try:
            import torch

            visible_gpu_count = len(self.allocation.gpu_ids)
            for local_idx in range(visible_gpu_count):
                frac = self.allocation.assigned.get(
                    f"gpu/{self.allocation.gpu_ids[local_idx]}",
                    1.0,
                )
                if frac < 1.0:
                    torch.cuda.memory.set_per_process_memory_fraction(
                        frac,
                        device=local_idx,
                    )
        except Exception:
            super().unapply_current()
            raise

    def unapply_current(self) -> None:
        if not self._applied:
            return

        try:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        finally:
            super().unapply_current()

    def compute_devices(self) -> list[str]:
        if self.allocation is None or self.allocation.num_gpus == 0:
            return ["cpu"]
        return [f"cuda:{i}" for i in range(self.allocation.num_gpus)]
