from __future__ import annotations

from ..context_tracker import ContextBootstrapError
from ..plain.context import PlainComputeContext


class TorchComputeContext(PlainComputeContext):
    name = "torch"

    def child_env(self) -> dict[str, str]:
        env = super().child_env()
        if self.allocation is None:
            raise ContextBootstrapError("No allocation available")

        if self.allocation.gpu_ids:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.allocation.gpu_ids))
        return env

    def child_setup(self) -> None:
        super().child_setup()

        if self.allocation is None:
            raise ContextBootstrapError("No allocation available")

        import torch

        # visible devices are renumbered after CUDA_VISIBLE_DEVICES
        visible_gpu_count = len(self.allocation.gpu_ids)
        for local_idx in range(visible_gpu_count):
            frac = self.allocation.assigned.get(f"gpu/{self.allocation.gpu_ids[local_idx]}", 1.0)
            if frac < 1.0:
                torch.cuda.memory.set_per_process_memory_fraction(frac, device=local_idx)

    def child_teardown(self) -> None:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def compute_devices(self) -> list[str]:
        if self.allocation is None or self.allocation.num_gpus == 0:
            return ["cpu"]
        return [f"cuda:{i}" for i in range(self.allocation.num_gpus)]
