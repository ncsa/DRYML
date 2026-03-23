from __future__ import annotations

import os
from ..context_tracker import ComputeContext, ContextBootstrapError

_THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


class PlainComputeContext(ComputeContext):
    name = "plain"

    def child_env(self) -> dict[str, str]:
        if self.allocation is None:
            raise ContextBootstrapError("No allocation available")

        env = {}
        if self.allocation.num_cpus > 0:
            n = str(self.allocation.num_cpus)
            for key in _THREAD_ENV_KEYS:
                env[key] = n
        return env

    def child_setup(self) -> None:
        if self.allocation is None:
            raise ContextBootstrapError("No allocation available")

        if self.allocation.cpu_ids and hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, set(self.allocation.cpu_ids))

        if self.allocation.memory_bytes is not None:
            try:
                import resource
                soft, hard = resource.getrlimit(resource.RLIMIT_AS)
                new_soft = self.allocation.memory_bytes
                if hard != resource.RLIM_INFINITY:
                    new_soft = min(new_soft, hard)
                resource.setrlimit(resource.RLIMIT_AS, (new_soft, hard))
            except Exception:
                pass
