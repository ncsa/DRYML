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

_MISSING = object()


class PlainComputeContext(ComputeContext):
    name = "plain"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._saved_env: dict[str, str | object] = {}
        self._saved_affinity: set[int] | None = None
        self._saved_rlimit_as: tuple[int, int] | None = None

    def bootstrap_env(self) -> dict[str, str]:
        if self.allocation is None:
            raise ContextBootstrapError("No allocation available")

        env = {}
        if self.allocation.num_cpus > 0:
            n = str(self.allocation.num_cpus)
            for key in _THREAD_ENV_KEYS:
                env[key] = n
        return env

    def validate_current(self) -> None:
        return None

    def _save_env(self, env: dict[str, str]) -> None:
        for key in env:
            if key not in self._saved_env:
                self._saved_env[key] = os.environ.get(key, _MISSING)

    def _set_env(self, env: dict[str, str]) -> None:
        for key, val in env.items():
            os.environ[key] = val

    def _restore_env(self) -> None:
        for key, old_val in reversed(list(self._saved_env.items())):
            if old_val is _MISSING:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_val
        self._saved_env.clear()

    def _restore_affinity(self) -> None:
        if self._saved_affinity is not None and hasattr(os, "sched_setaffinity"):
            try:
                os.sched_setaffinity(0, set(self._saved_affinity))
            except Exception:
                pass
            self._saved_affinity = None

    def _restore_rlimit(self) -> None:
        if self._saved_rlimit_as is not None:
            try:
                import resource
                resource.setrlimit(resource.RLIMIT_AS, self._saved_rlimit_as)
            except Exception:
                pass
            self._saved_rlimit_as = None

    def apply_current(self) -> None:
        if self.allocation is None:
            raise ContextBootstrapError("No allocation available")
        if self._applied:
            return

        env = self.bootstrap_env()

        self._save_env(env)

        if self.allocation.cpu_ids and hasattr(os, "sched_getaffinity"):
            try:
                self._saved_affinity = set(os.sched_getaffinity(0))
            except Exception:
                self._saved_affinity = None

        if self.allocation.memory_bytes is not None:
            try:
                import resource
                self._saved_rlimit_as = resource.getrlimit(resource.RLIMIT_AS)
            except Exception:
                self._saved_rlimit_as = None

        try:
            self._set_env(env)

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

            self._applied = True

        except Exception:
            self._restore_affinity()
            self._restore_rlimit()
            self._restore_env()
            raise

    def unapply_current(self) -> None:
        if not self._applied:
            return

        try:
            self._restore_affinity()
            self._restore_rlimit()
            self._restore_env()
        finally:
            self._applied = False
