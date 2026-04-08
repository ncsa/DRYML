from __future__ import annotations

import gc
import os
import sys

from ..context_tracker import ContextBootstrapError
from ..plain.context import PlainComputeContext
from dryml.core2.utils.general import module_is_available, module_is_imported


class JAXComputeContext(PlainComputeContext):
    name = "jax"

    _ENV_KEYS = (
        "JAX_PLATFORMS",
        "CUDA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "XLA_PYTHON_CLIENT_MEM_FRACTION",
        "XLA_PYTHON_CLIENT_PREALLOCATE",
    )

    def check_compatible_env(self):
        if not module_is_available('jax'):
            raise ContextBootstrapError("JAX is not available")

    def bootstrap_env(self) -> dict[str, str]:
        env = super().bootstrap_env()

        if self.allocation is None:
            raise ContextBootstrapError("No allocation available")

        gpu_ids = list(self.allocation.gpu_ids)

        if gpu_ids:
            visible = ",".join(map(str, gpu_ids))

            env["CUDA_VISIBLE_DEVICES"] = visible
            env["HIP_VISIBLE_DEVICES"] = visible

            fractional = [
                self.allocation.assigned.get(f"gpu/{gpu_id}", 1.0)
                for gpu_id in gpu_ids
                if self.allocation.assigned.get(f"gpu/{gpu_id}", 1.0) < 1.0
            ]
            if fractional:
                if len(set(fractional)) != 1:
                    raise ContextBootstrapError(
                        "JAX currently only supports one shared fraction "
                        "across visible GPUs"
                    )
                env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(fractional[0])

        else:
            env["JAX_PLATFORMS"] = "cpu"
            env["CUDA_VISIBLE_DEVICES"] = ""
            env["HIP_VISIBLE_DEVICES"] = ""
            env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

        return env

    def validate_current(self) -> None:
        super().validate_current()

        if self.allocation is None:
            raise ContextBootstrapError("No allocation available")

        desired_env = self.bootstrap_env()

        tf_imported = module_is_imported("tensorflow", include_children=True)
        jax_imported = module_is_imported(
            ["jax", "jaxlib"],
            match_all=False,
            include_children=True,
        )

        # Be conservative: TensorFlow may transitively import JAX via TFLite,
        # so once TensorFlow is in this process, JAX bootstrap configuration is
        # no longer something we can trust to apply safely.
        if tf_imported:
            raise ContextBootstrapError(
                "Cannot safely configure JAX after TensorFlow has been imported "
                "in this process."
            )

        # If JAX is not imported yet, bootstrap-time configuration is still possible.
        if not jax_imported:
            return

        mismatches = []
        for key in self._ENV_KEYS:
            desired = desired_env.get(key)
            current = os.environ.get(key)
            if desired != current:
                mismatches.append(
                    f"{key}: current={current!r}, desired={desired!r}"
                )

        if mismatches:
            raise ContextBootstrapError(
                "Cannot change JAX bootstrap environment after JAX is imported: "
                + "; ".join(mismatches)
            )


    def apply_current(self) -> None:
        if self.allocation is None:
            raise ContextBootstrapError("No allocation available")
        if self._applied:
            return

        super().apply_current()
        try:
            import jax

            devices = list(jax.devices())
            non_cpu_devices = [
                dev for dev in devices
                if getattr(dev, "platform", "cpu") != "cpu"
            ]

            if self.allocation.gpu_ids:
                expected = len(self.allocation.gpu_ids)
                if len(non_cpu_devices) < expected:
                    raise ContextBootstrapError(
                        f"Requested {expected} JAX GPU(s), "
                        f"but only found {len(non_cpu_devices)} visible "
                        f"accelerator device(s)"
                    )
            else:
                if non_cpu_devices:
                    raise ContextBootstrapError(
                        "CPU-only JAX context still sees accelerator devices"
                    )
        except Exception:
            super().unapply_current()
            raise

    def unapply_current(self) -> None:
        if not self._applied:
            return

        try:
            try:
                import jax
                jax.clear_caches()
            except Exception:
                pass
            gc.collect()
        finally:
            super().unapply_current()

    def compute_devices(self) -> list[str]:
        if self.allocation is None or not self.allocation.gpu_ids:
            return ["cpu"]
        return [f"gpu:{i}" for i in range(len(self.allocation.gpu_ids))]
