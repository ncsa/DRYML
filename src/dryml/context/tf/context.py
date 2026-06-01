from __future__ import annotations

import os
import sys

from ..context_tracker import ContextBootstrapError
from ..plain.context import PlainComputeContext
from dryml.core2.utils.general import module_is_available


class TFComputeContext(PlainComputeContext):
    name = "tf"

    def check_compatible_env(self):
        if not module_is_available('tensorflow'):
            raise ContextBootstrapError("Tensorflow not available")


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

        if "tensorflow" not in sys.modules:
            return

        env = self.bootstrap_env()

        if "CUDA_VISIBLE_DEVICES" in env:
            if os.environ.get("CUDA_VISIBLE_DEVICES") != env["CUDA_VISIBLE_DEVICES"]:
                raise ContextBootstrapError(
                    "Cannot change CUDA_VISIBLE_DEVICES after tensorflow is imported"
                )

        fractional = [
            self.allocation.assigned[key]
            for key in self.allocation.gpus
            if self.allocation.assigned[key] < 1.0
        ]
        if fractional:
            raise ContextBootstrapError(
                "Cannot configure TensorFlow fractional GPU memory limits "
                "after tensorflow is imported"
            )

    def apply_current(self) -> None:
        if self.allocation is None:
            raise ContextBootstrapError("No allocation available")
        if self._applied:
            return

        super().apply_current()
        try:
            import tensorflow as tf

            gpus = tf.config.list_physical_devices("GPU")
            if not gpus:
                return

            fractional = [
                self.allocation.assigned[key]
                for key in self.allocation.gpus
                if self.allocation.assigned[key] < 1.0
            ]
            if fractional:
                if len(set(fractional)) != 1:
                    raise ContextBootstrapError(
                        "TF currently only supports one shared fraction "
                        "across visible GPUs"
                    )

                frac = fractional[0]
                try:
                    import GPUtil
                    gpu_info = GPUtil.getGPUs()
                except Exception as e:
                    raise ContextBootstrapError(
                        "GPUtil is required for TF fractional GPU memory limits"
                    ) from e

                visible_ids = list(range(len(self.allocation.gpu_ids)))
                for visible_id in visible_ids:
                    total_mb = int(
                        gpu_info[self.allocation.gpu_ids[visible_id]].memoryTotal
                    )
                    tf.config.set_logical_device_configuration(
                        gpus[visible_id],
                        [tf.config.LogicalDeviceConfiguration(
                            memory_limit=int(total_mb * frac)
                        )]
                    )
        except Exception:
            super().unapply_current()
            raise

    def unapply_current(self) -> None:
        if not self._applied:
            return

        try:
            try:
                import tensorflow as tf
                tf.keras.backend.clear_session(free_memory=True)
            except Exception:
                pass
        finally:
            super().unapply_current()
