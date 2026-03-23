from __future__ import annotations

from ..context_tracker import ContextBootstrapError
from ..plain.context import PlainComputeContext


class TFComputeContext(PlainComputeContext):
    name = "tf"

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

        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            return

        # After CUDA_VISIBLE_DEVICES, these are renumbered 0..N-1 from the worker's view.
        # Fractional limits only make sense if specific gpu/<id> fractions were requested.
        fractional = [
            self.allocation.assigned[key]
            for key in self.allocation.gpus
            if self.allocation.assigned[key] < 1.0
        ]
        if fractional:
            if len(set(fractional)) != 1:
                raise ContextBootstrapError(
                    "TF child_setup currently only supports one shared fraction "
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

            # visible GPUs are renumbered after CUDA_VISIBLE_DEVICES
            visible_ids = list(range(len(self.allocation.gpu_ids)))
            for visible_id in visible_ids:
                total_mb = int(gpu_info[self.allocation.gpu_ids[visible_id]].memoryTotal)
                tf.config.set_logical_device_configuration(
                    gpus[visible_id],
                    [tf.config.LogicalDeviceConfiguration(
                        memory_limit=int(total_mb * frac)
                    )]
                )

    def child_teardown(self) -> None:
        try:
            import tensorflow as tf
            tf.keras.backend.clear_session(free_memory=True)
        except Exception:
            pass
