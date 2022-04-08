import tensorflow as tf
from dryml.context import ComputeContext, ResourcesUnavailableError


class TFComputeContext(ComputeContext):
    def __init__(self, num_cpus=1, num_gpus=-1):
        super().__init__(num_cpus=num_cpus, num_gpus=num_gpus)
        self.strategy = None
        self.strategy_scope = None

    def acquire_context(self):
        # Get a list of available GPUs
        gpus = tf.config.list_physical_devices(device_type='GPU')
        if self.num_gpus >= 0:
            if len(gpus) < self.num_gpus:
                raise ResourcesUnavailableError("Not enough gpus available!")
        elif self.num_gpus < 0:
            self.num_gpus = len(gpus)
        if len(gpus) > 1 and self.num_gpus > 1:
            tf.config.set_visible_devices(gpus[:self.num_gpus], device_type="GPU")
            prefix_len = len('/physical_device:')
            gpu_names = list(map(
                lambda p: p.name[prefix_len:], gpus[:self.num_gpus]))
            self.strategy = tf.distribute.MirroredStrategy(gpu_names)
            # Fix improper tear-down on program exit:
            # https://github.com/tensorflow/tensorflow/issues/50487
            import atexit
            atexit.register(self.strategy._extended._collective_ops._pool.close)

        cpus = tf.config.list_physical_devices(device_type='CPU')
        if len(cpus) < self.num_cpus:
            raise ResourcesUnavailableError("Not enough cpus available!")

        if self.num_cpus > 1:
            raise NotImplementedError(
                "Don't support more than one cpu simultaneously yet")

        if self.strategy is not None:
            self.strategy_scope = self.strategy.scope()
            self.strategy_scope.__enter__()

        # Let parent set the global context
        super().acquire_context()

    def release_context(self):
        # Let parent remove the global context
        super().release_context()

        if self.strategy_scope is not None:
            self.strategy_scope.__exit__(None, None, None)
            self.strategy_scope = None
            self.strategy = None
