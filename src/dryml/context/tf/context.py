import tensorflow as tf
from dryml.context import ComputeContext, ResourcesUnavailableError


class TFComputeContext(ComputeContext):
    def __init__(self, num_cpus=1, num_gpus=0):
        self.num_cpus = num_cpus
        self.num_gpus = num_gpus

    def acquire_context(self):
        # Get a list of available GPUs
        gpus = tf.config.list_physical_devices(device_type='GPU')
        if len(gpus) < self.num_gpus:
            raise ResourcesUnavailableError("Not enough gpus available!")

        cpus = tf.config.list_physical_devices(device_type='CPU')
        if len(cpus) < self.num_cpus:
            raise ResourcesUnavailableError("Not enough cpus available!")

        # Constrict tf to use just the specified cpus/gpus
        tf.config.set_visible_devices(
            cpus[:self.num_cpus]+gpus[:self.num_gpus])

        # Let parent set the global context
        super().acquire_context()
