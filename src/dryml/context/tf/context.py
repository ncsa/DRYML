import tensorflow as tf
from typing import Optional, Union
from dryml.context import ComputeContext
from dryml.context.context_tracker import ResourceRequest


class TFComputeContext(ComputeContext):
    def __init__(
            self,
            resource_request: Optional[Union[ResourceRequest, dict]] = {}):
        super().__init__(resource_request=resource_request)
        self.strategy = None
        self.strategy_scope = None

    def acquire_context(self):
        # Let parent handle allocation
        super().acquire_context()

        # Get allocated gpus
        alloc_gpus = self.allocation.gpus

        # check if we need to set memory limits on any of these gpus
        need_mem_limit = False
        for gpu_key in alloc_gpus:
            if self.allocation[gpu_key] < 1.:
                need_mem_limit = True
                break

        # If needed, get actual GPU information
        gpu_info = None
        if need_mem_limit:
            try:
                import GPUtil
                gpu_info = GPUtil.getGPUs()
            except ImportError as e:
                print("Need GPUtil available to set memory limits on gpus.")
                raise e

        def get_gpu_id(key):
            return int(key.split('/')[1])

        # Get ids of gpus assigned. We first need to remove other
        # gpus from being visible.
        visible_gpu_ids = list(map(get_gpu_id, alloc_gpus))
        tf_gpu_objs = tf.config.list_physical_devices(device_type='GPU')
        tf_gpu_objs_restricted = []
        for i in range(len(visible_gpu_ids)):
            tf_gpu_objs_restricted.append(tf_gpu_objs[visible_gpu_ids[i]])
        tf.config.set_visible_devices(tf_gpu_objs_restricted, 'GPU')

        if need_mem_limit:
            # We need to restrict memory for some gpus.
            restrict_gpu_keys = list(filter(
                lambda k: self.allocation[k] < 1.,
                alloc_gpus))

            # Create config objects
            for i in range(len(restrict_gpu_keys)):
                gpu_key = restrict_gpu_keys[i]
                gpu_id = get_gpu_id(gpu_key)
                target_mem = int(
                    gpu_info[gpu_id].memoryTotal*self.allocation[gpu_key])
                tf.config.set_logical_device_configuration(
                    tf_gpu_objs[gpu_id],
                    [tf.config.LogicalDeviceConfiguration(
                        memory_limit=target_mem)])

        logical_devices = tf.config.list_logical_devices('GPU')
        if len(logical_devices) > 1:
            # Get visible logical device names
            gpu_names = list(map(lambda device: device.name, logical_devices))

            # Create
            self.strategy = tf.distribute.MirroredStrategy(gpu_names)
            # Fix improper tear-down on program exit:
            # https://github.com/tensorflow/tensorflow/issues/50487
            import atexit
            atexit.register(
                self.strategy._extended._collective_ops._pool.close)

        if self.strategy is not None:
            self.strategy_scope = self.strategy.scope()
            self.strategy_scope.__enter__()

    def release_context(self):
        if self.strategy_scope is not None:
            self.strategy_scope.__exit__(None, None, None)
            self.strategy_scope = None
            self.strategy = None

        # Let parent handle releasing allocation
        super().release_context()
