import torch
from typing import Optional, Union
from dryml.context import ComputeContext
from dryml.context.context_tracker import ResourceRequest


class TorchComputeContext(ComputeContext):
    def __init__(
            self,
            resource_request: Optional[Union[ResourceRequest, dict]] = {}):
        super().__init__(resource_request=resource_request)

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

        if need_mem_limit:
            print("WARNING, currently there is no way to limit memory used by torch.")

    def compute_devices(self):
        device_list = list(map(
            lambda n: n.replace('gpu/', 'cuda:'),
            self.allocation.gpus))
        if len(device_list) == 0:
            device_list = ['cpu']
        return device_list

    def release_context(self):
        # Let parent handle releasing allocation
        super().release_context()
