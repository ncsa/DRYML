from dryml.models import DryTrainable
from dryml.data import DryData
from dryml.data.torch import TorchDataset


class TorchDevice(DryTrainable):
    def __init__(self, device='cpu'):
        self.device = device
        self.train_state = DryTrainable.trained

    def train(self, *args, train_spec=None, **kwargs):
        pass

    def eval(self, data: DryData, *args, **kwargs):
        if type(data) is not TorchDataset:
            raise TypeError("This trainable can only accept torch datasets.")

        return data.map_el(
            lambda el: el.to(self.device))
