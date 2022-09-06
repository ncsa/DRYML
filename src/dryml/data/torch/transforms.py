from dryml.dry_config import DryMeta
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


class TorchActivation(DryTrainable):
    @DryMeta.collect_args
    @DryMeta.collect_kwargs
    def __init__(self, cls, *args, **kwargs):
        self.cls = cls
        self.args = args
        self.kwargs = kwargs
        self.act = None

        self.train_state = DryTrainable.trained

    def train(self, *args, train_spec=None, **kwargs):
        pass

    def compute_prepare_imp(self):
        self.act = self.cls(*self.args, **self.kwargs)

    def compute_cleanup_imp(self):
        del self.act
        self.act = None

    def eval(self, data: DryData, *args, **kwargs):
        return data.apply_X(lambda X: self.act.forward(X))
