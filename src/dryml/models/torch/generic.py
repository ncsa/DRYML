from dryml.dry_config import DryMeta
from dryml.data import DryData
from dryml.models.torch.base import TorchObject
from dryml.models.torch.base import Model as BaseModel
from dryml.models.torch.base import TrainFunction as BaseTrainFunction
from dryml.models.torch.base import Trainable as BaseTrainable
from dryml.context import context
import zipfile
import torch


class Model(BaseModel):
    pass


class TrainFunction(BaseTrainFunction):
    pass


class Sequential(Model):
    def __init__(self, layer_defs = []):
        self.layer_defs = layer_defs
        self.mdl = None

    def compute_prepare_imp(self):
        # create_layers
        layers = []
        for layer in layer_defs:
            if type(layer[0]) is not type:
                raise TypeError("First element of a layer definition should be a type")
            layers.append(layer[0](*layer[1], **layer[2]))

        self.mdl = torch.nn.Sequential(
            layers)

    def load_compute_imp(self, file: zipfile.ZipFile) -> bool:
        try:
            with file.open('state.pth', 'r') as f:
                self.mdl.load_state_dict(torch.load(f))
            return True
        except:
            return False

    def save_compute_imp(self, file: zipfile.ZipFile) -> bool:
        try:
            with file.open('state.pth', 'w') as f:
                torch.save(self.mdl.state_dict(), f)
            return True
        except:
            return False

    def __call__(self, *args, **kwargs):
        return self.mdl.forward(*args, **kwargs)

    def compute_cleanup_imp(self):
        del self.mdl
        self.mdl = None

    def prep_eval(self):
        devs = context().get_torch_devices()
        self.mdl.to(devs[0])
        self.mdl.train(False)

    def prep_train(self):
        devs = context().get_torch_devices()
        self.mdl.to(devs[0])
        self.mdl.train(True)


class ModuleModel(Model):
    def __init__(self, model_obj: TorchObject):
        if type(cls) is not type:
            raise TypeError("first argument must be a class!")
        self.mdl = model_obj

    def __call__(self, *args, **kwargs):
        return self.mdl.obj.forward(*args, **kwargs)

    def prep_eval(self):
        devs = context().get_torch_devices()
        self.mdl.to(devs[0])
        self.mdl.obj.train(False)

    def prep_train(self):
        devs = context().get_torch_devices()
        self.mdl.to(devs[0])
        self.mdl.obj.train(True)


class Trainable(BaseTrainable):
    def __init__(
            self,
            model: Model = None,
            train_fn: TrainFunction = None):
        self.model = model
        self.train_fn = train_fn

    def train(
            self, data, train_spec=None, train_callbacks=[],
            metrics=[]):
        self.train_fn(
            self, data, train_spec=train_spec,
            train_callbacks=train_callbacks)
        self.train_state = DryTrainable.trained

    def prep_train(self):
        self.model.prep_train()

    def prep_eval(self):
        self.model.prep_eval()

    def eval(self, data: DryData, *args, eval_batch_size=32, **kwargs):
        if data.batched:
            # We can execute the method directly on the data
            return data.torch().apply_X(
                func=lambda X: self.model(X, *args, **kwargs))
        else:
            # We first need to batch the data, then unbatch to leave
            # The dataset character unchanged.
            return data.torch().batch(batch_size=eval_batch_size) \
                       .apply_X(
                            func=lambda X: self.model(X, *args, **kwargs)) \
                       .unbatch()


class BasicTraining(TrainFunction):
    def __init__(
            self,
            optimizer: TorchObject = None,
            loss: TorchObject = None,
            epochs=1):
        self.optimizer=optimizer
        self.loss = loss
        self.epochs=epochs

    def __call__(
            self, trainable: Model, data: DryData, train_spec=None,
            train_callbacks=[]):

        # Pop the epoch to resume from
        start_epoch = 0
        if train_spec is not None:
            start_epoch = train_spec.level_step()

        # Type checking training data, and converting if necessary
        data = data.torch()

        # Check data is supervised.
        if not data.supervised:
            raise RuntimeError(
                f"{__class__} requires supervised data")

        optimizer = self.optimizer.obj
        loss = self.loss.obj
        model = trainable.model

        for i in range(start_epoch, epochs):
            running_loss = 0.

            for X, Y in data:
                optimizer.zero_grad()

                outputs = trainable.model(X)
                loss_val = loss(outputs, Y)
                loss_val.backward()
                optimizer.step()

                running_loss += loss_val.item()
