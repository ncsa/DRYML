from dryml.config import Meta
from dryml.object import Object
from dryml.data import Dataset
from dryml.models.trainable import Trainable as BaseTrainable
from dryml.models.torch.base import Wrapper
from dryml.models.torch.base import Model as TorchModel
from dryml.models.torch.base import TrainFunction as TorchTrainFunction
from dryml.models.torch.base import Trainable as TorchTrainable
from dryml.context import context
import zipfile
import torch
import tqdm


class Model(TorchModel):
    def __call__(self, *args, **kwargs):
        return self.mdl.forward(*args, **kwargs)

    def compute_cleanup_imp(self):
        del self.mdl
        self.mdl = None

    def load_compute_imp(self, file: zipfile.ZipFile) -> bool:
        try:
            with file.open('state.pth', 'r') as f:
                self.mdl.load_state_dict(torch.load(f))
            return True
        except Exception:
            return False

    def save_compute_imp(self, file: zipfile.ZipFile) -> bool:
        try:
            with file.open('state.pth', 'w') as f:
                torch.save(self.mdl.state_dict(), f)
            return True
        except Exception:
            return False

    def prep_eval(self):
        devs = context().get_torch_devices()
        self.mdl.to(devs[0])
        self.mdl.train(False)

    def prep_train(self):
        devs = context().get_torch_devices()
        self.mdl.to(devs[0])
        self.mdl.train(True)


class TrainFunction(TorchTrainFunction):
    pass


class ModelWrapper(Model):
    @Meta.collect_args
    @Meta.collect_kwargs
    def __init__(self, cls, *args, **kwargs):
        self.cls = cls
        self.args = args
        self.kwargs = kwargs
        self.mdl = None

    def compute_prepare_imp(self):
        self.mdl = self.cls(*self.args, *self.kwargs)


class Sequential(Model):
    def __init__(self, layer_defs=[]):
        self.layer_defs = layer_defs
        self.mdl = None

    def compute_prepare_imp(self):
        # create_layers
        layers = []
        for layer in self.layer_defs:
            if type(layer[0]) is not type:
                raise TypeError(
                    "First element of a layer definition should be a type")
            layers.append(layer[0](*layer[1], **layer[2]))

        self.mdl = torch.nn.Sequential(
            *layers)


class TorchOptimizer(Object):
    @Meta.collect_args
    @Meta.collect_kwargs
    def __init__(self, cls, model: Model, *args, **kwargs):
        if type(cls) is not type:
            raise TypeError("first argument must be a class!")
        self.cls = cls
        self.model = model
        self.args = args
        self.kwargs = kwargs
        self.opt = None

    def compute_prepare_imp(self):
        self.opt = self.cls(
            self.model.mdl.parameters(),
            *self.args,
            **self.kwargs)

    def load_compute_imp(self, file: zipfile.ZipFile) -> bool:
        try:
            with file.open('state.pth', 'r') as f:
                self.opt.load_state_dict(torch.load(f))
            return True
        except Exception:
            return False

    def save_compute_imp(self, file: zipfile.ZipFile) -> bool:
        try:
            with file.open('state.pth', 'w') as f:
                torch.save(self.opt.state_dict(), f)
            return True
        except Exception:
            return False

    def compute_cleanup_imp(self):
        del self.opt
        self.opt = None


class TorchScheduler(Object):
    @Meta.collect_args
    @Meta.collect_kwargs
    def __init__(self, cls, optimizer: TorchOptimizer, *args, **kwargs):
        if type(cls) is not type:
            raise TypeError("first argument must be a class!")
        self.cls = cls
        self.optimizer = optimizer
        self.args = args
        self.kwargs = kwargs
        self.sched = None

    def compute_prepare_imp(self):
        self.sched = self.cls(
            self.optimizer.opt,
            *self.args,
            *self.kwargs)

    def load_compute_imp(self, file: zipfile.ZipFile) -> bool:
        try:
            with file.open('state.pth', 'r') as f:
                self.sched.load_state_dict(torch.load(f))
            return True
        except Exception:
            return False

    def save_compute_imp(self, file: zipfile.ZipFile) -> bool:
        try:
            with file.open('state.pth', 'w') as f:
                torch.save(self.sched.state_dict(), f)
            return True
        except Exception:
            return False

    def compute_cleanup_imp(self):
        del self.sched
        self.sched = None


class Trainable(TorchTrainable):
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
        self.train_state = BaseTrainable.trained

    def prep_train(self):
        self.model.prep_train()

    def prep_eval(self):
        self.model.prep_eval()

    def eval(self, data: Dataset, *args, eval_batch_size=32, **kwargs):
        # Move variables to same device as model
        devs = context().get_torch_devices()
        if data.batched:
            # We can execute the method directly on the data
            return data.torch() \
                       .map_el(lambda el: el.to(devs[0])) \
                       .apply_X(
                           func=lambda X: self.model(X, *args, **kwargs))
        else:
            # We first need to batch the data, then unbatch to leave
            # The dataset character unchanged.
            return data.torch() \
                       .batch(batch_size=eval_batch_size) \
                       .map_el(lambda el: el.to(devs[0])) \
                       .apply_X(
                            func=lambda X: self.model(X, *args, **kwargs)) \
                       .unbatch()


class BasicTraining(TrainFunction):
    def __init__(
            self,
            optimizer: Wrapper = None,
            loss: Wrapper = None,
            epochs=1):
        self.optimizer = optimizer
        self.loss = loss
        self.epochs = epochs

    def __call__(
            self, trainable: Model, data: Dataset, train_spec=None,
            train_callbacks=[]):

        # Pop the epoch to resume from
        start_epoch = 0
        if train_spec is not None:
            start_epoch = train_spec.level_step()
        # Type checking training data, and converting if necessary
        batch_size = 32
        data = data.torch().batch(batch_size=batch_size)
        total_batches = data.count()
        # Move variables to same device as model
        devs = context().get_torch_devices()
        data = data.map_el(lambda el: el.to(devs[0]))
        # Check data is supervised.
        if not data.supervised:
            raise RuntimeError(
                f"{__class__} requires supervised data")

        optimizer = self.optimizer.opt
        loss = self.loss.obj
        model = trainable.model
        for i in range(start_epoch, self.epochs):
            running_loss = 0.
            num_batches = 0
            t_data = tqdm.tqdm(data, total=total_batches)
            for X, Y in t_data:
                optimizer.zero_grad()
                outputs = model(X)
                loss_val = loss(outputs, Y)
                loss_val.backward()
                optimizer.step()
                running_loss += loss_val.item()
                num_batches += 1
                av_loss = running_loss/(num_batches*batch_size)
                t_data.set_postfix(loss=av_loss)
            print(f"Epoch {i+1} - Average Loss: {av_loss}")


class LRBasicTraining(TrainFunction):
    def __init__(
            self,
            optimizer: Wrapper = None,
            loss: Wrapper = None,
            scheduler: Wrapper = None,
            epochs=1):
        self.optimizer = optimizer
        self.loss = loss
        self.epochs = epochs
        self.scheduler = scheduler
        self.training_loss = []

    def __call__(
            self, trainable: Model, data: Dataset, train_spec=None,
            train_callbacks=[]):

        # Pop the epoch to resume from
        start_epoch = 0
        if train_spec is not None:
            start_epoch = train_spec.level_step()
        # Type checking training data, and converting if necessary
        batch_size = 32
        data = data.torch().batch(batch_size=batch_size)
        total_batches = data.count()
        # Move variables to same device as model
        devs = context().get_torch_devices()
        data = data.map_el(lambda el: el.to(devs[0]))
        # Check data is supervised.
        if not data.supervised:
            raise RuntimeError(
                f"{__class__} requires supervised data")
        optimizer = self.optimizer.opt
        loss = self.loss.obj
        scheduler = self.scheduler.sched
        model = trainable.model
        for i in range(start_epoch, self.epochs):
            running_loss = 0.
            num_batches = 0
            t_data = tqdm.tqdm(data, total=total_batches)
            for X, Y in t_data:
                optimizer.zero_grad()
                outputs = model(X)
                loss_val = loss(outputs, Y)
                loss_val.backward()
                optimizer.step()
                running_loss += loss_val.item()
                num_batches += 1
                av_loss = running_loss/(num_batches*batch_size)
                t_data.set_postfix(loss=av_loss)
            scheduler.step(av_loss)
            self.training_loss.append(av_loss)
            print(f"Epoch {i+1} - Average Loss: {av_loss}")

        
        
