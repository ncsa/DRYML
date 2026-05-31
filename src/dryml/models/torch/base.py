from __future__ import annotations

import os

from dryml.core2.object import Object
from dryml.core2.utils.general import revision_path, validate_class
from dryml.core2.utils.recurse import map_leaves
from dryml.data import Batch, iter_xy, maybe_unbatch_output_spec, fake_from_spec_tree
from dryml.models import Model as BaseModel
from dryml.models import TrainFunction as BaseTrainFunction
from dryml.models.utils import (
    advance_train_state,
    prepare_training_data,
    validate_num_examples,
)
from dryml.torch.tensor_spec import as_tensor_spec as torch_as_tensor_spec


def _resolve_device(torch):
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _tree_to_torch(value, torch, *, device=None):
    def leaf_to_torch(leaf):
        if isinstance(leaf, torch.Tensor):
            return leaf.to(device) if device is not None else leaf
        tensor = torch.as_tensor(leaf)
        return tensor.to(device) if device is not None else tensor

    return map_leaves(value, leaf_to_torch)


class Wrapper(Object):
    """Generic torch object wrapper exposing the backend object at ``.obj``."""

    def __init__(self, cls, *args, **kwargs):
        self.cls = validate_class(cls)
        self.args = args
        self.kwargs = kwargs
        self.obj = self.cls(*args, **kwargs)

    def save_state_to_dir_imp(self, dest_dir: str, revision: str | None = None):
        if hasattr(self.obj, "state_dict"):
            import torch

            torch.save(self.obj.state_dict(), revision_path("state", "pth", dest_dir, revision=revision))

    def restore_state_from_dir_imp(self, src_dir: str, revision: str | None):
        if hasattr(self.obj, "load_state_dict"):
            import torch

            state_path = revision_path("state", "pth", src_dir, revision=revision)
            self.obj.load_state_dict(torch.load(state_path, map_location="cpu"))


class Optimizer(Object):
    """Torch optimizer spec with runtime state bound when model parameters exist."""

    def __init__(self, cls, *args, target, **kwargs):
        self.cls = validate_class(cls)
        self.args = args
        self.kwargs = kwargs
        self.target = target
        parameters = list(target.trainable_parameters("torch"))
        if not parameters:
            raise ValueError("Torch Optimizer target exposes no trainable parameters.")
        self.obj = self.cls(parameters, *args, **kwargs)

    def save_state_to_dir_imp(self, dest_dir: str, revision: str | None = None):
        import torch

        torch.save(self.obj.state_dict(), revision_path("optimizer", "pth", dest_dir, revision=revision))

    def restore_state_from_dir_imp(self, src_dir: str, revision: str | None):
        import torch

        state_path = revision_path("optimizer", "pth", src_dir, revision=revision)
        if os.path.exists(state_path):
            self.obj.load_state_dict(torch.load(state_path, map_location="cpu"))


class Model(BaseModel):
    """Wrapper around a torch.nn.Module-style class."""

    def __init__(self, cls, *args, output_spec=None, **kwargs):
        self.cls = validate_class(cls)
        self.module_args = args
        self.module_kwargs = kwargs
        self.device = None
        self.obj = self.cls(*args, **kwargs)
        self.module = self.obj
        self.mdl = self.obj
        self.output_spec = output_spec

    def __call__(self, x, *args, **kwargs):
        import torch

        device = _resolve_device(torch)
        x = _tree_to_torch(x, torch, device=device)
        return self.obj(x, *args, **kwargs)

    def parameters(self):
        return self.trainable_parameters("torch")

    def trainable_parameters(self, backend: str | None = None):
        if backend not in (None, "torch"):
            return ()
        return self.obj.parameters()

    def to_device(self, device):
        self.device = str(device)
        if hasattr(self.module, "to"):
            self.obj.to(device)

    def prep_train(self):
        import torch

        device = _resolve_device(torch)
        self.to_device(device)
        if hasattr(self.obj, "train"):
            self.obj.train(True)

    def prep_eval(self):
        import torch

        device = _resolve_device(torch)
        self.to_device(device)
        if hasattr(self.obj, "eval"):
            self.obj.eval()

    def save_state_to_dir_imp(self, dest_dir: str, revision: str | None = None):
        import torch

        torch.save(self.obj.state_dict(), revision_path("state", "pth", dest_dir, revision=revision))

    def restore_state_from_dir_imp(self, src_dir: str, revision: str | None):
        import torch

        state_path = revision_path("state", "pth", src_dir, revision=revision)
        self.obj.load_state_dict(torch.load(state_path, map_location=self.device or "cpu"))

    def infer_output_spec(self, input_spec):
        if self.output_spec is not None:
            return super().infer_output_spec(input_spec)

        import torch

        was_training = bool(getattr(self.obj, "training", False))
        self.prep_eval()
        try:
            sample = _tree_to_torch(fake_from_spec_tree(input_spec), torch, device=_resolve_device(torch))
            with torch.no_grad():
                output = self.obj(sample)
            return maybe_unbatch_output_spec(torch_as_tensor_spec(output, batched=True), input_spec)
        finally:
            if was_training:
                self.prep_train()


class TrainFunction(BaseTrainFunction):
    pass


class BasicTraining(TrainFunction):
    """Train a torch model from an Experiment's supervised train_data."""

    def __init__(
        self,
        *,
        optimizer=None,
        optimizer_cls=None,
        optimizer_args=(),
        optimizer_kwargs=None,
        loss=None,
        loss_cls=None,
        loss_args=(),
        loss_kwargs=None,
        epochs: int = 1,
        batch_size: int | None = 32,
        x_path=0,
        y_path=1,
        num_examples: int | None = None,
        shuffle: bool = False,
        shuffle_seed=None,
        shuffle_buffer_size: int | None = None,
    ):
        if epochs < 0:
            raise ValueError("epochs must be non-negative.")
        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be positive or None.")
        validate_num_examples(num_examples)

        self.optimizer = optimizer
        self.optimizer_cls = optimizer_cls
        self.optimizer_args = tuple(optimizer_args)
        self.optimizer_kwargs = dict(optimizer_kwargs or {})
        self.loss = loss
        self.loss_cls = loss_cls
        self.loss_args = tuple(loss_args)
        self.loss_kwargs = dict(loss_kwargs or {})
        self.epochs = epochs
        self.batch_size = batch_size
        self.x_path = x_path
        self.y_path = y_path
        self.num_examples = num_examples
        self.shuffle = shuffle
        self.shuffle_seed = shuffle_seed
        self.shuffle_buffer_size = shuffle_buffer_size

    def __call__(self, exp):
        import torch

        train_data = prepare_training_data(
            exp.train_data,
            num_examples=self.num_examples,
            shuffle=self.shuffle,
            shuffle_seed=self.shuffle_seed,
            shuffle_buffer_size=self.shuffle_buffer_size,
        )
        if self.batch_size is not None:
            train_data = Batch(train_data, self.batch_size)

        device = _resolve_device(torch)
        if hasattr(exp.model, "to_device"):
            exp.model.to_device(device)
        exp.model.prep_train()

        optimizer = self._make_optimizer(torch, exp.model)
        loss_fn = self._make_loss(torch)
        losses = []
        steps = 0

        try:
            for _ in range(self.epochs):
                for x, y in iter_xy(
                    train_data,
                    x_path=self.x_path,
                    y_path=self.y_path,
                ):
                    x = _tree_to_torch(x, torch, device=device)
                    y = _tree_to_torch(y, torch, device=device)

                    optimizer.zero_grad()
                    y_pred = exp.model(x)
                    loss_value = loss_fn(y_pred, y)
                    loss_value.backward()
                    optimizer.step()

                    losses.append(float(loss_value.detach().cpu()))
                    steps += 1
        finally:
            exp.model.prep_eval()

        if steps == 0 and self.epochs > 0:
            raise ValueError("Cannot train on an empty dataset.")

        advance_train_state(exp, epochs=self.epochs, steps=steps)
        return losses

    def _make_optimizer(self, torch, model):
        optimizer = self.optimizer.obj if hasattr(self.optimizer, "obj") else self.optimizer
        if optimizer is not None:
            if isinstance(optimizer, type):
                return validate_class(optimizer)(
                    model.trainable_parameters("torch"),
                    *self.optimizer_args,
                    **self.optimizer_kwargs,
                )
            return optimizer

        optimizer_cls = self.optimizer_cls or torch.optim.Adam
        return validate_class(optimizer_cls)(
            model.trainable_parameters("torch"),
            *self.optimizer_args,
            **self.optimizer_kwargs,
        )

    def _make_loss(self, torch):
        loss = self.loss.obj if hasattr(self.loss, "obj") else self.loss
        if loss is not None:
            if isinstance(loss, type):
                return validate_class(loss)(*self.loss_args, **self.loss_kwargs)
            return loss

        loss_cls = self.loss_cls or torch.nn.MSELoss
        return validate_class(loss_cls)(*self.loss_args, **self.loss_kwargs)


class ModelWrapper(Model):
    pass


class Sequential(Model):
    def __init__(self, layer_defs=(), output_spec=None):
        import torch

        self.layer_defs = tuple(layer_defs)
        self.device = None
        layers = []
        for layer_def in self.layer_defs:
            if isinstance(layer_def, torch.nn.Module):
                layers.append(layer_def)
                continue

            if isinstance(layer_def, str):
                cls = getattr(torch.nn, layer_def)
                args = ()
                kwargs = {}
            elif isinstance(layer_def, type):
                cls = layer_def
                args = ()
                kwargs = {}
            elif len(layer_def) == 2:
                cls, kwargs = layer_def
                cls = getattr(torch.nn, cls) if isinstance(cls, str) else cls
                args = ()
            elif len(layer_def) == 3:
                cls, args, kwargs = layer_def
                cls = getattr(torch.nn, cls) if isinstance(cls, str) else cls
            else:
                raise ValueError(
                    "Layer definitions must be layer names, layer classes, layer instances, "
                    "(cls, kwargs), or (cls, args, kwargs)."
                )
            layers.append(validate_class(cls)(*args, **kwargs))

        self.obj = torch.nn.Sequential(*layers)
        self.module = self.obj
        self.mdl = self.obj
        self.output_spec = output_spec


__all__ = [
    "BasicTraining",
    "Model",
    "ModelWrapper",
    "Optimizer",
    "Sequential",
    "TrainFunction",
    "Wrapper",
]
