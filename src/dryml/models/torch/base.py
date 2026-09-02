from __future__ import annotations

import os
import weakref
from dataclasses import replace

from dryml.core.factory import FactorySpec
from dryml.core.object import Serializable
from dryml.core.repo import manage_repo
from dryml.core.tensor_spec import TensorSpec, iter_specs, match_input_batch
from dryml.core.utils.general import maybe_call_method, validate_class
from dryml.core.utils.recurse import map_leaf_groups, map_leaves
from dryml.data import Batch, Map, Project, Select
from dryml.models import Model as BaseModel
from dryml.models import TrainFunction as BaseTrainFunction
from dryml.models.progress import TrainingProgress, metric_value
from dryml.models.utils import (
    advance_train_state,
    finite_dataset_len,
    prepare_training_data,
    validate_num_examples,
)
from dryml.methods import MethodError, traits


def _resolve_device(torch):
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _tree_to_torch(value, torch, *, device=None):
    def leaf_to_torch(leaf):
        if isinstance(leaf, torch.Tensor):
            return leaf.to(device) if device is not None else leaf
        tensor = torch.as_tensor(leaf)
        return tensor.to(device) if device is not None else tensor

    return map_leaves(value, leaf_to_torch)


def _tree_to_torch_model_batch(value, torch, input_spec, *, device=None):
    def leaf_to_torch(values):
        leaf, spec = values
        if not isinstance(spec, TensorSpec):
            raise TypeError(f"Expected TensorSpec leaves, got {type(spec).__name__}.")
        if isinstance(leaf, torch.Tensor):
            tensor = leaf.to(device) if device is not None else leaf
        else:
            tensor = torch.as_tensor(leaf)
            tensor = tensor.to(device) if device is not None else tensor
        return tensor if spec.batched else tensor.unsqueeze(0)

    return map_leaf_groups((value, input_spec), leaf_to_torch)


def _unbatch_tree(value):
    return map_leaves(value, lambda leaf: leaf[0])


def _torch_metadata_shape(torch, module, shape):
    """Infer selected built-in module output shape without executing a module."""

    if isinstance(module, torch.nn.Sequential):
        for child in module.children():
            shape = _torch_metadata_shape(torch, child, shape)
        return shape
    if isinstance(module, torch.nn.Flatten):
        if module.start_dim != 1 or module.end_dim not in (-1, len(shape)):
            raise NotImplementedError("Only standard batch-preserving torch Flatten is supported.")
        size = 1
        for dimension in shape:
            size *= int(dimension)
        return (size,)
    if isinstance(module, torch.nn.Linear):
        if not shape:
            raise NotImplementedError("torch Linear requires a known feature axis.")
        return (*shape[:-1], int(module.out_features))
    if isinstance(
        module,
        (
            torch.nn.ReLU,
            torch.nn.Sigmoid,
            torch.nn.Tanh,
            torch.nn.Identity,
            torch.nn.Dropout,
        ),
    ):
        return shape
    raise NotImplementedError(f"No pure shape metadata route exists for {type(module).__name__}.")


def _unwrap_backend_obj(obj):
    return obj.obj if hasattr(obj, "obj") else obj


def _normalize_list(value):
    if value is None:
        return ()
    if isinstance(value, dict):
        return tuple(value.values())
    if isinstance(value, (tuple, list)):
        return tuple(value)
    return (value,)


def _reset_metric(metric):
    reset = getattr(metric, "reset", None) or getattr(metric, "reset_state", None)
    if reset is not None:
        reset()


def _metric_name(metric):
    return getattr(metric, "name", type(metric).__name__)


def _update_metric(metric, y_pred, y):
    update = getattr(metric, "update", None) or getattr(metric, "update_state", None)
    if update is not None:
        update(y_pred, y)
        return None
    return metric(y_pred, y)


def _metric_results(metrics):
    out = {}
    for metric in metrics:
        compute = getattr(metric, "compute", None) or getattr(metric, "result", None)
        if compute is not None:
            out[_metric_name(metric)] = metric_value(compute())
    return out


def _collect_trainable_parameters(target, *, repo):
    """Collect graph trainables using supplied explicit Repo authority.

    Args:
        target: Live DRYML graph root whose trainable nodes are collected.
        repo: Bounded Repo authority used to traverse retained runtime bindings.

    Returns:
        A flat list of PyTorch trainable parameters in post-order graph order.

    Raises:
        KeyError: If a required runtime binding is unavailable from ``target``.
    """

    results = repo.apply_graph(
        target,
        lambda obj: maybe_call_method(
            obj,
            "trainable_parameters",
            "torch",
            default=(),
        ),
        missing="raise",
        order="post",
    )

    parameters = []
    for result in results.values():
        if result is not None:
            parameters.extend(result)
    return parameters


class Wrapper(Serializable):
    """Generic torch object wrapper exposing the backend object at ``.obj``."""

    def __init__(self, cls, *args, **kwargs):
        self.cls = validate_class(cls)
        self.args = args
        self.kwargs = kwargs
        self.obj = self.cls(*args, **kwargs)

    def save_state_to_dir_imp(self, dest_dir: str, *, codec: str) -> None:
        if hasattr(self.obj, "state_dict"):
            import torch

            torch.save(self.obj.state_dict(), os.path.join(dest_dir, "state.pth"))

    def restore_state_from_dir_imp(self, src_dir: str, *, codec: str) -> None:
        if hasattr(self.obj, "load_state_dict"):
            import torch

            state_path = os.path.join(src_dir, "state.pth")
            self.obj.load_state_dict(torch.load(state_path, map_location="cpu"))


class Optimizer(Serializable):
    """Torch optimizer bound to trainable parameters from a live model graph.

    Parameter graph traversal receives temporary explicit Repo authority only
    while collecting retained runtime bindings. The optimizer retains the
    resulting PyTorch parameter objects and its own mutable optimizer state.
    """

    def __init__(self, cls, *args, target, **kwargs):
        """Construct a PyTorch optimizer for a target model graph.

        Args:
            cls: PyTorch optimizer class.
            *args: Positional arguments after the collected parameters.
            target: Live DRYML graph root that exposes trainable parameters.
            **kwargs: Keyword arguments passed to ``cls``.

        Raises:
            ValueError: If ``target`` exposes no trainable parameters.
            KeyError: If ``target`` lacks a retained runtime binding.

        Side Effects:
            Creates ``self.obj`` and temporarily installs managed Repo
            authority only while traversing ``target``.
        """
        self.cls = validate_class(cls)
        self.args = args
        self.kwargs = kwargs
        self.target = target
        with manage_repo() as repo:
            parameters = _collect_trainable_parameters(target, repo=repo)
        if not parameters:
            raise ValueError("Torch Optimizer target exposes no trainable parameters.")
        self.obj = self.cls(parameters, *args, **kwargs)

    def save_state_to_dir_imp(self, dest_dir: str, *, codec: str) -> None:
        import torch

        torch.save(self.obj.state_dict(), os.path.join(dest_dir, "optimizer.pth"))

    def restore_state_from_dir_imp(self, src_dir: str, *, codec: str) -> None:
        import torch

        state_path = os.path.join(src_dir, "optimizer.pth")
        if os.path.exists(state_path):
            self.obj.load_state_dict(torch.load(state_path, map_location="cpu"))


class Model(BaseModel, Serializable):
    """Wrapper around a torch.nn.Module-style class.

    Direct calls retain ordinary tensor conversion and raw module invocation.
    Spec-selected element calls author the former one-item batch adaptation;
    selected batched calls invoke the module without a second batch axis. Only
    recognized static module metadata can infer an output spec.
    """

    def __init__(self, cls, *args, output_spec=None, **kwargs):
        self.cls = validate_class(cls)
        self.module_args = args
        self.module_kwargs = kwargs
        self.device = None
        self.obj = self.cls(*args, **kwargs)
        self.module = self.obj
        self.mdl = self.obj
        self.output_spec = output_spec

    def _call_raw(self, x, *args, **kwargs):
        import torch

        device = _resolve_device(torch)
        x = _tree_to_torch(x, torch, device=device)
        return self.obj(x, *args, **kwargs)

    @traits()
    def raw_call(self, x, *args, **kwargs):
        """Invoke the raw module when direct-call batching intent is unknown."""

        return self._call_raw(x, *args, **kwargs)

    @traits(batch_mode="batched")
    def batched_call(self, x, *args, **kwargs):
        """Invoke one selected already-batched module input without adaptation."""

        return self._call_raw(x, *args, **kwargs)

    @traits(batch_mode="element")
    def element_call(self, x, *args, **kwargs):
        """Invoke an element directly when no supplied spec selected adaptation."""

        return self._call_raw(x, *args, **kwargs)

    def find_implementation(self, input_spec=None, *, backend=None, batch_mode=None):
        """Select a model call and attach element adaptation from ``input_spec``."""

        # Selected-call validation must recognize module outputs in a following
        # Pipe child. This optional backend registration stays in the selected
        # Torch model path rather than a lightweight package import.
        import dryml.torch

        implementation = super().find_implementation(
            input_spec,
            backend=backend,
            batch_mode=batch_mode,
        )
        return self._specialize_implementation(implementation, input_spec)

    def _prepare_implementation(self, input_spec, *, backend, batch_mode):
        """Build a learning-time selected model call without shared state mutation."""

        import dryml.torch

        implementation = super()._prepare_implementation(
            input_spec,
            backend=backend,
            batch_mode=batch_mode,
        )
        return self._specialize_implementation(implementation, input_spec)

    def _specialize_implementation(self, implementation, input_spec):
        """Attach the selected element's explicit one-item batch adaptation."""

        if implementation.name != "element_call" or input_spec is None:
            return implementation
        receiver_ref = weakref.ref(self)

        def invoke_element(x, *args, **kwargs):
            import torch

            receiver = receiver_ref()
            if receiver is None:
                raise MethodError("The selected Torch Model is no longer live.")
            device = _resolve_device(torch)
            batched = _tree_to_torch_model_batch(x, torch, input_spec, device=device)
            return _unbatch_tree(receiver.obj(batched, *args, **kwargs))

        return replace(implementation, _invoker=invoke_element)

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

    def save_state_to_dir_imp(self, dest_dir: str, *, codec: str) -> None:
        import torch

        torch.save(self.obj.state_dict(), os.path.join(dest_dir, "state.pth"))

    def restore_state_from_dir_imp(self, src_dir: str, *, codec: str) -> None:
        import torch

        state_path = os.path.join(src_dir, "state.pth")
        self.obj.load_state_dict(torch.load(state_path, map_location=self.device or "cpu"))

    def infer_output_spec(self, input_spec):
        if self.output_spec is not None:
            return super().infer_output_spec(input_spec)

        import torch
        if not isinstance(input_spec, TensorSpec) or input_spec.shape is None:
            raise NotImplementedError(
                f"Cannot infer output spec for {type(self).__name__} without executing the model; "
                "pass output_spec explicitly."
            )
        try:
            shape = _torch_metadata_shape(torch, self.obj, input_spec.shape)
        except (AttributeError, NotImplementedError, TypeError, ValueError) as error:
            raise NotImplementedError(
                f"Cannot infer output spec for {type(self).__name__} without executing the model; "
                "pass output_spec explicitly."
            ) from error
        dtype = next(iter_specs(input_spec)).dtype
        return match_input_batch(
            TensorSpec(dtype, shape=shape, backend="torch"),
            input_spec,
        )


class TrainFunction(BaseTrainFunction):
    pass


class Training(TrainFunction):
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
        metrics=(),
        epochs: int = 1,
        batch_size: int | None = 32,
        x_path=0,
        y_path=1,
        num_examples: int | None = None,
        shuffle: bool = False,
        shuffle_seed=None,
        shuffle_buffer_size: int | None = None,
        verbose: int = 1,
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
        self.metrics = _normalize_list(metrics)
        self.epochs = epochs
        self.batch_size = batch_size
        self.x_path = x_path
        self.y_path = y_path
        self.num_examples = num_examples
        self.shuffle = shuffle
        self.shuffle_seed = shuffle_seed
        self.shuffle_buffer_size = shuffle_buffer_size
        self.verbose = verbose

    def __call__(self, exp):
        """Train one Experiment with the PyTorch optimizer loop.

        Args:
            exp: Experiment providing model, data, state, and optional
                optimizer, loss, and metric capabilities.

        Returns:
            Per-batch scalar loss values in training order.

        Raises:
            ValueError: If the data is empty or the model exposes no trainable
                PyTorch parameters.
            KeyError: If the model graph lacks a retained runtime binding.

        Side Effects:
            Updates model parameters, optimizer state, metrics, progress output,
            and ``exp.state``. Graph traversal uses a temporary explicit Repo
            only for the duration of trainable-parameter collection.
        """
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
        train_xy = Map(train_data, Project(Select(self.x_path), Select(self.y_path)))

        val_xy = None
        if exp.val_data is not None:
            val_data = prepare_training_data(exp.val_data)
            if self.batch_size is not None:
                val_data = Batch(val_data, self.batch_size)
            val_xy = Map(val_data, Project(Select(self.x_path), Select(self.y_path)))

        device = _resolve_device(torch)
        if hasattr(exp.model, "to_device"):
            exp.model.to_device(device)
        exp.model.prep_train()

        optimizer = self._make_optimizer(torch, exp.model, exp)
        loss_fn = self._make_loss(torch, exp)
        metrics = self._metric_objects(exp)
        losses = []
        steps = 0
        steps_per_epoch = finite_dataset_len(train_data)
        total_steps = None if steps_per_epoch is None else int(steps_per_epoch) * self.epochs
        progress = TrainingProgress(total=total_steps, verbose=self.verbose, desc="Torch training")

        try:
            for epoch in range(self.epochs):
                for metric in metrics:
                    _reset_metric(metric)
                metric_totals = {}
                metric_counts = {}
                epoch_loss = 0.0
                epoch_steps = 0

                for x, y in train_xy:
                    x = _tree_to_torch(x, torch, device=device)
                    y = _tree_to_torch(y, torch, device=device)

                    optimizer.zero_grad()
                    y_pred = exp.model(x)
                    loss_value = loss_fn(y_pred, y)
                    loss_value.backward()
                    optimizer.step()

                    batch_metrics = self._update_metrics(metrics, y_pred, y)
                    for name, value in batch_metrics.items():
                        metric_totals[name] = metric_totals.get(name, 0.0) + value
                        metric_counts[name] = metric_counts.get(name, 0) + 1

                    loss_float = float(metric_value(loss_value))
                    losses.append(loss_float)
                    epoch_loss += loss_float
                    epoch_steps += 1
                    steps += 1

                    step_metrics = {"loss": loss_float}
                    step_metrics.update(batch_metrics)
                    step_metrics.update(_metric_results(metrics))
                    progress.update(1, step_metrics)

                if epoch_steps == 0:
                    continue

                epoch_metrics = {"loss": epoch_loss / epoch_steps}
                epoch_metrics.update(_metric_results(metrics))
                for name, total in metric_totals.items():
                    epoch_metrics.setdefault(name, total / metric_counts[name])

                if val_xy is not None:
                    val_metrics = self._evaluate(torch, exp.model, val_xy, loss_fn, metrics, device=device)
                    epoch_metrics.update({f"val_{name}": value for name, value in val_metrics.items()})

                progress.epoch_end(epoch + 1, epochs=self.epochs, metrics=epoch_metrics)
        finally:
            progress.close()
            exp.model.prep_eval()

        if steps == 0 and self.epochs > 0:
            raise ValueError("Cannot train on an empty dataset.")

        advance_train_state(exp, epochs=self.epochs, steps=steps)
        return losses

    def _capability(self, exp, name, default=None):
        return getattr(exp, "capabilities", {}).get(name, default)

    def _optimizer(self, exp):
        return self.optimizer if self.optimizer is not None else self._capability(exp, "optimizer")

    def _loss(self, exp):
        return self.loss if self.loss is not None else self._capability(exp, "loss")

    def _metrics(self, exp):
        if self.metrics:
            return self.metrics
        capability_metrics = self._capability(exp, "metrics")
        if capability_metrics is not None:
            return _normalize_list(capability_metrics)
        return _normalize_list(getattr(exp, "metrics", ()))

    def _make_optimizer(self, torch, model, exp):
        """Return an optimizer configured for one model's trainable graph.

        Args:
            torch: Imported PyTorch module.
            model: Live DRYML model graph root.
            exp: Experiment supplying optional optimizer capabilities.

        Returns:
            A PyTorch optimizer ready to update ``model``.

        Raises:
            ValueError: If ``model`` exposes no trainable parameters.
            KeyError: If ``model`` lacks a retained runtime binding.

        Side Effects:
            Temporarily installs managed Repo authority only while collecting
            trainable parameters.
        """
        optimizer = _unwrap_backend_obj(self._optimizer(exp))
        with manage_repo() as repo:
            parameters = _collect_trainable_parameters(model, repo=repo)
        if not parameters:
            raise ValueError("Torch model graph exposes no trainable parameters.")
        if optimizer is not None:
            if isinstance(optimizer, type):
                return validate_class(optimizer)(
                    parameters,
                    *self.optimizer_args,
                    **self.optimizer_kwargs,
                )
            return optimizer

        optimizer_cls = self.optimizer_cls or torch.optim.Adam
        return validate_class(optimizer_cls)(
            parameters,
            *self.optimizer_args,
            **self.optimizer_kwargs,
        )

    def _make_loss(self, torch, exp):
        loss = _unwrap_backend_obj(self._loss(exp))
        if loss is not None:
            if isinstance(loss, type):
                return validate_class(loss)(*self.loss_args, **self.loss_kwargs)
            return loss

        loss_cls = self.loss_cls or torch.nn.MSELoss
        return validate_class(loss_cls)(*self.loss_args, **self.loss_kwargs)

    def _metric_objects(self, exp):
        return [_unwrap_backend_obj(metric) for metric in self._metrics(exp)]

    def _update_metrics(self, metrics, y_pred, y):
        out = {}
        for metric in metrics:
            value = _update_metric(metric, y_pred, y)
            if value is not None:
                out[_metric_name(metric)] = float(metric_value(value))
        return out

    def _evaluate(self, torch, model, val_xy, loss_fn, metrics, *, device):
        for metric in metrics:
            _reset_metric(metric)
        metric_totals = {}
        metric_counts = {}
        total_loss = 0.0
        steps = 0
        with torch.no_grad():
            for x, y in val_xy:
                x = _tree_to_torch(x, torch, device=device)
                y = _tree_to_torch(y, torch, device=device)
                y_pred = model(x)
                loss_value = loss_fn(y_pred, y)
                total_loss += float(metric_value(loss_value))
                steps += 1
                batch_metrics = self._update_metrics(metrics, y_pred, y)
                for name, value in batch_metrics.items():
                    metric_totals[name] = metric_totals.get(name, 0.0) + value
                    metric_counts[name] = metric_counts.get(name, 0) + 1

        if steps == 0:
            return {}

        results = {"loss": total_loss / steps}
        results.update(_metric_results(metrics))
        for name, total in metric_totals.items():
            results.setdefault(name, total / metric_counts[name])
        return results


class ModelWrapper(Model):
    pass


class Sequential(Model):
    @classmethod
    def __prepare_args__(cls, layer_defs=(), output_spec=None):
        args = (FactorySpec.coerce_many(layer_defs),)
        kwargs = {}
        if output_spec is not None:
            kwargs["output_spec"] = output_spec
        return args, kwargs

    def __init__(self, layer_defs=(), output_spec=None):
        import torch

        self.layer_defs = tuple(layer_defs)
        self.device = None
        layers = []
        for layer_def in self.layer_defs:
            if isinstance(layer_def, FactorySpec):
                layers.append(
                    layer_def.build(
                        namespace=torch.nn,
                        instance_type=torch.nn.Module,
                    )
                )
                continue

            raise TypeError(
                "Sequential layer definitions must be FactorySpec values. "
                "Tuple and string shorthands should be normalized by __prepare_args__."
            )

        self.obj = torch.nn.Sequential(*layers)
        self.module = self.obj
        self.mdl = self.obj
        self.output_spec = output_spec


__all__ = [
    "Model",
    "ModelWrapper",
    "Optimizer",
    "Sequential",
    "Training",
    "TrainFunction",
    "Wrapper",
]
