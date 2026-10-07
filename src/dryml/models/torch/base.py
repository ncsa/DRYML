from __future__ import annotations

import math
import os
import weakref
from dataclasses import replace

from dryml.core.factory import FactorySpec
from dryml.core.backend import Backend
from dryml.core.object import Serializable
from dryml.core.repo import manage_repo
from dryml.core.tensor_spec import TensorSpec, iter_specs, map_spec_tree, match_input_batch
from dryml.core.utils.general import maybe_call_method, validate_class
from dryml.core.utils.recurse import map_leaf_groups, map_leaves
from dryml.models import Model as BaseModel
from dryml.models import TrainFunction as BaseTrainFunction
from dryml.models.experiment import _notify_host_observers
from dryml.models.progress import TrainingProgress, metric_value
from dryml.models.train_spec import (
    _EarlyStoppingState,
    _retained_early_stop_completed,
    _update_early_stopping,
    _validate_early_stopping_config,
)
from dryml.models.utils import (
    finite_dataset_len,
    record_train_update,
    require_bounded_safe_points,
    require_supervised_dataset,
    TrainingPreparation,
    validate_training_callbacks,
)
from dryml.methods import ImplementationSelectionError, MethodError, traits


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
        return (math.prod(int(dimension) for dimension in shape),)
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


def _require_mean_torch_loss(loss) -> None:
    """Reject Torch losses that cannot truthfully report a mean contribution."""

    if getattr(loss, "reduction", None) != "mean":
        raise ValueError(
            "Torch training accounting requires a loss with reduction='mean'."
        )


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

    native_backend = "torch"

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

    def _runtime_selection_facts(self, args, kwargs):
        """Select Torch's legacy eager wrapper path without value probing.

        Spec-driven preparation still declares and applies required data-boundary
        handoffs before this raw model body is reached.
        """

        del args, kwargs
        return Backend.torch, None

    @traits(backend="torch")
    def raw_call(self, x, *args, **kwargs):
        """Invoke the raw module when direct-call batching intent is unknown."""

        return self._call_raw(x, *args, **kwargs)

    @traits(backend="torch", batch_mode="batched")
    def batched_call(self, x, *args, **kwargs):
        """Invoke one selected already-batched module input without adaptation."""

        return self._call_raw(x, *args, **kwargs)

    @traits(backend="torch", batch_mode="element")
    def element_call(self, x, *args, **kwargs):
        """Invoke an element directly when no supplied spec selected adaptation."""

        return self._call_raw(x, *args, **kwargs)

    def find_implementation(self, input_spec=None, *additional_input_specs, backend=None, batch_mode=None,
                            output_spec=None):
        """Select a model call and attach explicit element batch adaptation.

        Args:
            input_spec: Optional normalized first-input specification retained
                for selected-call validation and element adaptation.
            *additional_input_specs: Unsupported because wrapped Models are unary.
            backend: Optional required backend value or closed string spelling.
            batch_mode: Optional required element/batched value or spelling.
            output_spec: Optional retained raw-result contract.

        Returns:
            A selected callable that moves element inputs to the selected Torch
            device, adds one batch axis, and removes it from the model output.

        Raises:
            ImplementationSelectionError: If constraints are malformed,
                conflicting, unsupported, or select no unique implementation.

        Side Effects:
            Registers the optional Torch tensor adapter, then binds a local
            selected callable without changing Method preparation state.
        """

        # Selected-call validation must recognize module outputs in a following
        # Pipe child. This optional backend registration stays in the selected
        # Torch model path rather than a lightweight package import.
        import dryml.torch

        if additional_input_specs:
            raise ImplementationSelectionError("conflict")
        implementation = super().find_implementation(
            input_spec,
            backend=backend,
            batch_mode=batch_mode,
            output_spec=output_spec,
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

    def infer_output_spec(self, input_spec, *additional_input_specs):
        """Infer Torch output metadata from exactly one logical input spec.

        Args:
            input_spec: Normalized Torch model input specification.
            *additional_input_specs: Unsupported additional inputs.

        Returns:
            The configured output specification or pure built-in module metadata
            translated to the input's logical batch representation.

        Raises:
            TypeError: If additional input specifications are supplied.
            NotImplementedError: If the module lacks supported static metadata or
                has no configured output specification.

        Side Effects:
            Imports Torch only to inspect recognized module metadata; it never
            invokes the module or allocates an input tensor.
        """

        if additional_input_specs:
            raise TypeError("Torch Model accepts exactly one input specification.")
        if self.output_spec is not None:
            return map_spec_tree(super().infer_output_spec(input_spec), lambda spec: TensorSpec(
                spec.dtype, shape=spec.shape, batch=spec.batch, backend="torch",
                layout=spec.layout, axis_names=spec.axis_names, batch_axis_name=spec.batch_axis_name,
            ))

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
    """Train a Torch model with retained update accounting and recovery.

    DRYML callbacks are preflighted before mutable training work. Each successful
    mean-reduced loss update records actual examples and the next batch before a
    callback runs; a retained epoch target distinguishes resume from a fresh
    later invocation.
    """

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
        verbose: int = 1,
    ):
        if epochs < 0:
            raise ValueError("epochs must be non-negative.")

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
        self.verbose = verbose

    __dryml_retired_constructor_parameters__ = (
        "batch_size", "num_examples", "shuffle", "shuffle_seed",
        "shuffle_buffer_size", "x_path", "y_path",
    )

    supports_observers = True

    def _validate_observer_session(self, observer_session) -> None:
        """Require host-callable telemetry before retained training starts."""

        observer_session.require_callables("Torch")

    def __call__(self, exp, *, callbacks=(), observer_session=None):
        """Train one Experiment with the PyTorch optimizer loop.

        Args:
            exp: Experiment providing model, data, state, and optional
                optimizer, loss, and metric capabilities.
            callbacks: Truthful post-update DRYML safe-point callbacks.
            observer_session: Private Experiment-owned invocation telemetry
                session, or ``None``.

        Returns:
            Per-batch scalar loss values in training order.

        Raises:
            ValueError: If the data is empty or the model exposes no trainable
                PyTorch parameters, the loss is not mean-reduced, or retained
                deterministic recovery cannot reach its invocation target.
            KeyError: If the model graph lacks a retained runtime binding.
            TypeError: If supplied DRYML callbacks are invalid.

        Side Effects:
            Updates model parameters, optimizer state, metrics, progress output,
            and ``exp.state``. Dataset backend preparation completes before the
            native forward/backward body; graph traversal uses a temporary
            explicit Repo only for trainable-parameter collection.
        """
        import torch

        callbacks = validate_training_callbacks(callbacks)
        if observer_session is not None:
            observer_session.require_callables("Torch")
        loss_fn = self._make_loss(torch, exp)
        _require_mean_torch_loss(loss_fn)
        self._begin_training_preparation_generation()
        train_data = exp.train_data
        require_supervised_dataset(train_data, batched=True)
        require_bounded_safe_points(train_data, callbacks)
        train_xy = train_data
        self.training_preparation = TrainingPreparation.from_specs(
            self, train_xy.spec[0], train_xy.spec[1], "torch"
        )
        prepared_train = train_xy.prepare()

        val_xy = None
        self.validation_preparation = None
        if exp.val_data is not None:
            val_data = exp.val_data
            require_supervised_dataset(val_data, batched=True)
            val_xy = val_data
            self.validation_preparation = TrainingPreparation.from_specs(
                self, val_xy.spec[0], val_xy.spec[1], "torch"
            )
            prepared_val = val_xy.prepare()
        else:
            prepared_val = None

        device = _resolve_device(torch)
        if hasattr(exp.model, "to_device"):
            exp.model.to_device(device)
        exp.model.prep_train()

        optimizer = self._make_optimizer(torch, exp.model, exp)
        metrics = self._metric_objects(exp)
        losses = []
        steps = 0
        steps_per_epoch = finite_dataset_len(train_data)
        start_epoch = exp.state.epoch
        fresh_target = exp.state.target_epoch is None
        initial_step = exp.state.step
        target_epoch = exp.state.begin_invocation(self.epochs)
        if self._completed_training_behavior_stop(exp):
            exp.model.prep_eval()
            exp.state.finish_invocation(target_epoch, accept_shortened=True)
            return []
        total_steps = None if steps_per_epoch is None else int(steps_per_epoch) * self.epochs
        progress = TrainingProgress(total=total_steps, verbose=self.verbose, desc="Torch training")

        try:
            stopped = self._finish_pending_epoch(
                torch, exp, prepared_val, loss_fn, metrics, device, progress, target_epoch
            )
            for epoch in range(start_epoch, target_epoch):
                if stopped:
                    break
                for metric in metrics:
                    _reset_metric(metric)
                metric_totals = {}
                metric_counts = {}
                epoch_loss = 0.0
                epoch_steps = 0

                from dryml.torch.training_data import iter_training_batches

                cursor = iter_training_batches(prepared_train, self.training_preparation)
                resume_batch = exp.state.next_batch if epoch == start_epoch else 0
                if resume_batch:
                    cursor.skip(resume_batch)
                try:
                    for x, y in cursor:
                        batch_index = exp.state.next_batch
                        examples = train_xy.examples_in((x, y))
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
                        complete_epoch = (
                            steps_per_epoch is not None
                            and exp.state.next_batch + 1 == steps_per_epoch
                        )
                        completed_metrics = None
                        if complete_epoch:
                            completed_metrics = {
                                "loss": (epoch_loss + loss_float) / (epoch_steps + 1)
                            }
                            completed_metrics.update(_metric_results(metrics))
                            for name, total in metric_totals.items():
                                completed_metrics.setdefault(name, total / metric_counts[name])
                        record_train_update(
                            exp,
                            x,
                            loss_float,
                            batched=True,
                            examples=examples,
                            complete_epoch=complete_epoch,
                            epoch_metrics=completed_metrics,
                            callbacks=callbacks,
                        )
                        losses.append(loss_float)
                        epoch_loss += loss_float
                        epoch_steps += 1
                        steps += 1

                        step_metrics = {"loss": loss_float}
                        step_metrics.update(batch_metrics)
                        step_metrics.update(_metric_results(metrics))
                        progress.update(1, step_metrics)
                        _notify_host_observers(observer_session, {
                            "event": "train_batch_end",
                            "epoch": epoch,
                            "batch": batch_index,
                            "step": exp.state.step,
                            "examples_seen": exp.state.examples_seen,
                            **step_metrics,
                        })
                finally:
                    cursor.close()

                if epoch_steps == 0 and resume_batch == 0:
                    continue

                # A resumed suffix cannot truthfully represent a full epoch mean.
                epoch_metrics = {"loss": epoch_loss / epoch_steps} if epoch_steps and not resume_batch else {}
                if not resume_batch:
                    epoch_metrics.update(_metric_results(metrics))
                    for name, total in metric_totals.items():
                        epoch_metrics.setdefault(name, total / metric_counts[name])

                stopped = self._finish_epoch(
                    torch,
                    exp,
                    epoch,
                    epoch_metrics,
                    prepared_val,
                    loss_fn,
                    metrics,
                    device,
                    progress,
                    target_epoch,
                )
                _notify_host_observers(observer_session, {
                    "event": "epoch_end",
                    "epoch": epoch,
                    "step": exp.state.step,
                    "examples_seen": exp.state.examples_seen,
                    "stopped": stopped,
                    **epoch_metrics,
                })
        except BaseException:
            if fresh_target:
                try:
                    exp.state.abandon_new_invocation(
                        target_epoch,
                        initial_epoch=start_epoch,
                        initial_step=initial_step,
                    )
                except ValueError:
                    pass
            raise
        finally:
            progress.close()
            exp.model.prep_eval()

        if (
            steps == 0
            and target_epoch > start_epoch
            and exp.state.epoch < target_epoch
            and not stopped
        ):
            if fresh_target:
                exp.state.abandon_new_invocation(
                    target_epoch,
                    initial_epoch=start_epoch,
                    initial_step=initial_step,
                )
            raise ValueError("Cannot train on an empty dataset.")
        if exp.state.epoch < target_epoch and not stopped:
            raise ValueError("Torch training ended before its retained invocation target.")
        exp.state.finish_invocation(target_epoch, accept_shortened=stopped)

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

    def _finish_pending_epoch(self, torch, exp, val_data, loss_fn, metrics, device, progress, target_epoch):
        """Run a retained validation/progress postlude before later updates."""

        epoch = exp.state.pending_epoch_postlude
        if epoch is None:
            return False
        epoch_metrics = dict(exp.state.pending_epoch_metrics or {})
        if exp.state.pending_epoch_postlude_phase == "start":
            if val_data is not None:
                val_metrics = self._evaluate(torch, exp.model, val_data, loss_fn, metrics, device=device)
                epoch_metrics.update({f"val_{name}": value for name, value in val_metrics.items()})
            exp.state.advance_epoch_postlude(epoch, "progress", metrics=epoch_metrics)
        if exp.state.pending_epoch_postlude_phase == "progress":
            stopped = self._finish_training_behavior_epoch(
                exp, epoch, exp.state.pending_epoch_metrics or epoch_metrics
            )
            progress.epoch_end(epoch + 1, epochs=target_epoch, metrics=epoch_metrics)
            exp.state.finish_epoch_postlude(epoch)
            return stopped
        return False

    def _finish_epoch(self, torch, exp, epoch, epoch_metrics, val_data, loss_fn, metrics, device, progress, target_epoch):
        """Complete one Torch epoch postlude without replaying its final update."""

        if exp.state.epoch == epoch:
            exp.state.finish_epoch(postlude_pending=True)
        if exp.state.pending_epoch_postlude == epoch:
            if exp.state.pending_epoch_postlude_phase == "start":
                if val_data is not None:
                    val_metrics = self._evaluate(torch, exp.model, val_data, loss_fn, metrics, device=device)
                    epoch_metrics.update({f"val_{name}": value for name, value in val_metrics.items()})
                exp.state.advance_epoch_postlude(epoch, "progress", metrics=epoch_metrics)
            if exp.state.pending_epoch_postlude_phase == "progress":
                stopped = self._finish_training_behavior_epoch(
                    exp, epoch, exp.state.pending_epoch_metrics or epoch_metrics
                )
                progress.epoch_end(
                    epoch + 1,
                    epochs=target_epoch,
                    metrics=exp.state.pending_epoch_metrics or epoch_metrics,
                )
                exp.state.finish_epoch_postlude(epoch)
                return stopped
            return False
        if val_data is not None:
            val_metrics = self._evaluate(torch, exp.model, val_data, loss_fn, metrics, device=device)
            epoch_metrics.update({f"val_{name}": value for name, value in val_metrics.items()})
        progress.epoch_end(epoch + 1, epochs=target_epoch, metrics=epoch_metrics)
        return self._finish_training_behavior_epoch(exp, epoch, epoch_metrics)

    def _finish_training_behavior_epoch(self, exp, epoch, metrics):
        """Run saved completed-epoch behavior; ordinary Training has none."""

        del exp, epoch, metrics
        return False

    def _completed_training_behavior_stop(self, exp) -> bool:
        """Return whether saved behavior already completed a shortened target."""

        del exp
        return False

    def _evaluate(self, torch, model, val_data, loss_fn, metrics, *, device):
        for metric in metrics:
            _reset_metric(metric)
        metric_totals = {}
        metric_counts = {}
        total_loss = 0.0
        total_examples = 0
        from dryml.torch.training_data import iter_training_batches

        cursor = iter_training_batches(val_data, self.validation_preparation)
        try:
            with torch.no_grad():
                for x, y in cursor:
                    examples = val_data.dataset.examples_in((x, y))
                    x = _tree_to_torch(x, torch, device=device)
                    y = _tree_to_torch(y, torch, device=device)
                    y_pred = model(x)
                    loss_value = loss_fn(y_pred, y)
                    total_loss += float(metric_value(loss_value)) * examples
                    total_examples += examples
                    batch_metrics = self._update_metrics(metrics, y_pred, y)
                    for name, value in batch_metrics.items():
                        metric_totals[name] = metric_totals.get(name, 0.0) + value * examples
                        metric_counts[name] = metric_counts.get(name, 0) + examples
        finally:
            cursor.close()

        if total_examples == 0:
            return {}

        results = {"loss": total_loss / total_examples}
        results.update(_metric_results(metrics))
        for name, total in metric_totals.items():
            results.setdefault(name, total / metric_counts[name])
        return results


class EarlyStoppingTraining(Training, Serializable):
    """Train Torch with recoverable completed-epoch early stopping.

    Args:
        monitor: Completed-epoch metric name, such as ``"loss"`` or
            ``"val_loss"``.
        patience: Complete non-improving epochs tolerated before stopping.
        mode: ``"min"`` for decreasing metrics or ``"max"`` for increasing.
        min_delta: Required nonnegative absolute improvement.
        restore_best_weights: Restore only Model parameters and buffers from the
            best completed epoch. Optimizer state remains at the stopping epoch.
        **kwargs: Arguments accepted by :class:`Training`.

    Raises:
        TypeError: If configuration types are invalid.
        ValueError: If configuration values or a completed monitor value are
            invalid or missing.

    Side Effects:
        Retains the decision and a bounded best ``state_dict`` snapshot as this
        TrainFunction's saved state. Optional restoration occurs once before the
        terminal Experiment checkpoint.
    """

    def __init__(
        self,
        *,
        monitor: str = "val_loss",
        patience: int = 3,
        mode: str = "min",
        min_delta: float = 0.0,
        restore_best_weights: bool = True,
        **kwargs,
    ):
        monitor, patience, mode, min_delta, restore_best_weights = (
            _validate_early_stopping_config(
                monitor=monitor,
                patience=patience,
                mode=mode,
                min_delta=min_delta,
                restore_best_weights=restore_best_weights,
            )
        )
        super().__init__(**kwargs)
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.early_stopping = _EarlyStoppingState()

    def __call__(self, exp, *, callbacks=(), observer_session=None):
        """Run or resume one retained early-stopping invocation.

        Args:
            exp: Experiment providing Torch Model, Datasets, and TrainState.
            callbacks: Truthful post-update DRYML safe-point callbacks.
            observer_session: Private Experiment-owned invocation telemetry
                session, or ``None``.

        Returns:
            Per-batch losses accepted before the full or shortened target.

        Raises:
            TypeError: If callbacks or native inputs violate Training contracts.
            ValueError: If recovery, Dataset, or completed monitor facts are
                invalid.

        Side Effects:
            Updates Model, Optimizer, Experiment, and this TrainFunction's saved
            continuation, and may restore the best Model state before return.
        """

        if exp.state.target_epoch is None:
            self.early_stopping.reset()
        return super().__call__(
            exp, callbacks=callbacks, observer_session=observer_session,
        )

    def _finish_training_behavior_epoch(self, exp, epoch, metrics):
        def capture():
            return {
                name: value.detach().cpu().clone()
                for name, value in exp.model.obj.state_dict().items()
            }

        return _update_early_stopping(
            self.early_stopping,
            epoch=epoch,
            metrics=metrics,
            monitor=self.monitor,
            patience=self.patience,
            mode=self.mode,
            min_delta=self.min_delta,
            capture_best=capture,
            restore_best=exp.model.obj.load_state_dict,
            restore_best_weights=self.restore_best_weights,
        )

    def _completed_training_behavior_stop(self, exp) -> bool:
        """Recognize a retained stop after its normalized postlude completed."""

        return _retained_early_stop_completed(self.early_stopping, exp.state)

    def save_state_to_dir_imp(self, dest_dir: str, *, codec: str) -> None:
        """Persist the retained decision and detached best Model snapshot.

        Args:
            dest_dir: Empty Store-owned local-state directory.
            codec: Opaque selected codec, accepted for hook compatibility.

        Side Effects:
            Writes one Torch continuation payload inside ``dest_dir``.
        """

        del codec
        import torch

        torch.save(
            self.early_stopping.to_payload(),
            os.path.join(dest_dir, "early-stopping.pth"),
        )

    def restore_state_from_dir_imp(self, src_dir: str, *, codec: str) -> None:
        """Validate and restore this TrainFunction's saved continuation.

        Args:
            src_dir: Store-owned directory containing the continuation payload.
            codec: Opaque selected codec, accepted for hook compatibility.

        Raises:
            ValueError: If retained decision facts are malformed.

        Side Effects:
            Replaces ``early_stopping`` only after payload decoding succeeds.
        """

        del codec
        import torch

        path = os.path.join(src_dir, "early-stopping.pth")
        self.early_stopping = _EarlyStoppingState.from_payload(
            torch.load(path, map_location="cpu", weights_only=False)
        )


class ModelWrapper(Model):
    pass


class Sequential(Model):
    """PyTorch Sequential model constructed from explicit layer factories.

    Args:
        layer_defs: A list or tuple of :class:`~dryml.core.FactorySpec` values
            resolved in ``torch.nn`` when this model is constructed.
        output_spec: Optional explicit DRYML output specification.

    Raises:
        TypeError: If ``layer_defs`` or any of its elements is not an explicit
            FactorySpec, or a built layer is not a PyTorch Module.

    Side Effects:
        Imports PyTorch and constructs the declared modules.
    """

    def __init__(self, layer_defs=(), output_spec=None):
        """Construct the PyTorch Sequential backend object from explicit factories.

        Args:
            layer_defs: A list or tuple containing only FactorySpec values.
            output_spec: Optional explicit DRYML output specification.

        Raises:
            TypeError: If layer declarations are not explicit factories or a
                constructed object is not a PyTorch Module.

        Side Effects:
            Imports PyTorch and constructs every validated layer.
        """
        import torch

        if not isinstance(layer_defs, (list, tuple)) or not all(
            isinstance(layer_def, FactorySpec) for layer_def in layer_defs
        ):
            raise TypeError(
                "Sequential layer definitions must be a list or tuple of explicit "
                "FactorySpec values. Use F(\"Linear\", 3, 8) or FactorySpec(...)."
            )
        self.layer_defs = tuple(layer_defs)
        self.device = None
        layers = [
            layer_def.build(
                namespace=torch.nn,
                instance_type=torch.nn.Module,
            )
            for layer_def in self.layer_defs
        ]

        self.obj = torch.nn.Sequential(*layers)
        self.module = self.obj
        self.mdl = self.obj
        self.output_spec = output_spec


__all__ = [
    "EarlyStoppingTraining",
    "Model",
    "ModelWrapper",
    "Optimizer",
    "Sequential",
    "Training",
    "TrainFunction",
    "Wrapper",
]
