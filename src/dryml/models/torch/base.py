from __future__ import annotations

import importlib.util
import inspect
import json
import os
import pickle
import random
import sys
import tempfile
from pathlib import Path

import numpy as np

from dryml.core.factory import FactorySpec
from dryml.core.object import Serializable
from dryml.core.repo import get_default_repo
from dryml.core.tensor_spec import TensorSpec, fake_from_spec_tree, maybe_unbatch_output_spec, spec_tree_is_batched
from dryml.core.utils.general import maybe_call_method, revision_path, validate_class
from dryml.core.utils.recurse import map_leaf_groups, map_leaves
from dryml.data import Batch, Map, Project, Select
from dryml.managed import (
    ControlRequest,
    ManagedCapabilityError,
    OperationResult,
    current_operation_context,
)
from dryml.models import Model as BaseModel
from dryml.models import TrainFunction as BaseTrainFunction
from dryml.models import TrainCapability, TrainResumeMode
from dryml.models.progress import TrainingProgress, metric_value
from dryml.models.train_spec import TRAIN_CHECKPOINT_SCHEMA, TrainState
from dryml.models.utils import (
    advance_train_state,
    finite_dataset_len,
    prepare_training_data,
    validate_num_examples,
)
from dryml.torch.tensor_spec import as_tensor_spec as torch_as_tensor_spec


def _torch():
    from dryml.runtime import import_configured_framework

    return import_configured_framework("torch")


def _resolve_device(torch):
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _definition_arguments(cls, definition):
    signature = inspect.signature(cls.__init__)
    bound = signature.bind_partial(None, *definition.args, **dict(definition.kwargs))
    bound.apply_defaults()
    return {name: value for name, value in bound.arguments.items() if name != "self"}


def _definition_is(value, cls):
    ref = getattr(value, "cls", None)
    return all(
        (
            getattr(ref, "module", None) == cls.__module__,
            getattr(ref, "qualname", None) == cls.__qualname__,
        )
    )


def _import_module(value):
    return getattr(value, "module", None)


def _managed_world_size():
    size = 1
    for name in ("DRYML_WORLD_SIZE", "WORLD_SIZE"):
        value = os.environ.get(name)
        if value is None:
            continue
        try:
            size = int(value)
        except ValueError as exc:
            raise ManagedCapabilityError(
                f"managed Torch training received invalid {name}"
            ) from exc
        if size < 1:
            raise ManagedCapabilityError(
                f"managed Torch training received invalid {name}"
            )
        break
    if size != 1:
        return size
    torch = sys.modules.get("torch")
    distributed = getattr(torch, "distributed", None)
    if all(
        (
            distributed is not None,
            distributed is not None and distributed.is_available(),
            distributed is not None and distributed.is_initialized(),
        )
    ):
        return distributed.get_world_size()
    return size


def _operation_context_or_none():
    try:
        return current_operation_context()
    except RuntimeError:
        return None


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


def _collect_trainable_parameters(target):
    repo = get_default_repo()
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

    def save_state_to_dir_imp(self, dest_dir: str, revision: str | None = None):
        if hasattr(self.obj, "state_dict"):
            torch = _torch()

            torch.save(self.obj.state_dict(), revision_path("state", "pth", dest_dir, revision=revision))

    def restore_state_from_dir_imp(self, src_dir: str, revision: str | None):
        if hasattr(self.obj, "load_state_dict"):
            torch = _torch()

            state_path = revision_path("state", "pth", src_dir, revision=revision)
            self.obj.load_state_dict(torch.load(state_path, map_location="cpu"))


class Optimizer(Serializable):
    """Torch optimizer spec with runtime state bound when model parameters exist."""

    def __init__(self, cls, *args, target, **kwargs):
        self.cls = validate_class(cls)
        self.args = args
        self.kwargs = kwargs
        self.target = target
        parameters = _collect_trainable_parameters(target)
        if not parameters:
            raise ValueError("Torch Optimizer target exposes no trainable parameters.")
        self.obj = self.cls(parameters, *args, **kwargs)

    def save_state_to_dir_imp(self, dest_dir: str, revision: str | None = None):
        torch = _torch()

        torch.save(self.obj.state_dict(), revision_path("optimizer", "pth", dest_dir, revision=revision))

    def restore_state_from_dir_imp(self, src_dir: str, revision: str | None):
        torch = _torch()

        state_path = revision_path("optimizer", "pth", src_dir, revision=revision)
        if os.path.exists(state_path):
            self.obj.load_state_dict(torch.load(state_path, map_location="cpu"))


class Model(BaseModel, Serializable):
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
        torch = _torch()

        device = _resolve_device(torch)
        x = _tree_to_torch(x, torch, device=device)
        return self.obj(x, *args, **kwargs)

    def bind_first(self, first_value, *, input_spec=None):
        if input_spec is None or spec_tree_is_batched(input_spec):
            return self, self(first_value)

        torch = _torch()

        device = _resolve_device(torch)

        def bound_model(x):
            batched = _tree_to_torch_model_batch(x, torch, input_spec, device=device)
            return _unbatch_tree(self.obj(batched))

        return bound_model, bound_model(first_value)

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
        torch = _torch()

        device = _resolve_device(torch)
        self.to_device(device)
        if hasattr(self.obj, "train"):
            self.obj.train(True)

    def prep_eval(self):
        torch = _torch()

        device = _resolve_device(torch)
        self.to_device(device)
        if hasattr(self.obj, "eval"):
            self.obj.eval()

    def save_state_to_dir_imp(self, dest_dir: str, revision: str | None = None):
        torch = _torch()

        torch.save(self.obj.state_dict(), revision_path("state", "pth", dest_dir, revision=revision))

    def restore_state_from_dir_imp(self, src_dir: str, revision: str | None):
        torch = _torch()

        state_path = revision_path("state", "pth", src_dir, revision=revision)
        self.obj.load_state_dict(torch.load(state_path, map_location=self.device or "cpu"))

    def infer_output_spec(self, input_spec):
        if self.output_spec is not None:
            return super().infer_output_spec(input_spec)

        torch = _torch()

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


class Training(TrainFunction):
    """Train a Torch model from an Experiment's supervised train data.

    Single-process managed execution is exactly resumable at completed epoch
    boundaries for the built-in, unshuffled loop. Checkpoints contain the full
    DRYML model state, optimizer state, epoch/step progress, and Python, NumPy,
    Torch CPU, and applicable CUDA RNG state. Custom loops, explicit stateful
    loss objects, trainer metrics, shuffle, and multi-rank publication do not
    receive this guarantee.
    """

    @classmethod
    def resume_capability(cls, definition=None):
        """Report exact epoch-boundary capability for a compatible definition."""

        if definition is None:
            return TrainCapability.none("Torch trainer definition is required")
        if cls is not Training:
            return TrainCapability.none(
                "custom Torch training loops have no exact checkpoint contract"
            )
        if importlib.util.find_spec("torch") is None:
            return TrainCapability.none("Torch is unavailable in the selected environment")
        config = _definition_arguments(cls, definition)
        unsupported = []
        if config["shuffle"]:
            unsupported.append("shuffle cursor/RNG")
        if config["metrics"]:
            unsupported.append("trainer metric state")
        if config["loss"] is not None:
            unsupported.append("explicit loss object state")
        loss_cls = config["loss_cls"]
        if loss_cls is not None and not (_import_module(loss_cls) or "").startswith(
            "torch.nn.modules.loss"
        ):
            unsupported.append("custom loss state")
        optimizer = config["optimizer"]
        if optimizer is not None and not _definition_is(optimizer, Optimizer):
            unsupported.append("opaque optimizer binding/state")
        optimizer_cls = config["optimizer_cls"]
        if optimizer_cls is not None and not (_import_module(optimizer_cls) or "").startswith(
            "torch.optim"
        ):
            unsupported.append("custom optimizer state")
        if unsupported:
            return TrainCapability.none(
                f"Torch epoch-boundary resume cannot checkpoint {', '.join(unsupported)}"
            )
        return TrainCapability.exact(
            "Torch model, optimizer, progress, RNG, and full-epoch input boundary are checkpointed",
            early_completion=True,
        )

    @classmethod
    def managed_pipeline_capability(cls, definition, capabilities):
        """Narrow exact resume and reject unsupported managed publication."""

        if _managed_world_size() != 1:
            raise ManagedCapabilityError(
                "multi-rank managed Torch checkpoint publication is unsupported"
            )
        capability = cls.resume_capability(definition)
        if capability.mode is not TrainResumeMode.EXACT:
            return capability
        configured = tuple(
            name
            for name in ("optimizer", "loss", "metrics")
            if capabilities.get(name) is not None
        )
        if configured:
            return TrainCapability.none(
                f"Torch Experiment capability state is not checkpointed: {', '.join(configured)}"
            )
        return capability

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
        torch = _torch()
        context = _operation_context_or_none()

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
        resume = exp.resume_payload if context is not None else None
        initial_epoch = exp.state.epoch
        target_epoch = (
            resume["descriptor"]["target_epoch"]
            if resume is not None
            else initial_epoch + self.epochs
        )
        if resume is not None:
            self._restore_backend_checkpoint(torch, optimizer, resume)
        losses = []
        steps = 0
        steps_per_epoch = finite_dataset_len(train_data)
        remaining_epochs = target_epoch - initial_epoch
        if remaining_epochs < 0:
            raise RuntimeError("Torch training checkpoint target precedes its progress")
        total_steps = (
            None
            if steps_per_epoch is None
            else int(steps_per_epoch) * remaining_epochs
        )
        progress = TrainingProgress(total=total_steps, verbose=self.verbose, desc="Torch training")
        early_completed = False

        try:
            for epoch in range(initial_epoch, target_epoch):
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

                progress.epoch_end(
                    epoch - initial_epoch + 1,
                    epochs=remaining_epochs,
                    metrics=epoch_metrics,
                )
                if context is not None:
                    exp.state.epoch = epoch + 1
                    exp.state.step += epoch_steps
                    context.progress(
                        exp.state.epoch,
                        total=target_epoch,
                        message="Torch training epochs",
                        metrics=epoch_metrics,
                    )
                    control = self._managed_safe_point(
                        context,
                        lambda: self._checkpoint_training(
                            torch,
                            exp,
                            optimizer,
                            target_epoch=target_epoch,
                        ),
                    )
                    if control is ControlRequest.GRACEFUL_STOP:
                        early_completed = True
                        break
        finally:
            progress.close()
            exp.model.prep_eval()

        if steps == 0 and remaining_epochs > 0:
            raise ValueError("Cannot train on an empty dataset.")

        if context is not None and initial_epoch == target_epoch:
            context.progress(
                initial_epoch,
                total=target_epoch,
                message="Torch training epochs",
            )
            self._managed_safe_point(
                context,
                lambda: self._checkpoint_training(
                    torch,
                    exp,
                    optimizer,
                    target_epoch=target_epoch,
                ),
            )
        if context is not None:
            return OperationResult(early_completed=early_completed)
        advance_train_state(exp, epochs=remaining_epochs, steps=steps)
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
        optimizer = _unwrap_backend_obj(self._optimizer(exp))
        parameters = _collect_trainable_parameters(model)
        if not parameters:
            raise ValueError("Torch model graph exposes no trainable parameters.")
        if optimizer is not None:
            configured = self._optimizer(exp)
            if isinstance(configured, Optimizer) and _operation_context_or_none() is not None:
                return configured.cls(
                    parameters,
                    *configured.args,
                    **configured.kwargs,
                )
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

    def restore_checkpoint(self, exp, checkpoint_root):
        """Validate a Torch checkpoint and restore a fresh model's progress."""

        torch = _torch()
        root = Path(checkpoint_root)
        try:
            descriptor = json.loads(
                (root / "torch-train.json").read_text(encoding="utf-8")
            )
        except Exception as exc:
            raise RuntimeError("Torch training checkpoint descriptor is unreadable") from exc
        required = {
            "schema",
            "schema_version",
            "model_cdef_id",
            "trainer_cdef_id",
            "torch_version",
            "completed_epoch",
            "completed_step",
            "target_epoch",
        }
        if not isinstance(descriptor, dict) or set(descriptor) != required:
            raise RuntimeError("Torch training checkpoint descriptor is malformed")
        expected = {
            "schema": TRAIN_CHECKPOINT_SCHEMA,
            "schema_version": 1,
            "model_cdef_id": exp.model.definition.stable_hash(),
            "trainer_cdef_id": self.definition.stable_hash(),
            "torch_version": torch.__version__,
        }
        for name, value in expected.items():
            if descriptor.get(name) != value:
                if name == "torch_version":
                    raise ManagedCapabilityError(
                        "Torch checkpoint version does not match the active environment"
                    )
                raise ManagedCapabilityError(
                    "Torch training checkpoint is incompatible with the selected pipeline"
                )
        try:
            state = pickle.loads((root / "train-state.pkl").read_bytes())
        except Exception as exc:
            raise RuntimeError("Torch training progress checkpoint is unreadable") from exc
        if not isinstance(state, TrainState):
            raise RuntimeError("Torch training progress checkpoint is malformed")
        progress = (state.epoch, state.step)
        expected_progress = (
            descriptor["completed_epoch"],
            descriptor["completed_step"],
        )
        if progress != expected_progress or state.epoch > descriptor["target_epoch"]:
            raise RuntimeError("Torch training progress checkpoint is inconsistent")
        exp.model.restore_state_from_dir(str(root / "model"))
        exp.state = state
        return {"root": root, "descriptor": descriptor}

    def _checkpoint_training(self, torch, exp, optimizer, *, target_epoch):
        context = current_operation_context()
        with tempfile.TemporaryDirectory(prefix="dryml-torch-train-checkpoint-") as temp:
            root = Path(temp)
            model_root = root / "model"
            model_root.mkdir()
            exp.model.save_state_to_dir(str(model_root))
            torch.save(optimizer.state_dict(), root / "optimizer.pth")
            rng_state = {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else None
                ),
            }
            (root / "rng-state.pkl").write_bytes(
                pickle.dumps(rng_state, protocol=5)
            )
            (root / "train-state.pkl").write_bytes(
                pickle.dumps(exp.state, protocol=5)
            )
            descriptor = {
                "schema": TRAIN_CHECKPOINT_SCHEMA,
                "schema_version": 1,
                "model_cdef_id": exp.model.definition.stable_hash(),
                "trainer_cdef_id": self.definition.stable_hash(),
                "torch_version": torch.__version__,
                "completed_epoch": exp.state.epoch,
                "completed_step": exp.state.step,
                "target_epoch": target_epoch,
            }
            (root / "torch-train.json").write_text(
                json.dumps(descriptor, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            for path in sorted(item for item in root.rglob("*") if item.is_file()):
                context.write_checkpoint(
                    path.relative_to(root).as_posix(),
                    _file_chunks(path),
                )
        return context.commit_checkpoint(
            metadata={
                "backend": "torch",
                "epoch": exp.state.epoch,
                "step": exp.state.step,
            }
        )

    def _restore_backend_checkpoint(self, torch, optimizer, resume):
        root = resume["root"]
        try:
            optimizer_state = torch.load(
                root / "optimizer.pth",
                map_location="cpu",
                weights_only=False,
            )
            rng_state = pickle.loads((root / "rng-state.pkl").read_bytes())
            if not isinstance(rng_state, dict) or set(rng_state) != {
                "python",
                "numpy",
                "torch",
                "cuda",
            }:
                raise ValueError("invalid RNG state")
            optimizer.load_state_dict(optimizer_state)
            random.setstate(rng_state["python"])
            np.random.set_state(rng_state["numpy"])
            torch.set_rng_state(rng_state["torch"])
            if rng_state["cuda"] is not None:
                if not torch.cuda.is_available():
                    raise ManagedCapabilityError(
                        "Torch checkpoint requires CUDA RNG state in the active environment"
                    )
                torch.cuda.set_rng_state_all(rng_state["cuda"])
        except ManagedCapabilityError:
            raise
        except Exception as exc:
            raise RuntimeError("Torch optimizer or RNG checkpoint is unreadable") from exc

    @staticmethod
    def _managed_safe_point(context, checkpoint):
        checkpoint_capable = context.checkpoint_schema == TRAIN_CHECKPOINT_SCHEMA
        before = context.checkpoint_head
        control = context.safe_point(
            checkpoint=checkpoint if checkpoint_capable else None
        )
        if checkpoint_capable and context.checkpoint_head == before:
            checkpoint()
        return control


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
        torch = _torch()

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


def _file_chunks(path: Path, size: int = 1024 * 1024):
    with path.open("rb") as handle:
        while chunk := handle.read(size):
            yield chunk


__all__ = [
    "Model",
    "ModelWrapper",
    "Optimizer",
    "Sequential",
    "Training",
    "TrainFunction",
    "Wrapper",
]
