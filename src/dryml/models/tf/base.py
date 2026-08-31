from __future__ import annotations

import os
import shutil

from dryml.core.object import Serializable
from dryml.core.repo import get_default_repo
from dryml.core.tensor_spec import Dynamic, TensorSpec, fake_from_spec_tree, maybe_unbatch_output_spec, spec_tree_is_batched
from dryml.core.utils.general import maybe_call_method, validate_class
from dryml.core.utils.recurse import map_leaf_groups, map_leaves
from dryml.data import Batch, Map, Project, Select
from dryml.models import Model as BaseModel
from dryml.models import TrainFunction as BaseTrainFunction
from dryml.models.progress import TrainingProgress, metric_value
from dryml.models.utils import advance_train_state, finite_dataset_len, prepare_training_data, validate_num_examples
from dryml.tf.tensor_spec import as_tensor_spec as tf_as_tensor_spec
from dryml.tf.tensor_spec import output_signature as tf_output_signature


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


def _collect_trainable_parameters(target, *, repo=None):
    """Collect graph trainables using explicit or context-local Repo authority."""

    repo = repo or get_default_repo()
    if repo is None:
        raise RuntimeError("TensorFlow trainable-parameter collection requires an active Repo.")
    results = repo.apply_graph(
        target,
        lambda obj: maybe_call_method(
            obj,
            "trainable_parameters",
            "tf",
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


def _dims_to_keras_shape(shape):
    if shape is None:
        return None
    return tuple(None if dim is Dynamic else int(dim) for dim in shape)


def _keras_inputs_from_spec(tf, spec_tree):
    def build(spec, path):
        if isinstance(spec, TensorSpec):
            suffix = "_".join(map(str, path)) if path else "0"
            return tf.keras.Input(
                shape=_dims_to_keras_shape(spec.shape),
                dtype=spec.dtype.tf(),
                name=f"input_{suffix}",
            )
        if isinstance(spec, dict):
            return {k: build(v, (*path, k)) for k, v in spec.items()}
        if isinstance(spec, tuple):
            return tuple(build(v, (*path, i)) for i, v in enumerate(spec))
        if isinstance(spec, list):
            return [build(v, (*path, i)) for i, v in enumerate(spec)]
        raise TypeError(f"Expected TensorSpec leaves, got {type(spec).__name__}.")

    return build(spec_tree, ())


def _tree_to_tf(tf, value):
    def leaf_to_tf(leaf):
        if tf.is_tensor(leaf):
            return leaf
        return tf.convert_to_tensor(leaf)

    return map_leaves(value, leaf_to_tf)


def _tree_to_tf_model_batch(tf, value, input_spec):
    def leaf_to_tf(values):
        leaf, spec = values
        if not isinstance(spec, TensorSpec):
            raise TypeError(f"Expected TensorSpec leaves, got {type(spec).__name__}.")
        tensor = leaf if tf.is_tensor(leaf) else tf.convert_to_tensor(leaf)
        return tensor if spec.batched else tf.expand_dims(tensor, axis=0)

    return map_leaf_groups((value, input_spec), leaf_to_tf)


def _unbatch_tree(value):
    return map_leaves(value, lambda leaf: leaf[0])


def _reset_metric(metric):
    reset = getattr(metric, "reset_state", None) or getattr(metric, "reset_states", None)
    if reset is not None:
        reset()


def _update_metric(metric, y, y_pred):
    update = getattr(metric, "update_state", None)
    if update is not None:
        update(y, y_pred)
        return None
    return metric(y, y_pred)


def _metric_results(metrics):
    out = {}
    for metric in metrics:
        result = getattr(metric, "result", None)
        if result is None:
            continue
        name = getattr(metric, "name", type(metric).__name__)
        out[name] = metric_value(result())
    return out


class Wrapper(Serializable):
    """Generic TensorFlow object wrapper exposing the backend object at ``.obj``."""

    def __init__(self, cls, *args, **kwargs):
        self.cls = validate_class(cls)
        self.args = args
        self.kwargs = kwargs
        self.obj = self.cls(*args, **kwargs)
        self._pending_restore_path = None
        self._restore_checkpoint = None
        self._restore_status = None

    def save_state_to_dir_imp(self, dest_dir: str, *, codec: str) -> None:
        """Save backend checkpoint files when the wrapped object has state.

        Args:
            dest_dir: Empty local-state data directory owned by the Store.
            codec: Active local-state codec identifier.

        Side Effects:
            Writes TensorFlow checkpoint files only when TensorFlow can emit a
            non-empty checkpoint, preserving the Store manifest's regular-file
            payload invariant for stateless wrapped values.
        """
        import tensorflow as tf

        if not getattr(self, "_checkpoint_stateful", True):
            return

        ckpt_dir = os.path.join(dest_dir, "object.ckpt")
        os.makedirs(ckpt_dir, exist_ok=True)
        try:
            checkpoint = tf.train.Checkpoint(obj=self.obj)
        except ValueError:
            return

        manager = tf.train.CheckpointManager(checkpoint, ckpt_dir, max_to_keep=1)
        manager.save()
        if not any(files for _, _, files in os.walk(ckpt_dir)):
            shutil.rmtree(ckpt_dir)

    def restore_state_from_dir_imp(self, src_dir: str, *, codec: str) -> None:
        import tensorflow as tf

        ckpt_dir = os.path.join(src_dir, "object.ckpt")
        latest = tf.train.latest_checkpoint(ckpt_dir)
        if latest is None:
            return
        self._pending_restore_path = latest
        try:
            self._restore_checkpoint = tf.train.Checkpoint(obj=self.obj)
        except ValueError:
            return
        self._restore_status = self._restore_checkpoint.restore(latest)

    def restore_pending(self):
        return self._restore_status


class Optimizer(Wrapper):
    """First-class Keras optimizer object for experiment hyperparameters."""

    def __init__(self, cls, *args, **kwargs):
        super().__init__(cls, *args, **kwargs)
        self._pending_restore_path = None
        self._restore_checkpoint = None
        self._restore_status = None

    def save_state_to_dir_imp(self, dest_dir: str, *, codec: str) -> None:
        import tensorflow as tf

        ckpt_dir = os.path.join(dest_dir, "optimizer.ckpt")
        os.makedirs(ckpt_dir, exist_ok=True)
        manager = tf.train.CheckpointManager(
            tf.train.Checkpoint(optimizer=self.obj),
            ckpt_dir,
            max_to_keep=1,
        )
        manager.save()

    def restore_state_from_dir_imp(self, src_dir: str, *, codec: str) -> None:
        import tensorflow as tf

        ckpt_dir = os.path.join(src_dir, "optimizer.ckpt")
        latest = tf.train.latest_checkpoint(ckpt_dir)
        if latest is None:
            return
        self._pending_restore_path = latest
        self._restore_checkpoint = tf.train.Checkpoint(optimizer=self.obj)
        self._restore_status = self._restore_checkpoint.restore(latest)

    def restore_pending(self):
        if self._pending_restore_path is None:
            return None
        return self._restore_status


class Loss(Wrapper):
    """First-class Keras loss object for experiment hyperparameters."""

    _checkpoint_stateful = False

    def save_state_to_dir_imp(self, dest_dir: str, *, codec: str) -> None:
        """Leave local state empty because Keras losses are structural values.

        Args:
            dest_dir: Empty Store-owned local-state data directory.
            codec: Active local-state codec identifier.

        Side Effects:
            None. Loss configuration is already represented by its immutable
            concrete definition, so no empty checkpoint directory is emitted.
        """


class Metric(Wrapper):
    """First-class Keras metric object for experiment hyperparameters."""


class Model(BaseModel, Serializable):
    """Wrapper around a TensorFlow/Keras model class."""

    def __init__(self, cls, *args, output_spec=None, **kwargs):
        self.cls = validate_class(cls)
        self.model_args = args
        self.model_kwargs = kwargs
        self.obj = self.cls(*args, **kwargs)
        self.model = self.obj
        self.mdl = self.obj
        self.output_spec = output_spec
        self._pending_restore_path = None
        self._restore_checkpoint = None
        self._restore_status = None

    def __call__(self, x, *args, **kwargs):
        return self.obj(x, *args, **kwargs)

    def bind_first(self, first_value, *, input_spec=None):
        if input_spec is None or spec_tree_is_batched(input_spec):
            return self, self(first_value)

        import tensorflow as tf

        def bound_model(x):
            batched = _tree_to_tf_model_batch(tf, x, input_spec)
            return _unbatch_tree(self.obj(batched))

        return bound_model, bound_model(first_value)

    def fit(self, *args, **kwargs):
        return self.obj.fit(*args, **kwargs)

    def trainable_parameters(self, backend: str | None = None):
        if backend not in (None, "tf"):
            return ()
        return tuple(self.obj.trainable_variables)

    def compile(self, *, optimizer=None, loss=None, metrics=None, **kwargs):
        if optimizer is not None:
            kwargs["optimizer"] = optimizer.obj if hasattr(optimizer, "obj") else optimizer
        if loss is not None:
            kwargs["loss"] = loss.obj if hasattr(loss, "obj") else loss
        if metrics is not None:
            kwargs["metrics"] = [metric.obj if hasattr(metric, "obj") else metric for metric in metrics]
        return self.obj.compile(**kwargs)

    def infer_output_spec(self, input_spec):
        if self.output_spec is not None:
            return super().infer_output_spec(input_spec)

        import warnings
        import tensorflow as tf

        sample = map_leaves(fake_from_spec_tree(input_spec), tf.convert_to_tensor)
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                output = self.obj(sample, training=False)
        except ValueError as e:
            message = str(e)
            if "input" in message and ("expects" in message or "Arguments received" in message):
                raise ValueError(
                    "Input spec structure does not match the TensorFlow model input structure. "
                    "If the dataset element contains both features and labels, select the feature branch first, "
                    "for example Map(dataset, Select(0), model)."
                ) from e
            raise

        for warning in caught:
            message = str(warning.message)
            if "structure of `inputs` doesn't match" in message:
                raise ValueError(
                    "Input spec structure does not match the TensorFlow model input structure. "
                    "If the dataset element contains both features and labels, select the feature branch first, "
                    "for example Map(dataset, Select(0), model)."
                )

        return maybe_unbatch_output_spec(tf_as_tensor_spec(output, batched=True), input_spec)

    def save_state_to_dir_imp(self, dest_dir: str, *, codec: str) -> None:
        import tensorflow as tf

        ckpt_dir = os.path.join(dest_dir, "model.ckpt")
        os.makedirs(ckpt_dir, exist_ok=True)
        manager = tf.train.CheckpointManager(
            tf.train.Checkpoint(model=self.obj),
            ckpt_dir,
            max_to_keep=1,
        )
        manager.save()

    def restore_state_from_dir_imp(self, src_dir: str, *, codec: str) -> None:
        import tensorflow as tf

        ckpt_dir = os.path.join(src_dir, "model.ckpt")
        latest = tf.train.latest_checkpoint(ckpt_dir)
        if latest is None:
            return
        self._pending_restore_path = latest
        self._restore_checkpoint = tf.train.Checkpoint(model=self.obj)
        self._restore_status = self._restore_checkpoint.restore(latest)

    def restore_pending(self):
        if self._pending_restore_path is None:
            return None
        return self._restore_status


class TrainFunction(BaseTrainFunction):
    pass


class BasicTraining(TrainFunction):
    """Fit a Keras model from an Experiment's supervised train_data."""

    def __init__(
        self,
        *,
        optimizer=None,
        loss=None,
        metrics=(),
        compile_kwargs=None,
        epochs: int = 1,
        batch_size: int | None = 32,
        x_path=0,
        y_path=1,
        num_examples: int | None = None,
        shuffle: bool = False,
        shuffle_seed=None,
        shuffle_buffer_size: int | None = None,
        callbacks=(),
        fit_args=(),
        fit_kwargs=None,
        verbose: int = 1,
    ):
        if epochs < 0:
            raise ValueError("epochs must be non-negative.")
        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be positive or None.")
        validate_num_examples(num_examples)

        self.optimizer = optimizer
        self.loss = loss
        self.metrics = _normalize_list(metrics)
        self.compile_kwargs = dict(compile_kwargs or {})
        self.epochs = epochs
        self.batch_size = batch_size
        self.x_path = x_path
        self.y_path = y_path
        self.num_examples = num_examples
        self.shuffle = shuffle
        self.shuffle_seed = shuffle_seed
        self.shuffle_buffer_size = shuffle_buffer_size
        if callbacks is None:
            self.callbacks = ()
        elif isinstance(callbacks, (tuple, list)):
            self.callbacks = tuple(callbacks)
        else:
            self.callbacks = (callbacks,)
        self.fit_args = tuple(fit_args)
        self.fit_kwargs = dict(fit_kwargs or {})
        self.verbose = verbose

    def __call__(self, exp):
        import tensorflow as tf

        train_data = self._prepare_data(exp.train_data, for_training=True)
        train_xy = self._xy_data(train_data)

        val_data = None
        val_xy = None
        if exp.val_data is not None:
            val_data = self._prepare_data(exp.val_data, for_training=False)
            val_xy = self._xy_data(val_data)

        training_model = self._training_model(tf, exp.model, train_xy.spec[0])
        compile_kwargs = self._compile_kwargs(exp)
        if compile_kwargs:
            training_model.compile(**compile_kwargs)
            optimizer = self._optimizer(exp)
            if hasattr(optimizer, "restore_pending"):
                optimizer.restore_pending()

        exp.model.prep_train()
        if hasattr(exp.model, "restore_pending"):
            exp.model.restore_pending()
        fit_kwargs = dict(self.fit_kwargs)
        fit_kwargs.setdefault("verbose", self.verbose)
        callbacks = self._callbacks(tf)
        callbacks.extend(fit_kwargs.pop("callbacks", []) or [])

        if "steps_per_epoch" not in fit_kwargs:
            steps_per_epoch = finite_dataset_len(train_data)
            if steps_per_epoch is not None:
                fit_kwargs["steps_per_epoch"] = steps_per_epoch

        if val_xy is not None and "validation_steps" not in fit_kwargs:
            validation_steps = finite_dataset_len(val_data)
            if validation_steps is not None:
                fit_kwargs["validation_steps"] = validation_steps

        ds_train = self._tf_dataset(tf, train_xy)
        if fit_kwargs.get("steps_per_epoch") is not None:
            ds_train = ds_train.repeat()

        ds_val = None
        if val_xy is not None:
            ds_val = self._tf_dataset(tf, val_xy)
            if fit_kwargs.get("validation_steps") is not None:
                ds_val = ds_val.repeat()

        try:
            history = training_model.fit(
                ds_train,
                *self.fit_args,
                validation_data=ds_val,
                initial_epoch=exp.state.epoch,
                epochs=exp.state.epoch + self.epochs,
                callbacks=callbacks or None,
                **fit_kwargs,
            )
        finally:
            exp.model.prep_eval()

        steps_per_epoch = fit_kwargs.get("steps_per_epoch")
        advance_train_state(exp, epochs=self.epochs, steps=(int(steps_per_epoch) if steps_per_epoch is not None else 0) * self.epochs)
        return history

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

    def _compile_kwargs(self, exp):
        compile_kwargs = dict(self.compile_kwargs)
        optimizer = self._optimizer(exp)
        loss = self._loss(exp)
        metrics = self._metrics(exp)
        if optimizer is not None:
            compile_kwargs["optimizer"] = _unwrap_backend_obj(optimizer)
        if loss is not None:
            compile_kwargs["loss"] = _unwrap_backend_obj(loss)
        if metrics:
            compile_kwargs["metrics"] = [_unwrap_backend_obj(metric) for metric in metrics]
        return compile_kwargs

    def _training_model(self, tf, model, x_spec):
        if hasattr(model, "compile") and hasattr(model, "fit"):
            return model

        inputs = _keras_inputs_from_spec(tf, x_spec)
        outputs = model(inputs)
        return tf.keras.Model(inputs=inputs, outputs=outputs)

    def _prepare_data(self, data, *, for_training: bool):
        data = prepare_training_data(
            data,
            num_examples=self.num_examples if for_training else None,
            shuffle=self.shuffle if for_training else False,
            shuffle_seed=self.shuffle_seed,
            shuffle_buffer_size=self.shuffle_buffer_size,
        )
        if self.batch_size is not None:
            data = Batch(data, self.batch_size)
        return data

    def _xy_data(self, data):
        return Map(data, Project(Select(self.x_path), Select(self.y_path)))

    def _tf_dataset(self, tf, data):
        return tf.data.Dataset.from_generator(
            lambda: iter(data),
            output_signature=tf_output_signature(data.spec),
        )

    def _callbacks(self, tf):
        return [callback.obj if hasattr(callback, "obj") else callback for callback in self.callbacks]


class Training(BasicTraining):
    """Low-level TensorFlow training loop for arbitrary TF-callable DRYML models."""

    def __call__(self, exp):
        import tensorflow as tf

        train_data = self._prepare_data(exp.train_data, for_training=True)
        train_xy = self._xy_data(train_data)

        val_xy = None
        if exp.val_data is not None:
            val_data = self._prepare_data(exp.val_data, for_training=False)
            val_xy = self._xy_data(val_data)

        optimizer = self._make_optimizer(tf, exp)
        loss_fn = self._make_loss(tf, exp)
        metrics = self._metric_objects(exp)
        trainable_variables = None
        losses = []
        steps = 0
        steps_per_epoch = finite_dataset_len(train_data)
        total_steps = None if steps_per_epoch is None else steps_per_epoch * self.epochs
        progress = TrainingProgress(total=total_steps, verbose=self.verbose, desc="TF training")

        exp.model.prep_train()
        try:
            for epoch in range(self.epochs):
                for metric in metrics:
                    _reset_metric(metric)
                epoch_loss = 0.0
                epoch_steps = 0

                for x, y in train_xy:
                    x = _tree_to_tf(tf, x)
                    y = _tree_to_tf(tf, y)

                    with tf.GradientTape() as tape:
                        y_pred = exp.model(x)
                        loss_value = tf.reduce_mean(loss_fn(y, y_pred))

                    if trainable_variables is None:
                        trainable_variables = _collect_trainable_parameters(exp.model)
                        if not trainable_variables:
                            raise ValueError("TensorFlow model graph exposes no trainable parameters.")

                    grads = tape.gradient(loss_value, trainable_variables)
                    grad_pairs = [
                        (grad, var)
                        for grad, var in zip(grads, trainable_variables)
                        if grad is not None
                    ]
                    optimizer.apply_gradients(grad_pairs)

                    for metric in metrics:
                        _update_metric(metric, y, y_pred)

                    loss_float = float(metric_value(loss_value))
                    losses.append(loss_float)
                    epoch_loss += loss_float
                    epoch_steps += 1
                    steps += 1

                    step_metrics = {"loss": loss_float}
                    step_metrics.update(_metric_results(metrics))
                    progress.update(1, step_metrics)

                if epoch_steps == 0:
                    continue

                epoch_metrics = {"loss": epoch_loss / epoch_steps}
                epoch_metrics.update(_metric_results(metrics))

                if val_xy is not None:
                    val_metrics = self._evaluate(tf, exp.model, val_xy, loss_fn, metrics)
                    epoch_metrics.update({f"val_{name}": value for name, value in val_metrics.items()})

                progress.epoch_end(epoch + 1, epochs=self.epochs, metrics=epoch_metrics)
        finally:
            progress.close()
            exp.model.prep_eval()

        if steps == 0 and self.epochs > 0:
            raise ValueError("Cannot train on an empty dataset.")

        advance_train_state(exp, epochs=self.epochs, steps=steps)
        return losses

    def _make_optimizer(self, tf, exp):
        optimizer = _unwrap_backend_obj(self._optimizer(exp))
        if optimizer is not None:
            if isinstance(optimizer, type):
                return validate_class(optimizer)()
            return optimizer
        return tf.keras.optimizers.Adam()

    def _make_loss(self, tf, exp):
        loss = _unwrap_backend_obj(self._loss(exp))
        if loss is not None:
            if isinstance(loss, type):
                return validate_class(loss)()
            return loss
        return tf.keras.losses.MeanSquaredError()

    def _metric_objects(self, exp):
        return [_unwrap_backend_obj(metric) for metric in self._metrics(exp)]

    def _evaluate(self, tf, model, val_xy, loss_fn, metrics):
        for metric in metrics:
            _reset_metric(metric)
        total_loss = 0.0
        steps = 0
        for x, y in val_xy:
            x = _tree_to_tf(tf, x)
            y = _tree_to_tf(tf, y)
            y_pred = model(x)
            loss_value = tf.reduce_mean(loss_fn(y, y_pred))
            total_loss += float(metric_value(loss_value))
            steps += 1
            for metric in metrics:
                _update_metric(metric, y, y_pred)

        if steps == 0:
            return {}

        results = {"loss": total_loss / steps}
        results.update(_metric_results(metrics))
        return results


class BasicEarlyStoppingTraining(BasicTraining):
    def __init__(
        self,
        *args,
        patience: int = 3,
        monitor: str = "val_loss",
        restore_best_weights: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.patience = patience
        self.monitor = monitor
        self.restore_best_weights = restore_best_weights

    def _callbacks(self, tf):
        callbacks = super()._callbacks(tf)
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                patience=self.patience,
                monitor=self.monitor,
                restore_best_weights=self.restore_best_weights,
            )
        )
        return callbacks


ModelWrapper = Model


__all__ = [
    "BasicEarlyStoppingTraining",
    "BasicTraining",
    "Loss",
    "Metric",
    "Model",
    "ModelWrapper",
    "Optimizer",
    "Training",
    "TrainFunction",
    "Wrapper",
]
