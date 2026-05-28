from __future__ import annotations

import os

from dryml.core2.object import Object
from dryml.core2.tensor_spec import Dynamic, Layout
from dryml.core2.utils.general import revision_path, validate_class
from dryml.core2.utils.recurse import map_leaves
from dryml.data import Batch, iter_xy
from dryml.models import Model as BaseModel
from dryml.models import TrainFunction as BaseTrainFunction
from dryml.models.utils import advance_train_state, finite_dataset_len, prepare_training_data, validate_num_examples
from dryml.data.transforms import Select


def _tf_shape(shape):
    if shape is None:
        return None
    return tuple(None if dim is Dynamic else int(dim) for dim in shape)


def _tf_signature_leaf(tf, spec):
    # DRYML Batch(..., drop_remainder=False) can yield a smaller final batch,
    # so TensorFlow signatures must not pin the leading batch dimension.
    if spec.batched and spec.shape is not None:
        shape = (None, *_tf_shape(spec.shape))
    else:
        shape = _tf_shape(spec.full_shape)
    dtype = tf.as_dtype(spec.dtype.name)

    if spec.layout is Layout.DENSE:
        return tf.TensorSpec(shape=shape, dtype=dtype)
    if spec.layout is Layout.RAGGED:
        return tf.RaggedTensorSpec(
            shape=shape,
            dtype=dtype,
            ragged_rank=spec.ragged_rank,
            row_splits_dtype=(
                tf.int64 if spec.row_splits_dtype is None else tf.as_dtype(spec.row_splits_dtype.name)
            ),
        )
    if spec.layout is Layout.SPARSE:
        return tf.SparseTensorSpec(shape=shape, dtype=dtype)
    raise TypeError(f"Unsupported TensorFlow layout: {spec.layout}.")


def _tf_output_signature(tf, spec_tree):
    return map_leaves(spec_tree, lambda spec: _tf_signature_leaf(tf, spec))


def _path_select(path):
    if isinstance(path, (tuple, list)):
        return Select(*path)
    return Select(path)


def _tf_dataset_from_xy(tf, dataset, *, x_path, y_path):
    x_select = _path_select(x_path)
    y_select = _path_select(y_path)
    x_spec = x_select.infer_output_spec(dataset.spec)
    y_spec = y_select.infer_output_spec(dataset.spec)
    output_signature = (
        _tf_output_signature(tf, x_spec),
        _tf_output_signature(tf, y_spec),
    )

    return tf.data.Dataset.from_generator(
        lambda: iter_xy(dataset, x_path=x_path, y_path=y_path),
        output_signature=output_signature,
    )


class Wrapper(Object):
    """Generic TensorFlow object wrapper exposing the backend object at ``.obj``."""

    def __init__(self, cls, *args, **kwargs):
        self.cls = validate_class(cls)
        self.args = args
        self.kwargs = kwargs
        self.obj = self.cls(*args, **kwargs)
        self._pending_restore_path = None
        self._restore_checkpoint = None
        self._restore_status = None

    def save_state_to_dir_imp(self, dest_dir: str, revision: str | None = None):
        import tensorflow as tf

        ckpt_dir = revision_path("object", "ckpt", dest_dir, revision=revision)
        os.makedirs(ckpt_dir, exist_ok=True)
        manager = tf.train.CheckpointManager(
            tf.train.Checkpoint(obj=self.obj),
            ckpt_dir,
            max_to_keep=1,
        )
        manager.save()

    def restore_state_from_dir_imp(self, src_dir: str, revision: str | None):
        import tensorflow as tf

        ckpt_dir = revision_path("object", "ckpt", src_dir, revision=revision)
        latest = tf.train.latest_checkpoint(ckpt_dir)
        if latest is None:
            return
        self._pending_restore_path = latest
        self._restore_checkpoint = tf.train.Checkpoint(obj=self.obj)
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

    def save_state_to_dir_imp(self, dest_dir: str, revision: str | None = None):
        import tensorflow as tf

        ckpt_dir = revision_path("optimizer", "ckpt", dest_dir, revision=revision)
        os.makedirs(ckpt_dir, exist_ok=True)
        manager = tf.train.CheckpointManager(
            tf.train.Checkpoint(optimizer=self.obj),
            ckpt_dir,
            max_to_keep=1,
        )
        manager.save()

    def restore_state_from_dir_imp(self, src_dir: str, revision: str | None):
        import tensorflow as tf

        ckpt_dir = revision_path("optimizer", "ckpt", src_dir, revision=revision)
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


class Metric(Wrapper):
    """First-class Keras metric object for experiment hyperparameters."""


class Model(BaseModel):
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

    def fit(self, *args, **kwargs):
        return self.obj.fit(*args, **kwargs)

    def compile(self, *, optimizer=None, loss=None, metrics=None, **kwargs):
        if optimizer is not None:
            kwargs["optimizer"] = optimizer.obj if hasattr(optimizer, "obj") else optimizer
        if loss is not None:
            kwargs["loss"] = loss.obj if hasattr(loss, "obj") else loss
        if metrics is not None:
            kwargs["metrics"] = [metric.obj if hasattr(metric, "obj") else metric for metric in metrics]
        return self.obj.compile(**kwargs)

    def save_state_to_dir_imp(self, dest_dir: str, revision: str | None = None):
        import tensorflow as tf

        ckpt_dir = revision_path("model", "ckpt", dest_dir, revision=revision)
        os.makedirs(ckpt_dir, exist_ok=True)
        manager = tf.train.CheckpointManager(
            tf.train.Checkpoint(model=self.obj),
            ckpt_dir,
            max_to_keep=1,
        )
        manager.save()

    def restore_state_from_dir_imp(self, src_dir: str, revision: str | None):
        import tensorflow as tf

        ckpt_dir = revision_path("model", "ckpt", src_dir, revision=revision)
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
    ):
        if epochs < 0:
            raise ValueError("epochs must be non-negative.")
        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be positive or None.")
        validate_num_examples(num_examples)

        self.optimizer = optimizer
        self.loss = loss
        if metrics is None:
            self.metrics = ()
        elif isinstance(metrics, (tuple, list)):
            self.metrics = tuple(metrics)
        else:
            self.metrics = (metrics,)
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

    def __call__(self, exp):
        import tensorflow as tf

        train_data = self._prepare_data(exp.train_data, for_training=True)
        ds_train = _tf_dataset_from_xy(
            tf,
            train_data,
            x_path=self.x_path,
            y_path=self.y_path,
        )

        ds_val = None
        if exp.val_data is not None:
            val_data = self._prepare_data(exp.val_data, for_training=False)
            ds_val = _tf_dataset_from_xy(
                tf,
                val_data,
                x_path=self.x_path,
                y_path=self.y_path,
            )

        compile_kwargs = dict(self.compile_kwargs)
        if self.optimizer is not None:
            compile_kwargs["optimizer"] = self.optimizer
        if self.loss is not None:
            compile_kwargs["loss"] = self.loss
        if self.metrics:
            compile_kwargs["metrics"] = self.metrics
        if compile_kwargs:
            exp.model.compile(**compile_kwargs)
            optimizer = compile_kwargs.get("optimizer")
            if hasattr(optimizer, "restore_pending"):
                optimizer.restore_pending()

        exp.model.prep_train()
        if hasattr(exp.model, "restore_pending"):
            exp.model.restore_pending()
        fit_kwargs = dict(self.fit_kwargs)
        callbacks = self._callbacks(tf)
        callbacks.extend(fit_kwargs.pop("callbacks", []) or [])
        steps_per_epoch = finite_dataset_len(train_data)
        if steps_per_epoch is not None:
            fit_kwargs.setdefault("steps_per_epoch", steps_per_epoch)
        if ds_val is not None:
            validation_steps = finite_dataset_len(val_data)
            if validation_steps is not None:
                fit_kwargs.setdefault("validation_steps", validation_steps)

        try:
            history = exp.model.fit(
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

        advance_train_state(exp, epochs=self.epochs, steps=(steps_per_epoch or 0) * self.epochs)
        return history

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

    def _callbacks(self, tf):
        return [callback.obj if hasattr(callback, "obj") else callback for callback in self.callbacks]


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
    "TrainFunction",
    "Wrapper",
]
