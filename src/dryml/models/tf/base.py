from __future__ import annotations

import os
import shutil
import weakref
from dataclasses import replace

from dryml.core.object import Serializable
from dryml.core.backend import Backend
from dryml.core.repo import manage_repo
from dryml.core.tensor_spec import Dynamic, TensorSpec, batch_spec_tree, iter_specs, map_spec_tree, maybe_unbatch_output_spec, spec_tree_is_batched
from dryml.core.utils.general import maybe_call_method, validate_class
from dryml.core.utils.recurse import map_leaf_groups, map_leaves
from dryml.data import Batch, Map, Project, Select
from dryml.models import Model as BaseModel
from dryml.models import TrainFunction as BaseTrainFunction
from dryml.models.progress import TrainingProgress, metric_value
from dryml.models.utils import (
    finite_dataset_len,
    prepare_training_data,
    record_train_update,
    require_bounded_safe_points,
    TrainingPreparation,
    validate_num_examples,
    validate_training_callbacks,
)
from dryml.tf.tensor_spec import output_signature as tf_output_signature
from dryml.methods import ImplementationSelectionError, MethodError, traits


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


def _collect_trainable_parameters(target, *, repo):
    """Collect graph trainables using the supplied explicit Repo authority.

    Args:
        target: Live DRYML graph root whose trainable nodes are collected.
        repo: Bounded Repo authority used to traverse retained runtime bindings.

    Returns:
        A flat list of TensorFlow trainable variables in post-order graph order.

    Raises:
        KeyError: If a required runtime binding is unavailable from ``target``.
    """

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


def _keras_input_shapes(tf, spec_tree):
    """Convert normalized input specs into TensorFlow shape metadata only."""

    return map_leaves(
        spec_tree,
        lambda spec: tf.TensorShape(_dims_to_keras_shape(spec.framework_shape())),
    )


def _keras_output_specs(output_shapes, *, dtype):
    """Translate Keras output shape metadata into batched TensorSpecs."""

    if hasattr(output_shapes, "as_list"):
        dims = output_shapes.as_list()
    elif isinstance(output_shapes, (tuple, list)) and all(
        dimension is None or isinstance(dimension, int) for dimension in output_shapes
    ):
        dims = output_shapes
    else:
        if isinstance(output_shapes, dict):
            return {key: _keras_output_specs(value, dtype=dtype) for key, value in output_shapes.items()}
        if isinstance(output_shapes, tuple):
            return tuple(_keras_output_specs(value, dtype=dtype) for value in output_shapes)
        if isinstance(output_shapes, list):
            return [_keras_output_specs(value, dtype=dtype) for value in output_shapes]
        raise TypeError("TensorFlow output shape metadata must be a TensorShape tree.")

    if dims is None or not dims:
        raise NotImplementedError("TensorFlow output shape metadata has unknown rank.")
    return TensorSpec(
        dtype=dtype,
        shape=tuple(Dynamic if dim is None else int(dim) for dim in dims[1:]),
        batch=Dynamic if dims[0] is None else int(dims[0]),
        backend="tf",
    )


def _keras_output_dtype(tf, model, fallback):
    """Return trustworthy built-in Keras dtype metadata without user calls."""

    layers = tuple(model.layers)
    for layer in layers:
        module = type(layer).__module__
        if isinstance(layer, tf.keras.layers.Lambda) or not module.startswith(("keras.", "tensorflow.")):
            raise NotImplementedError("Custom TensorFlow layers require an explicit output_spec.")
    return getattr(layers[-1], "compute_dtype", None) if layers else fallback


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


def _require_mean_keras_loss(loss) -> None:
    """Reject Keras losses whose scalar result has no mean-loss meaning."""

    reduction = getattr(loss, "reduction", None)
    reduction = getattr(reduction, "value", reduction)
    if reduction not in {"auto", "mean", "mean_with_sample_weight", "sum_over_batch_size"}:
        raise ValueError(
            "TensorFlow training accounting requires a Keras loss with mean reduction."
        )


def _require_supported_keras_accounting(compile_kwargs, fit_kwargs) -> None:
    """Reject Keras objective options that cannot retain an honest loss mean."""

    if compile_kwargs.get("loss_weights") is not None:
        raise ValueError("Keras loss_weights are unsupported by retained loss accounting.")
    if fit_kwargs.get("class_weight") is not None:
        raise ValueError("Keras class_weight is unsupported by retained loss accounting.")
    if fit_kwargs.get("sample_weight") is not None:
        raise ValueError("Keras sample_weight is unsupported by retained loss accounting.")


def _require_supported_keras_execution(compile_kwargs, optimizer) -> None:
    """Reject Keras execution grouping that hides individual optimizer updates.

    Args:
        compile_kwargs: Pending Keras compile keyword arguments.
        optimizer: Resolved native Keras optimizer, when configured.

    Raises:
        ValueError: If Keras would aggregate multiple updates behind one callback.

    Side Effects:
        None. This preflight runs before compiling, model preparation, target
        allocation, or Dataset iteration.
    """

    steps_per_execution = compile_kwargs.get("steps_per_execution", 1)
    if type(steps_per_execution) is not int or steps_per_execution != 1:
        raise ValueError("Keras steps_per_execution must be exactly 1 for retained update accounting.")
    accumulation = getattr(optimizer, "gradient_accumulation_steps", None)
    if accumulation is not None and (type(accumulation) is not int or accumulation != 1):
        raise ValueError("Keras optimizer gradient accumulation is unsupported by retained update accounting.")


def _freeze_native_keras_callbacks(tf, configured, fit_kwargs) -> tuple:
    """Freeze and validate native callback inputs before any training setup.

    Args:
        tf: Imported TensorFlow module supplying the Keras callback base type.
        configured: Native callbacks configured on the trainer.
        fit_kwargs: Per-call Keras fit keywords; its ``callbacks`` entry is
            consumed after validation.

    Returns:
        A frozen native callback sequence in the order Keras will receive it.

    Raises:
        TypeError: If a callback keyword is malformed or a member is not a
            native Keras callback.

    Side Effects:
        Removes the validated ``callbacks`` keyword from this local fit-keyword
        copy. It does not prepare data, mutate a Method graph, compile, or touch
        model/optimizer restoration.
    """

    def freeze(value, name):
        if value is None:
            return ()
        if isinstance(value, (tuple, list)):
            values = tuple(value)
        else:
            values = (value,)
        values = tuple(_unwrap_backend_obj(callback) for callback in values)
        if not all(isinstance(callback, tf.keras.callbacks.Callback) for callback in values):
            raise TypeError(f"Keras {name} must contain only keras.callbacks.Callback instances.")
        return values

    return (*freeze(configured, "callbacks"), *freeze(fit_kwargs.pop("callbacks", ()), "fit callbacks"))


def _keras_accounting_model(tf, model):
    """Wrap one Keras model with owned post-update accounting facts."""

    class AccountingModel(tf.keras.Model):
        def __init__(self, base):
            super().__init__(name=f"dryml_accounting_{base.name}")
            self.base = base
            self.completed_loss = tf.Variable(0.0, dtype=tf.float64, trainable=False)
            self.completed_examples = tf.Variable(0, dtype=tf.int64, trainable=False)
            self.has_completed_step = tf.Variable(False, dtype=tf.bool, trainable=False)

        def call(self, inputs, training=None):
            return self.base(inputs, training=training)

        def train_step(self, data):
            x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
            with tf.GradientTape() as tape:
                y_pred = self.base(x, training=True)
                loss = self.compiled_loss(
                    y,
                    y_pred,
                    sample_weight=sample_weight,
                    regularization_losses=self.base.losses,
                )
            variables = self.base.trainable_variables
            gradients = tape.gradient(loss, variables)
            self.optimizer.apply_gradients(
                (gradient, variable)
                for gradient, variable in zip(gradients, variables)
                if gradient is not None
            )
            self.compiled_metrics.update_state(y, y_pred, sample_weight=sample_weight)
            # This is the exact scalar differentiated by the completed update,
            # including regularization and dynamic add_loss contributions.
            self.completed_loss.assign(tf.cast(loss, tf.float64))
            first = tf.nest.flatten(x)[0]
            self.completed_examples.assign(tf.cast(tf.shape(first)[0], tf.int64))
            self.has_completed_step.assign(True)
            return {metric.name: metric.result() for metric in self.metrics}

        def test_step(self, data):
            """Evaluate the wrapped model without entering the owned update step."""

            x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
            y_pred = self.base(x, training=False)
            self.compiled_loss(
                y,
                y_pred,
                sample_weight=sample_weight,
                regularization_losses=self.base.losses,
            )
            self.compiled_metrics.update_state(y, y_pred, sample_weight=sample_weight)
            return {metric.name: metric.result() for metric in self.metrics}

    return AccountingModel(model)


def _keras_accounting_callback(
    tf,
    exp,
    *,
    reporter,
    callbacks,
    steps_per_epoch,
    batch_offset=0,
):
    """Create Keras's post-update accounting callback for one fit segment."""

    class AccountingCallback(tf.keras.callbacks.Callback):
        def on_train_batch_end(self, batch, logs=None):
            del logs
            if not bool(reporter.has_completed_step.numpy()):
                raise RuntimeError("DRYML Keras accounting did not receive a completed train step.")
            physical_batch = batch_offset + batch
            record_train_update(
                exp,
                None,
                float(metric_value(reporter.completed_loss)),
                batched=True,
                examples=int(metric_value(reporter.completed_examples)),
                complete_epoch=(
                    steps_per_epoch is not None
                    and physical_batch + 1 == steps_per_epoch
                ),
                callbacks=callbacks,
            )

        def on_epoch_end(self, epoch, logs=None):
            del logs
            # Unknown finite streams reveal exhaustion only here. Retain the
            # postlude before later native callbacks can fail.
            if steps_per_epoch is None and exp.state.epoch == epoch:
                exp.state.finish_epoch(postlude_pending=True)

    return AccountingCallback()


def _keras_postlude_callback(tf, exp):
    """Clear a retained finite-epoch postlude after Keras completes it."""

    class PostludeCallback(tf.keras.callbacks.Callback):
        def on_train_batch_end(self, batch, logs=None):
            del batch, logs
            epoch = exp.state.pending_epoch_postlude
            if epoch is not None and exp.state.pending_epoch_postlude_phase == "start":
                exp.state.advance_epoch_postlude(epoch, "validation")

        def on_test_end(self, logs=None):
            del logs
            epoch = exp.state.pending_epoch_postlude
            if epoch is not None and exp.state.pending_epoch_postlude_phase == "validation":
                exp.state.advance_epoch_postlude(epoch, "epoch_end")

        def on_test_begin(self, logs=None):
            del logs
            # Unknown cardinality becomes observable when Keras starts the
            # validation postlude. Normalize before validation can fail.
            if exp.state.pending_epoch_postlude is None:
                exp.state.finish_epoch(postlude_pending=True)

        def on_epoch_end(self, epoch, logs=None):
            del logs
            if exp.state.pending_epoch_postlude == epoch:
                exp.state.finish_epoch_postlude(epoch)

    return PostludeCallback()


def _resume_keras_postlude(
    exp,
    training_model,
    native_callbacks,
    *,
    validation_data,
    validation_steps,
    steps_per_epoch,
) -> None:
    """Complete only the bounded Keras postlude that needs no native replay."""

    epoch = exp.state.pending_epoch_postlude
    if epoch is None:
        return
    phase = exp.state.pending_epoch_postlude_phase
    if native_callbacks:
        raise ValueError(
            "Keras cannot truthfully replay native callbacks after a saved post-update interruption."
        )
    if phase == "start":
        if validation_data is None:
            exp.state.finish_epoch_postlude(epoch)
            return
        exp.state.advance_epoch_postlude(epoch, "validation")
        phase = "validation"
    if phase == "validation":
        if validation_data is not None:
            training_model.evaluate(
                validation_data,
                steps=validation_steps,
                verbose=0,
                return_dict=True,
            )
        exp.state.advance_epoch_postlude(epoch, "epoch_end")
        phase = "epoch_end"
    if phase == "epoch_end":
        exp.state.finish_epoch_postlude(epoch)
        return
    if phase is not None:
        raise ValueError("Keras cannot truthfully replay an unknown native postlude phase.")


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
            shutil.rmtree(ckpt_dir)
            return

        manager = tf.train.CheckpointManager(checkpoint, ckpt_dir, max_to_keep=1)
        manager.save()
        if not any(files for _, _, files in os.walk(ckpt_dir)):
            shutil.rmtree(ckpt_dir)

    def restore_state_from_dir_imp(self, src_dir: str, *, codec: str) -> None:
        """Record a checkpoint for restoration after optimizer slots exist."""
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
        self._restore_checkpoint = None
        self._restore_status = None

    def restore_pending(self):
        """Consume and restore the checkpoint against current optimizer slots.

        Returns:
            TensorFlow restore status, or ``None`` when no checkpoint is pending.

        Side Effects:
            Attaches checkpoint state to current variables and consumes the
            pending path so later training cannot replay old optimizer state.
        """
        if self._pending_restore_path is None:
            return None
        import tensorflow as tf

        self._restore_checkpoint = tf.train.Checkpoint(optimizer=self.obj)
        self._restore_status = self._restore_checkpoint.restore(
            self._pending_restore_path
        )
        self._pending_restore_path = None
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
    """Wrapper around a TensorFlow/Keras model class.

    Direct calls use the raw Keras model. Spec-selected element calls add and
    remove one batch axis, while selected batched calls pass their input through.
    Output specs use Keras shape metadata only; unsupported custom models must
    receive an explicit ``output_spec``.
    """

    native_backend = "tf"

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

    def _call_raw(self, x, *args, **kwargs):
        result = self.obj(x, *args, **kwargs)
        if self._pending_restore_path is not None:
            self.restore_pending()
            result = self.obj(x, *args, **kwargs)
        return result

    def _runtime_selection_facts(self, args, kwargs):
        """Select TensorFlow's legacy eager wrapper path without value probing.

        Prepared calls use declared specs and local conversion edges instead. This
        hook preserves direct eager wrapper behavior for existing native training
        bodies, which are outside Dataset handoff planning.
        """

        del args, kwargs
        return Backend.tf, None

    @traits(backend="tf")
    def raw_call(self, x, *args, **kwargs):
        """Invoke the raw Keras model when batching intent is unavailable."""

        return self._call_raw(x, *args, **kwargs)

    @traits(backend="tf", batch_mode="batched")
    def batched_call(self, x, *args, **kwargs):
        """Invoke the raw Keras model with an already batched selected input."""

        return self._call_raw(x, *args, **kwargs)

    @traits(backend="tf", batch_mode="element")
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
            A selected callable that adds and removes one TensorFlow batch axis
            for element calls when ``input_spec`` is supplied.

        Raises:
            ImplementationSelectionError: If constraints are malformed,
                conflicting, unsupported, or select no unique implementation.

        Side Effects:
            Binds a local selected callable without changing preparation state.
        """

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
            import tensorflow as tf

            receiver = receiver_ref()
            if receiver is None:
                raise MethodError("The selected TensorFlow Model is no longer live.")
            batched = _tree_to_tf_model_batch(tf, x, input_spec)
            return _unbatch_tree(receiver._call_raw(batched, *args, **kwargs))

        return replace(implementation, _invoker=invoke_element)

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

    def infer_output_spec(self, input_spec, *additional_input_specs):
        """Infer TensorFlow output metadata from exactly one logical input spec.

        Args:
            input_spec: Normalized TensorFlow model input specification.
            *additional_input_specs: Unsupported additional inputs.

        Returns:
            The configured output specification or pure Keras shape metadata
            translated to the input's logical batch representation.

        Raises:
            TypeError: If additional input specifications are supplied.
            NotImplementedError: If the model lacks supported static metadata and
                has no configured output specification.
            ValueError: If supplied structured input metadata cannot match a
                supported Keras input structure.

        Side Effects:
            Imports TensorFlow only to inspect supported Keras metadata; it never
            invokes the model or allocates input tensors.
        """

        if additional_input_specs:
            raise TypeError("TensorFlow Model accepts exactly one input specification.")
        if self.output_spec is not None:
            return map_spec_tree(super().infer_output_spec(input_spec), lambda spec: TensorSpec(
                spec.dtype, shape=spec.shape, batch=spec.batch, backend="tf",
                layout=spec.layout, axis_names=spec.axis_names, batch_axis_name=spec.batch_axis_name,
            ))

        import tensorflow as tf
        if not isinstance(self.obj, tf.keras.Sequential):
            raise NotImplementedError(
                f"Cannot infer output spec for {type(self).__name__} without executing the model; "
                "pass output_spec explicitly."
            )
        source_spec = input_spec
        model_input_spec = input_spec if spec_tree_is_batched(input_spec) else batch_spec_tree(input_spec)
        try:
            output_shapes = self.obj.compute_output_shape(
                _keras_input_shapes(tf, model_input_spec)
            )
            dtype = _keras_output_dtype(tf, self.obj, next(iter_specs(source_spec)).dtype)
            output_spec = _keras_output_specs(output_shapes, dtype=dtype)
        except (AttributeError, TypeError, ValueError, NotImplementedError) as error:
            if isinstance(source_spec, tuple):
                raise ValueError(
                    "Input spec structure does not match the TensorFlow model input structure. "
                    "If the dataset element contains both features and labels, select the feature branch first, "
                    "for example Map(dataset, Select(0), model)."
                ) from error
            raise NotImplementedError(
                f"Cannot infer output spec for {type(self).__name__} without executing the model; "
                "pass output_spec explicitly."
            ) from error
        return maybe_unbatch_output_spec(output_spec, source_spec)

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
        """Record a checkpoint for restoration after model variables exist."""
        import tensorflow as tf

        ckpt_dir = os.path.join(src_dir, "model.ckpt")
        latest = tf.train.latest_checkpoint(ckpt_dir)
        if latest is None:
            return
        self._pending_restore_path = latest
        self._restore_checkpoint = None
        self._restore_status = None
        if getattr(self.obj, "built", False):
            self.restore_pending()
            self._pending_restore_path = None

    def restore_pending(self):
        """Consume and restore the checkpoint against current model variables.

        Returns:
            TensorFlow restore status, or ``None`` when no checkpoint is pending.

        Side Effects:
            Attaches checkpoint state to current variables and consumes the
            pending path so later inference or training cannot replay old state.
        """
        if self._pending_restore_path is None:
            return None
        import tensorflow as tf

        self._restore_checkpoint = tf.train.Checkpoint(model=self.obj)
        self._restore_status = self._restore_checkpoint.restore(
            self._pending_restore_path
        )
        self._pending_restore_path = None
        return self._restore_status


class TrainFunction(BaseTrainFunction):
    pass


class BasicTraining(TrainFunction):
    """Fit a Keras model with retained post-update training accounting.

    The owned Keras train step reports its actual completed-update mean supplied
    loss and example count before DRYML safe-point callbacks, then preserves
    caller native-callback order. Interrupted calls retain an epoch target and
    next-batch position so deterministic data resumes the same invocation.

    Args:
        optimizer: Keras optimizer wrapper or native optimizer capability.
        loss: Mean-reduced Keras loss wrapper or native loss capability.
        metrics: Optional native Keras metric collection.
        compile_kwargs: Additional Keras compile options.
        epochs: Epochs requested by a fresh invocation.
        batch_size: Positive trainer batch size, or ``None`` for source batches.
        x_path: Dataset path selecting model inputs.
        y_path: Dataset path selecting training targets.
        num_examples: Optional finite training-source prefix.
        shuffle: Whether to apply deterministic Dataset shuffling.
        shuffle_seed: Optional shuffle seed.
        shuffle_buffer_size: Required finite shuffle buffer for unknown sources.
        callbacks: Native Keras callbacks, invoked after DRYML accounting.
        fit_args: Positional arguments forwarded to ``keras.Model.fit``.
        fit_kwargs: Keyword arguments forwarded to ``keras.Model.fit``.
        verbose: Keras fit verbosity.

    Raises:
        TypeError: If DRYML safe-point callbacks are invalid.
        ValueError: If the loss is not mean-reduced, recovery cannot replay a
            deterministic finite batch position, or training ends early.

    Side Effects:
        Compiles/trains the Keras adapter, changes model and optimizer state,
        and advances ``Experiment.state`` only after successful updates.
    """

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

    def __call__(self, exp, *, callbacks=()):
        import tensorflow as tf

        callbacks = validate_training_callbacks(callbacks)
        fit_kwargs = dict(self.fit_kwargs)
        native_callbacks = _freeze_native_keras_callbacks(tf, self._callbacks(tf), fit_kwargs)
        if callbacks and (native_callbacks or exp.val_data is not None):
            raise ValueError(
                "Keras DRYML safe-point recovery does not support native callbacks or validation."
            )
        compile_kwargs = self._compile_kwargs(exp)
        _require_supported_keras_accounting(compile_kwargs, fit_kwargs)
        _require_supported_keras_execution(compile_kwargs, _unwrap_backend_obj(self._optimizer(exp)))
        configured_loss = compile_kwargs.get("loss", self._make_loss(tf, exp))
        _require_mean_keras_loss(configured_loss)
        self._begin_training_preparation_generation()
        train_data = self._prepare_data(exp.train_data, for_training=True)
        require_bounded_safe_points(train_data, callbacks)
        train_xy = self._xy_data(train_data)
        self.training_preparation = TrainingPreparation.from_specs(
            self, train_xy.spec[0], train_xy.spec[1], "tf"
        )

        val_data = None
        val_xy = None
        self.validation_preparation = None
        if exp.val_data is not None:
            val_data = self._prepare_data(exp.val_data, for_training=False)
            val_xy = self._xy_data(val_data)
            self.validation_preparation = TrainingPreparation.from_specs(
                self, val_xy.spec[0], val_xy.spec[1], "tf"
            )

        training_model = self._training_model(tf, exp.model, train_xy.spec[0])
        if compile_kwargs:
            training_model.compile(**compile_kwargs)
            optimizer = self._optimizer(exp)
            if hasattr(optimizer, "restore_pending"):
                optimizer.restore_pending()

        exp.model.prep_train()
        if hasattr(exp.model, "restore_pending"):
            exp.model.restore_pending()
        fit_kwargs.setdefault("verbose", self.verbose)
        if "steps_per_epoch" not in fit_kwargs:
            steps_per_epoch = finite_dataset_len(train_data)
            if steps_per_epoch is not None:
                fit_kwargs["steps_per_epoch"] = steps_per_epoch

        if val_xy is not None and "validation_steps" not in fit_kwargs:
            validation_steps = finite_dataset_len(val_data)
            if validation_steps is not None:
                fit_kwargs["validation_steps"] = validation_steps

        ds_train = self._tf_dataset(tf, train_xy, self.training_preparation)
        if fit_kwargs.get("steps_per_epoch") is not None:
            ds_train = ds_train.repeat()

        ds_val = None
        if val_xy is not None:
            ds_val = self._tf_dataset(tf, val_xy, self.validation_preparation)
            if fit_kwargs.get("validation_steps") is not None:
                ds_val = ds_val.repeat()

        start_epoch = exp.state.epoch
        steps_per_epoch = fit_kwargs.get("steps_per_epoch")
        fresh_target = exp.state.target_epoch is None
        initial_step = exp.state.step
        target_epoch = exp.state.begin_invocation(self.epochs)

        def fit_segment(dataset, *, initial_epoch, epochs, batch_offset, segment_steps):
            segment_kwargs = dict(fit_kwargs)
            if segment_steps is not None:
                segment_kwargs["steps_per_epoch"] = segment_steps
            accounting = _keras_accounting_callback(
                tf,
                exp,
                reporter=training_model,
                callbacks=callbacks,
                steps_per_epoch=steps_per_epoch,
                batch_offset=batch_offset,
            )
            return training_model.fit(
                dataset,
                *self.fit_args,
                validation_data=ds_val,
                initial_epoch=initial_epoch,
                epochs=epochs,
                callbacks=[accounting, *native_callbacks, _keras_postlude_callback(tf, exp)],
                **segment_kwargs,
            )

        try:
            _resume_keras_postlude(
                exp,
                training_model,
                native_callbacks,
                validation_data=ds_val,
                validation_steps=fit_kwargs.get("validation_steps"),
                steps_per_epoch=steps_per_epoch,
            )
            if target_epoch <= start_epoch and exp.state.pending_epoch_postlude is None:
                history = None
            elif exp.state.next_batch:
                if steps_per_epoch is None:
                    raise ValueError("Keras recovery requires a finite deterministic batch count.")
                if exp.state.next_batch > steps_per_epoch:
                    raise ValueError("Saved Keras batch position exceeds the training epoch.")
                remaining = steps_per_epoch - exp.state.next_batch
                if remaining:
                    history = fit_segment(
                        ds_train.skip(exp.state.next_batch),
                        initial_epoch=start_epoch,
                        epochs=start_epoch + 1,
                        batch_offset=exp.state.next_batch,
                        segment_steps=remaining,
                    )
                else:
                    # Accept a legacy/external checkpoint captured after the
                    # final update but before its epoch-normalization bridge.
                    exp.state.finish_epoch(postlude_pending=True)
                    _resume_keras_postlude(
                        exp, training_model, native_callbacks,
                        validation_data=ds_val,
                        validation_steps=fit_kwargs.get("validation_steps"),
                        steps_per_epoch=steps_per_epoch,
                    )
                    history = None
                if exp.state.epoch < target_epoch:
                    history = fit_segment(
                        ds_train,
                        initial_epoch=exp.state.epoch,
                        epochs=target_epoch,
                        batch_offset=0,
                        segment_steps=steps_per_epoch,
                    )
            else:
                if steps_per_epoch is None:
                    # Keras does not reopen an exhausted unknown-cardinality
                    # Dataset for later epochs. Recreate the prepared source per
                    # epoch without scanning it for a length.
                    history = None
                    for epoch in range(start_epoch, target_epoch):
                        history = fit_segment(
                            self._tf_dataset(tf, train_xy, self.training_preparation),
                            initial_epoch=epoch,
                            epochs=epoch + 1,
                            batch_offset=0,
                            segment_steps=None,
                        )
                else:
                    history = fit_segment(
                        ds_train,
                        initial_epoch=start_epoch,
                        epochs=target_epoch,
                        batch_offset=0,
                        segment_steps=steps_per_epoch,
                    )
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
            exp.model.prep_eval()

        if exp.state.epoch < target_epoch and (
            not self._accept_shortened_completion() or exp.state.epoch == start_epoch
        ):
            if fresh_target:
                exp.state.abandon_new_invocation(
                    target_epoch,
                    initial_epoch=start_epoch,
                    initial_step=initial_step,
                )
            raise ValueError("Keras training ended before its retained invocation target.")
        exp.state.finish_invocation(
            target_epoch,
            accept_shortened=self._accept_shortened_completion(),
        )
        return history

    def _capability(self, exp, name, default=None):
        return getattr(exp, "capabilities", {}).get(name, default)

    def _optimizer(self, exp):
        return self.optimizer if self.optimizer is not None else self._capability(exp, "optimizer")

    def _loss(self, exp):
        return self.loss if self.loss is not None else self._capability(exp, "loss")

    def _make_loss(self, tf, exp):
        """Resolve one Keras loss for preflighted mean-loss accounting."""

        loss = _unwrap_backend_obj(self._loss(exp))
        if loss is not None:
            if isinstance(loss, type):
                return validate_class(loss)()
            return loss
        return tf.keras.losses.MeanSquaredError()

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
        if hasattr(model, "obj") and isinstance(model.obj, tf.keras.Model):
            base = model.obj
        else:
            inputs = _keras_inputs_from_spec(tf, x_spec)
            base = tf.keras.Model(inputs=inputs, outputs=model(inputs))
        return _keras_accounting_model(tf, base)

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

    def _tf_dataset(self, tf, data, preparation):
        """Expose explicitly prepared Dataset batches to Keras iteration."""

        def prepared_values():
            cursor = data.iterator()
            try:
                for x, y in cursor:
                    yield preparation.prepare(x, y)
            finally:
                cursor.close()

        return tf.data.Dataset.from_generator(
            prepared_values,
            output_signature=tf_output_signature(preparation.consumer_specs),
        )

    def _accept_shortened_completion(self) -> bool:
        """Return whether this trainer treats Keras's normal early stop as success."""

        return False

    def _callbacks(self, tf):
        return [callback.obj if hasattr(callback, "obj") else callback for callback in self.callbacks]


class Training(BasicTraining):
    """Low-level TensorFlow training loop for arbitrary TF-callable DRYML models."""

    def __call__(self, exp, *, callbacks=()):
        """Train one Experiment with TensorFlow gradient tapes.

        Args:
            exp: Experiment providing the model, data, state, and optional
                optimizer, loss, and metric capabilities.

        Returns:
            Per-batch scalar loss values in training order.

        Raises:
            ValueError: If the data is empty or the model exposes no trainable
                TensorFlow variables, the loss is not mean-reduced, or retained
                deterministic recovery cannot reach its invocation target.
            TypeError: If supplied DRYML callbacks are invalid.
            KeyError: If the model graph lacks a retained runtime binding.

        Side Effects:
            Updates model variables, optimizer state, metrics, progress output,
            and ``exp.state``. Dataset backend preparation completes before the
            tape body; graph traversal uses a temporary explicit Repo only for
            trainable-variable collection.
        """
        import tensorflow as tf

        callbacks = validate_training_callbacks(callbacks)
        loss_fn = self._make_loss(tf, exp)
        _require_mean_keras_loss(loss_fn)
        self._begin_training_preparation_generation()
        train_data = self._prepare_data(exp.train_data, for_training=True)
        require_bounded_safe_points(train_data, callbacks)
        train_xy = self._xy_data(train_data)
        self.training_preparation = TrainingPreparation.from_specs(
            self, train_xy.spec[0], train_xy.spec[1], "tf"
        )

        val_xy = None
        self.validation_preparation = None
        if exp.val_data is not None:
            val_data = self._prepare_data(exp.val_data, for_training=False)
            val_xy = self._xy_data(val_data)
            self.validation_preparation = TrainingPreparation.from_specs(
                self, val_xy.spec[0], val_xy.spec[1], "tf"
            )

        optimizer_wrapper = self._optimizer(exp)
        optimizer = self._make_optimizer(tf, exp)
        metrics = self._metric_objects(exp)
        trainable_variables = None
        losses = []
        steps = 0
        steps_per_epoch = finite_dataset_len(train_data)
        start_epoch = exp.state.epoch
        fresh_target = exp.state.target_epoch is None
        initial_step = exp.state.step
        target_epoch = exp.state.begin_invocation(self.epochs)
        total_steps = None if steps_per_epoch is None else steps_per_epoch * self.epochs
        progress = TrainingProgress(total=total_steps, verbose=self.verbose, desc="TF training")

        try:
            exp.model.prep_train()
            self._finish_pending_epoch(
                tf, exp, val_xy, loss_fn, metrics, progress, target_epoch
            )
            for epoch in range(start_epoch, target_epoch):
                for metric in metrics:
                    _reset_metric(metric)
                epoch_loss = 0.0
                epoch_steps = 0

                cursor = train_xy.iterator()
                resume_batch = exp.state.next_batch if epoch == start_epoch else 0
                if resume_batch:
                    cursor.skip(resume_batch)
                try:
                    for x, y in cursor:
                        # The once-planned U6 handoff executes before this native
                        # differentiation scope begins.
                        x, y = self.training_preparation.prepare(x, y)

                        with tf.GradientTape() as tape:
                            y_pred = exp.model(x)
                            loss_value = tf.reduce_mean(loss_fn(y, y_pred))

                        if trainable_variables is None:
                            with manage_repo() as repo:
                                trainable_variables = _collect_trainable_parameters(exp.model, repo=repo)
                            if not trainable_variables:
                                raise ValueError("TensorFlow model graph exposes no trainable parameters.")
                            if hasattr(optimizer, "build"):
                                optimizer.build(trainable_variables)
                            if hasattr(optimizer_wrapper, "restore_pending"):
                                optimizer_wrapper.restore_pending()

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
                        record_train_update(
                            exp,
                            x,
                            loss_float,
                            batched=self.batch_size is not None,
                            complete_epoch=(
                                steps_per_epoch is not None
                                and exp.state.next_batch + 1 == steps_per_epoch
                            ),
                            callbacks=callbacks,
                        )
                        losses.append(loss_float)
                        epoch_loss += loss_float
                        epoch_steps += 1
                        steps += 1

                        step_metrics = {"loss": loss_float}
                        step_metrics.update(_metric_results(metrics))
                        progress.update(1, step_metrics)
                finally:
                    cursor.close()

                if epoch_steps == 0 and resume_batch == 0:
                    continue

                # A resumed suffix cannot truthfully represent a full epoch mean.
                epoch_metrics = {"loss": epoch_loss / epoch_steps} if epoch_steps and not resume_batch else {}
                if not resume_batch:
                    epoch_metrics.update(_metric_results(metrics))

                self._finish_epoch(
                    tf,
                    exp,
                    epoch,
                    epoch_metrics,
                    val_xy,
                    loss_fn,
                    metrics,
                    progress,
                    target_epoch,
                )
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

        if steps == 0 and target_epoch > start_epoch and exp.state.epoch < target_epoch:
            if fresh_target:
                exp.state.abandon_new_invocation(
                    target_epoch,
                    initial_epoch=start_epoch,
                    initial_step=initial_step,
                )
            raise ValueError("Cannot train on an empty dataset.")
        if exp.state.epoch < target_epoch:
            raise ValueError("TensorFlow training ended before its retained invocation target.")
        exp.state.finish_invocation(target_epoch)

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

    def _finish_pending_epoch(self, tf, exp, val_xy, loss_fn, metrics, progress, target_epoch):
        """Run the validation/progress postlude retained after a final callback."""

        epoch = exp.state.pending_epoch_postlude
        if epoch is None:
            return
        epoch_metrics = dict(exp.state.pending_epoch_metrics or {})
        if exp.state.pending_epoch_postlude_phase == "start":
            if val_xy is not None:
                val_metrics = self._evaluate(tf, exp.model, val_xy, loss_fn, metrics)
                epoch_metrics.update({f"val_{name}": value for name, value in val_metrics.items()})
            exp.state.advance_epoch_postlude(epoch, "progress", metrics=epoch_metrics)
        if exp.state.pending_epoch_postlude_phase == "progress":
            progress.epoch_end(epoch + 1, epochs=target_epoch, metrics=epoch_metrics)
            exp.state.finish_epoch_postlude(epoch)

    def _finish_epoch(self, tf, exp, epoch, epoch_metrics, val_xy, loss_fn, metrics, progress, target_epoch):
        """Complete one explicit-loop epoch without replaying its final update."""

        if exp.state.epoch == epoch:
            exp.state.finish_epoch(postlude_pending=True)
        if exp.state.pending_epoch_postlude == epoch:
            if exp.state.pending_epoch_postlude_phase == "start":
                if val_xy is not None:
                    val_metrics = self._evaluate(tf, exp.model, val_xy, loss_fn, metrics)
                    epoch_metrics.update({f"val_{name}": value for name, value in val_metrics.items()})
                exp.state.advance_epoch_postlude(epoch, "progress", metrics=epoch_metrics)
            if exp.state.pending_epoch_postlude_phase == "progress":
                progress.epoch_end(
                    epoch + 1,
                    epochs=target_epoch,
                    metrics=exp.state.pending_epoch_metrics or epoch_metrics,
                )
                exp.state.finish_epoch_postlude(epoch)
            return
        if val_xy is not None:
            val_metrics = self._evaluate(tf, exp.model, val_xy, loss_fn, metrics)
            epoch_metrics.update({f"val_{name}": value for name, value in val_metrics.items()})
        progress.epoch_end(epoch + 1, epochs=target_epoch, metrics=epoch_metrics)

    def _evaluate(self, tf, model, val_xy, loss_fn, metrics):
        for metric in metrics:
            _reset_metric(metric)
        total_loss = 0.0
        steps = 0
        for x, y in val_xy:
            x, y = self.validation_preparation.prepare(x, y)
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
    """Fit Keras with built-in EarlyStopping and accepted short-target completion.

    Args:
        patience: Completed non-improving epochs tolerated by Keras.
        monitor: Keras history metric used by EarlyStopping.
        restore_best_weights: Whether Keras restores its best observed weights.
        *args: Positional arguments accepted by :class:`BasicTraining`.
        **kwargs: Keyword arguments accepted by :class:`BasicTraining`.

    Raises:
        ValueError: If BasicTraining's accounting/recovery contract is violated.

    Side Effects:
        Installs one Keras EarlyStopping callback. A normal completed-epoch early
        stop clears the retained invocation target rather than reporting a failed
        incomplete invocation.
    """

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

    def _accept_shortened_completion(self) -> bool:
        """Accept Keras's normal completed-epoch early-stop result."""

        return True


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
