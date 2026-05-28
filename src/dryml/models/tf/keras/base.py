from __future__ import annotations

from dryml.models.tf.base import (
    BasicEarlyStoppingTraining,
    BasicTraining,
    Loss,
    Metric,
    Model,
    ModelWrapper,
    Optimizer,
    TrainFunction,
    Wrapper,
)


class SequentialFunctionalModel(Model):
    def __init__(self, input_shape=(1,), layer_defs=(), output_spec=None):
        import tensorflow as tf

        self.input_shape = input_shape
        self.layer_defs = tuple(layer_defs)
        inp = tf.keras.layers.Input(self.input_shape)
        last_layer = inp
        for layer_name, layer_kwargs in self.layer_defs:
            last_layer = getattr(tf.keras.layers, layer_name)(**layer_kwargs)(last_layer)

        self.obj = tf.keras.Model(inputs=inp, outputs=last_layer)
        self.model = self.obj
        self.mdl = self.obj
        self.output_spec = output_spec
        self._pending_restore_path = None
        self._restore_checkpoint = None
        self._restore_status = None


def keras_sequential_functional_class(name, input_shape, output_shape, base_classes=(SequentialFunctionalModel,)):
    def __init__(self, layer_defs, *args, out_activation="linear", output_spec=None, **kwargs):
        import numpy as np
        import tensorflow as tf

        self.layer_defs = tuple(layer_defs)
        self.out_activation = out_activation
        inp = tf.keras.layers.Input(input_shape)
        last_layer = inp
        for layer_name, layer_kwargs in self.layer_defs:
            last_layer = getattr(tf.keras.layers, layer_name)(**layer_kwargs)(last_layer)
        last_layer = tf.keras.layers.Flatten()(last_layer)
        output_units = int(np.prod(output_shape))
        last_layer = tf.keras.layers.Dense(output_units, activation=self.out_activation)(last_layer)
        last_layer = tf.keras.layers.Reshape(output_shape)(last_layer)

        self.obj = tf.keras.Model(inputs=inp, outputs=last_layer)
        self.model = self.obj
        self.mdl = self.obj
        self.output_spec = output_spec
        self._pending_restore_path = None
        self._restore_checkpoint = None
        self._restore_status = None

    new_cls = type(name, base_classes, {"__init__": __init__, "__module__": __name__})
    return new_cls


__all__ = [
    "BasicEarlyStoppingTraining",
    "BasicTraining",
    "Loss",
    "Metric",
    "Model",
    "ModelWrapper",
    "Optimizer",
    "SequentialFunctionalModel",
    "TrainFunction",
    "Wrapper",
    "keras_sequential_functional_class",
]
