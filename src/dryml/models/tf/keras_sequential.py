from dryml.models.tf.tf_base import TFKerasModelBase
import tensorflow as tf
import numpy as np
from dryml.utils import adjust_class_module


class KerasSequentialFunctionalModel(TFKerasModelBase):
    def __init__(
        self, input_shape=(1,), layer_defs=[]):

        self.input_shape = input_shape
        self.layer_defs = layer_defs

    def compute_prepare_imp(self):
        # Build Functional Model
        inp = tf.keras.layers.Input(self.input_shape)
        last_layer = inp
        for layer_name, layer_kwargs in self.layer_defs:
            last_layer = getattr(
                tf.keras.layers, layer_name)(**layer_kwargs)(last_layer)
        self.mdl = tf.keras.Model(inputs=inp, outputs=last_layer)

    def compute_cleanup_imp(self):
        # Delete the contained model
        del self.mdl
        self.mdl = None


def keras_sequential_functional_class(
        name, input_shape, output_shape, base_classes=(TFKerasModelBase,)):

    def __init__(
            self, layer_defs, *args, out_activation='linear',
            **kwargs):
        self.layer_defs = layer_defs
        self.out_activation = out_activation

    def compute_prepare_imp(self):
        # Build Functional Model
        inp = tf.keras.layers.Input(input_shape)
        last_layer = inp
        for layer_name, layer_kwargs in self.layer_defs:
            last_layer = getattr(
                tf.keras.layers, layer_name)(**layer_kwargs)(last_layer)
        # Initially flatten result
        last_layer = tf.keras.layers.Flatten()(last_layer)
        # Compute number of output units
        output_units = np.cumprod(output_shape)
        last_layer = tf.keras.layers.Dense(
            output_units, activation=self.out_activation)(last_layer)
        # Respect final shape
        last_layer = tf.keras.layers.Reshape(output_shape)(last_layer)
        self.mdl = tf.keras.Model(inputs=inp, outputs=last_layer)

    def compute_cleanup_imp(self):
        # Delete the contained model
        del self.mdl
        self.mdl = None

    # Create the new class
    new_cls = type(name, base_classes, {
        '__init__': __init__,
        'compute_prepare_imp': compute_prepare_imp,
        'compute_cleanup_imp': compute_cleanup_imp,
    })

    adjust_class_module(new_cls)

    return new_cls
