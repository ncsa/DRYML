from dryml.tf.tf_base import TFBase
import tensorflow as tf
import numpy as np


def keras_sequential_functional_class(
        name, input_shape, output_shape, base_classes=(TFBase,)):

    def __init__(
            self, layer_defs, *args, out_activation='linear',
            **kwargs):

        # Build Functional Model
        inp = tf.keras.layers.Input(input_shape)
        last_layer = inp
        for layer_name, layer_kwargs in layer_defs:
            last_layer = getattr(
                tf.keras.layers, layer_name)(**layer_kwargs)(last_layer)
        # Initially flatten result
        last_layer = tf.keras.layers.Flatten()(last_layer)
        # Compute number of output units
        output_units = np.cumprod(output_shape)
        last_layer = tf.keras.layers.Dense(
            output_units, activation=out_activation)(last_layer)
        # Respect final shape
        last_layer = tf.keras.layers.Reshape(output_shape)(last_layer)
        self.mdl = tf.keras.Model(inputs=inp, outputs=last_layer)

    # Create the new class
    new_cls = type(name, base_classes, {'__init__': __init__})

    return new_cls
