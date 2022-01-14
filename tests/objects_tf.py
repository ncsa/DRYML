import dryml
import dryml.tf
import tensorflow as tf


class TestTF1(dryml.tf.TFBase):
    def __init__(
            self, *args, in_dim=32,
            dense_layers=[128, 128, 128, 128, 128],
            dry_args=None, dry_kwargs=None, **kwargs):
        dry_kwargs = dryml.utils.init_arg_dict_handler(dry_kwargs)
        dry_kwargs['dense_layers'] = dense_layers
        dry_kwargs['in_dim'] = in_dim
        super().__init__(
            *args,
            dry_args=dry_args,
            dry_kwargs=dry_kwargs,
            **kwargs)

        # Create dummy tensorflow model
        inp = tf.keras.layers.Input((in_dim,))

        last_layer = inp
        for i in range(len(dense_layers)):
            dim = dense_layers[i]
            last_layer = tf.keras.layers.Dense(dim)(last_layer)

        self.mdl = tf.keras.Model(inputs=inp, outputs=last_layer)
