import dryml
import dryml.models.tf
import tensorflow as tf


class TestTF1(dryml.models.tf.keras.Model):
    def __init__(
            self, *args, in_dim=32,
            dense_layers=[128, 128, 128, 128, 128], **kwargs):
        self.in_dim = in_dim
        self.dense_layers = dense_layers

    def compute_prepare_imp(self):
        # Build Functional Model
        inp = tf.keras.layers.Input((self.in_dim,))

        last_layer = inp
        for i in range(len(self.dense_layers)):
            dim = self.dense_layers[i]
            last_layer = tf.keras.layers.Dense(dim)(last_layer)

        self.mdl = tf.keras.Model(inputs=inp, outputs=last_layer)

    def compute_cleanup_imp(self):
        # Delete the contained model
        del self.mdl
        self.mdl = None
