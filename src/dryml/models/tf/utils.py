import tensorflow as tf


class keras_train_spec_updater(tf.keras.callbacks.Callback):
    def __init__(self, train_spec, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_spec = train_spec

    def on_epoch_end(self, epoch, logs=None):
        # Advance the train spec at end of an epoch
        self.train_spec.advance()


class keras_callback_wrapper(tf.keras.callbacks.Callback):
    def __init__(self, callback, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.callback = callback

    def on_epoch_end(self, epoch, logs=None):
        # Call the callback at the end of the epoch
        self.callback()
