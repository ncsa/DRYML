import tensorflow as tf
from dryml import DryComponent
import tempfile
import zipfile
import os


def keras_load_weights_from_zip(
        mdl, zipfile: zipfile.ZipFile,
        filename: str) -> bool:
    if mdl is None:
        raise RuntimeError("keras model can't be None!")

    try:
        with tempfile.NamedTemporaryFile(mode='w+b') as f:
            # Copy the zipfile saved weights file to a named temp file
            with zipfile.open(filename) as zf:
                f.write(zf.read())
            # Flush data to disk
            f.flush()

            # Load the weights into the model with the load weights function
            mdl.load_weights(f.name)
            return True
    except Exception as e:
        print(f"Issue loading weights! {e}")
        return False


def keras_save_weights_to_zip(
        mdl: tf.keras.Model, zipfile: zipfile.ZipFile,
        filename: str) -> bool:
    try:
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as d:
            # Save weights to a file
            temp_weights_path = os.path.join(d, filename)
            mdl.save_weights(temp_weights_path)

            # Copy file into the zipfile
            with zipfile.open(filename, 'w') as zf:
                with open(temp_weights_path, 'rb') as f:
                    zf.write(f.read())
                    return True
    except Exception as e:
        print(f"Error saving keras weights! {e}")
        return False


class TFBase(DryComponent):
    def __init__(
            self, *args, dry_args=None,
            dry_kwargs=None, **kwargs):
        # It is subclass's responsibility to fill this
        # attribute with an actual keras class
        self.mdl = None
        super().__init__(
            *args,
            dry_args=dry_args,
            dry_kwargs=dry_kwargs,
            **kwargs)

    def load_object_imp(self, file: zipfile.ZipFile) -> bool:
        # Load parent components first
        if not super().load_object_imp(file):
            return False

        # Load Weights
        if not keras_load_weights_from_zip(self.mdl, file, 'weights.hdf5'):
            print("Error loading keras weights")
            return False

        return True

    def save_object_imp(self, file: zipfile.ZipFile) -> bool:
        # Save Weights
        if not keras_save_weights_to_zip(self.mdl, file, 'weights.hdf5'):
            return False

        # Save parent components
        return super().save_object_imp(file)

    def train(self, data, *args, val_split=0.2, batch_size=32,
              shuffle_buffer=None, callbacks=[], **kwargs):
        # Here, we provide a basic training algorithm which
        # will work in many TF model cases
        data = self.prepare_data(data, target=True, index=False)
        # Check if we need to create a dataset
        if not isinstance(data, tf.data.Dataset):
            ds = tf.data.Dataset.from_tensor_slices(data)
        else:
            ds = data
        num_total = len(ds)
        num_val = int(num_total*val_split)
        num_train = num_total-num_val
        if shuffle_buffer is None:
            shuffle_buffer = num_train

        ds_val = ds.take(num_val)
        ds_val = ds_val.batch(batch_size)
        ds_val = ds_val.prefetch(tf.data.AUTOTUNE)

        ds_train = ds.skip(num_val)
        ds_train = ds_train.cache()
        ds_train = ds_train.shuffle(shuffle_buffer)
        ds_train = ds_train.batch(batch_size)
        ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

        # Fit model
        self.mdl.fit(
            ds_train, validation_data=ds_val,
            callbacks=callbacks, **kwargs)

        # Finalization of training
        super().train()

    def eval(self, X, *args, **kwargs):
        return self.mdl(X, *args, **kwargs)
