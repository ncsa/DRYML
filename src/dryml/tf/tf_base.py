import tensorflow as tf
from dryml import DryComponent
import tempfile
import zipfile
import os
import re


def keras_load_checkpoint_from_zip(
        mdl: tf.keras.Model,
        zf: zipfile.ZipFile,
        checkpoint_name: str,
        checkpoint_dir: str = 'checkpoints') -> bool:
    if mdl is None:
        raise RuntimeError("keras model can't be None!")

    try:
        # Get namelist
        ns = zf.namelist()

        # Add a slash at the end
        if checkpoint_dir[-1] != '/':
            checkpoint_dir += '/'

        # Get files inside the checkpoint directory
        orig_ns = list(filter(lambda n: n.startswith(checkpoint_dir), ns))
        if len(orig_ns) == 0:
            raise RuntimeError(f"No directory {checkpoint_dir} in zipfile!")

        # Get destination names
        dest_ns = list(map(lambda n: n[len(checkpoint_dir):], orig_ns))

        ns = zip(orig_ns, dest_ns)

        checkpoint_file_re = re.compile(f"{checkpoint_name}\\.(index|data)")
        ns = filter(lambda t: checkpoint_file_re.search(t[0]) is not None, ns)

        # Create temp directory
        with tempfile.TemporaryDirectory() as d:
            # Extract files from zipfile to the temp directory
            for orig_n, dest_n in ns:
                zf.extract(orig_n, path=d)

            # Load the weights into the model with the load weights function
            mdl.load_weights(os.path.join(d, checkpoint_dir, checkpoint_name))
    except Exception as e:
        print(f"Issue loading checkpoint! {e}")
        return False
    return True


def keras_save_checkpoint_to_zip(
        mdl: tf.keras.Model,
        zf: zipfile.ZipFile,
        checkpoint_name: str,
        checkpoint_dir: str = 'checkpoints') -> bool:
    try:
        # Adjust checkpoint directory
        if checkpoint_dir[-1] != '/':
            checkpoint_dir += '/'

        # Get namelist
        ns = zf.namelist()

        # Get files inside the checkpoint directory
        ns = filter(lambda n: n.startswith(checkpoint_dir), ns)
        checkpoint_file_re = re.compile(f"{checkpoint_name}\\.(index|data)")
        ns = list(filter(
            lambda n: checkpoint_file_re.search(n) is not None,
            ns))

        # Save checkpoint to a temporary directory
        with tempfile.TemporaryDirectory() as d:
            mdl.save_weights(
                os.path.join(d, checkpoint_name),
                save_format='tf')

            checkpoint_files = os.listdir(d)

            # We would like to delete existing files from the directory
            # But we currently can't do this with python.
            # Instead, we should fail if the files which were written
            # for the new checkpoint don't match those in the
            # zipfile already.
            if len(ns) != 0:
                # List
                for f_name in checkpoint_files:
                    if f_name not in ns:
                        raise RuntimeError(
                            f"Zipfile {zf.filename} already has a checkpoint "
                            f"of name {checkpoint_name} saved. The new "
                            f"checkpoint file {f_name} not already in "
                            "existing checkpoint file set! (python can't "
                            "remove files from open zipfiles)")

            # Write or Overwrite files in the zipfile
            for f_name in checkpoint_files:
                zf.write(
                    os.path.join(d, f_name),
                    arcname=os.path.join(checkpoint_dir, f_name))
    except Exception as e:
        print(f"Error saving keras weights! {e}")
        return False
    return True


class TFBase(DryComponent):
    def __init__(
            self, *args, **kwargs):
        # It is subclass's responsibility to fill this
        # attribute with an actual keras class
        self.mdl = None

    def load_object_imp(self, file: zipfile.ZipFile) -> bool:
        # Load parent components first
        if not super().load_object_imp(file):
            return False

        # Load Weights
        if not keras_load_checkpoint_from_zip(self.mdl, file, 'ckpt'):
            print("Error loading keras weights")
            return False

        return True

    def save_object_imp(self, file: zipfile.ZipFile) -> bool:
        # Save Weights
        if not keras_save_checkpoint_to_zip(self.mdl, file, 'ckpt'):
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
