import tensorflow as tf
from dryml.models import DryTrainable, DryComponent
from dryml.data import DryData
from dryml.data.tf import TFDataset
import tempfile
import zipfile
import os
import re


class TFLikeTrainFunction(DryComponent):
    def __call__(self, trainable, train_data, *args, **kwargs):
        raise NotImplementedError("method must be implemented in a subclass")


class TFBasicTraining(TFLikeTrainFunction):
    def __init__(
            self, *args, val_split=0.2, val_num=None,
            shuffle_buffer=None, num_total=None, **kwargs):
        self.val_split = val_split
        self.val_num = val_num
        self.shuffle_buffer = shuffle_buffer
        self.num_total = num_total

    def __call__(
            self, trainable, data: DryData, *args, batch_size=32,
            callbacks=[], **kwargs):

        # Type checking training data, and converting if necessary
        if type(data) is not TFDataset:
            if not hasattr(data, 'to_TFDataset'):
                raise TypeError(
                    f"Type {type(data)} can't be converted to TFDataset!")
            data = data.to_TFDataset()

        # Check data is supervised.
        if not data.supervised():
            raise RuntimeError(
                "TFBasicTraining requires supervised data")

        # Make sure data is unbatched. We want the function to control this.
        data = data.unbatch()

        # Get tf.data.Dataset backing type
        ds = data.data()

        num_total = self.num_total
        if num_total is None:
            # Attempt to measure dataset size
            num_total = len(ds)
        if self.num_val is not None:
            num_val = self.num_val
            if self.val_split is not None:
                print("Overridding val_split with val_num.")
            if self.num_val > num_total:
                raise ValueError("num_val cannot be larger than num_total!")
        else:
            num_val = int(num_total*self.val_split)

        num_train = num_total-num_val

        if self.shuffle_buffer is None:
            shuffle_buffer = num_train
        else:
            shuffle_buffer = self.shuffle_buffer

        ds_val = ds.take(num_val)
        ds_val = ds_val.batch(batch_size)
        ds_val = ds_val.prefetch(tf.data.AUTOTUNE)

        ds_train = ds.skip(num_val)
        ds_train = ds_train.cache()
        ds_train = ds_train.shuffle(shuffle_buffer)
        ds_train = ds_train.batch(batch_size)
        ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

        # Fit model
        trainable.model.mdl.fit(
            ds_train, validation_data=ds_val,
            callbacks=callbacks, **kwargs)


class TFBasicEarlyStoppingTraining(TFLikeTrainFunction):
    def __init__(
            self, *args, val_split=0.2, val_num=None, num_total=None,
            shuffle_buffer=None, patience=3, epochs=10000, **kwargs):
        self.val_split = val_split
        self.val_num = val_num
        self.shuffle_buffer = shuffle_buffer
        self.patience = patience
        self.epochs = epochs
        self.num_total = num_total

    def __call__(
            self, trainable, data, *args, batch_size=32,
            callbacks=[], **kwargs):

        # Type checking training data, and converting if necessary
        if type(data) is not TFDataset:
            if not hasattr(data, 'to_TFDataset'):
                raise TypeError(
                    f"Type {type(data)} can't be converted to TFDataset!")
            data = data.to_TFDataset()

        # Check data is supervised.
        if not data.supervised():
            raise RuntimeError(
                "TFBasicTraining requires supervised data")

        # Make sure data is unbatched. We want the function to control this.
        data = data.unbatch()

        # Get tf.data.Dataset backing type
        ds = data.data()

        num_total = self.num_total
        if num_total is None:
            # Attempt to measure dataset size
            num_total = len(ds)
        if self.val_num is not None:
            num_val = self.val_num
            if self.val_split is not None:
                print("Overridding val_split with val_num.")
            if self.val_num > num_total:
                raise ValueError("val_num cannot be larger than num_total!")
        else:
            num_val = int(num_total*self.val_split)

        num_train = num_total-num_val

        if self.shuffle_buffer is None:
            shuffle_buffer = num_train
        else:
            shuffle_buffer = self.shuffle_buffer

        ds_val = ds.take(num_val)
        ds_val = ds_val.batch(batch_size)
        ds_val = ds_val.prefetch(tf.data.AUTOTUNE)

        ds_train = ds.skip(num_val)
        ds_train = ds_train.cache()
        ds_train = ds_train.shuffle(shuffle_buffer)
        ds_train = ds_train.batch(batch_size)
        ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

        # Add Early Stopping Callback
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            patience=self.patience,
            monitor='val_loss',
            restore_best_weights=True))

        # Fit model
        trainable.model.mdl.fit(
            ds_train, validation_data=ds_val,
            callbacks=callbacks, epochs=self.epochs, **kwargs)


class TFLikeModel(DryComponent):
    def __call__(self, X, *args, target=True, index=False, **kwargs):
        return self.mdl(X, *args, **kwargs)


class TFLikeTrainable(DryTrainable):
    __dry_compute_context__ = 'tf'

    def __init__(
            self, model: TFLikeModel = None, optimizer=None, loss=None,
            train_fn: TFLikeTrainFunction = None):
        if model is None:
            raise ValueError(
                "You need to set the model component of this trainable!")
        self.model = model

        if optimizer is None:
            raise ValueError(
                "You need to set the optimizer component of this trainable!")
        self.optimizer = optimizer

        if loss is None:
            raise ValueError(
                "You need to set the loss component of this trainable!")
        self.loss = loss

        if train_fn is None:
            raise ValueError(
                "You need to set the train_fn component of this trainable!")
        self.train_fn = train_fn

    def train(self, data, metrics=[], *args, **kwargs):
        # compile the model.
        self.model.mdl.compile(
            optimizer=self.optimizer.obj,
            loss=self.loss.obj,
            metrics=metrics)

        self.train_fn(self, data, *args, **kwargs)

        self.train_state = DryTrainable.trained

    def eval(self, data: DryData, *args, eval_batch_size=32, **kwargs):
        if data.batched():
            # We can execute the method directly on the data
            return data.apply_X(func=lambda X: self.model(X, *args, **kwargs))
        else:
            # We first need to batch the data, then unbatch to leave
            # The dataset character unchanged.
            return data.batch(batch_size=eval_batch_size) \
                       .apply_X(
                            func=lambda X: self.model(X, *args, **kwargs)) \
                       .unbatch()


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


class TFKerasModelBase(TFLikeModel):
    __dry_compute_context__ = 'tf'

    def __init__(
            self, *args, **kwargs):
        # It is subclass's responsibility to fill this
        # attribute with an actual keras class
        self.mdl = None

    def load_compute_imp(self, file: zipfile.ZipFile) -> bool:
        # Load Weights
        if not keras_load_checkpoint_from_zip(self.mdl, file, 'ckpt'):
            print("Error loading keras weights")
            return False

        return True

    def save_compute_imp(self, file: zipfile.ZipFile) -> bool:
        # Save Weights
        if not keras_save_checkpoint_to_zip(self.mdl, file, 'ckpt'):
            print("Error saving keras weights")
            return False

        return True
