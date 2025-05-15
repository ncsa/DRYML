import tensorflow as tf
import numpy as np
from dryml.utils import get_temp_checkpoint_dir, cleanup_checkpoint_dir, \
    adjust_class_module
from dryml import Meta
from dryml.data import Dataset
from dryml.data.tf import TFDataset
from dryml.models.trainable import Trainable as BaseTrainable
from dryml.models.tf.base import Model as TFModel
from dryml.models.tf.base import Trainable as TFTrainable
from dryml.models.tf.base import TrainFunction as TFTrainFunction
from dryml.models.tf.base import Wrapper
from dryml.models.tf.utils import keras_train_spec_updater, \
    keras_callback_wrapper
import tempfile
import zipfile
import os
import re
import shutil
from typing import List
from tensorflow import keras 
from inspect import isclass

def keras_load_checkpoint_from_zip(
        mdl: tf.keras.Model,
        zf: zipfile.ZipFile,
        checkpoint_name: str,
        checkpoint_dir: str = 'checkpoints') -> bool:
    if mdl is None:
        raise RuntimeError(
            "keras model can't be None! "
            "Did you run the object's compute_prepare method?")

    try:
        # Get namelist
        ns = zf.namelist()

        # Detect actual checkpoint name
        checkpoint_name_re = re.compile(f"({checkpoint_name}.*)\\.index")

        checkpoint_name_matches = []
        for n in ns:
            re_res = checkpoint_name_re.search(n)
            if re_res is not None:
                checkpoint_name_matches.append(re_res.groups(1)[0])

        if len(checkpoint_name_matches) > 1:
            print("Warning! More than one checkpoing matched in save file!")

        checkpoint_name = checkpoint_name_matches[-1]

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

        checkpoint_file_re = re.compile(f"{checkpoint_name}.*\\.(index|data)")
        ns = filter(lambda t: checkpoint_file_re.search(t[0]) is not None, ns)

        # Create temp directory
        with tempfile.TemporaryDirectory() as d:
            # Extract files from zipfile to the temp directory
            for orig_n, dest_n in ns:
                zf.extract(orig_n, path=d)

            # Load the weights into the model with the load weights function
            checkpoint_path = os.path.join(d, checkpoint_dir, checkpoint_name)
            # expect to load partial because we may
            # not have optimizer in place yet.
            tf.train.Checkpoint(mdl).restore(checkpoint_path).expect_partial()
    except Exception as e:
        print(f"Issue loading checkpoint! {e}")
        return False
    return True


def keras_load_checkpoint_from_zip_to_dir(
        mdl: tf.keras.Model,
        zf: zipfile.ZipFile,
        checkpoint_name: str,
        temp_checkpoint_dir: str,
        zip_checkpoint_dir: str = 'checkpoints') -> bool:
    if mdl is None:
        raise ValueError(
            "keras model can't be None! "
            "Did you run the object's compute_prepare method?")

    try:
        # Get namelist
        ns = zf.namelist()

        # Detect actual checkpoint name
        checkpoint_name_re = re.compile(f"({checkpoint_name}.*)\\.index")

        checkpoint_name_matches = []
        for n in ns:
            re_res = checkpoint_name_re.search(n)
            if re_res is not None:
                checkpoint_name_matches.append(re_res.groups(1)[0])

        if len(checkpoint_name_matches) > 1:
            print("Warning! More than one checkpoing matched in save file!")

        checkpoint_name = checkpoint_name_matches[-1]

        # Add a slash at the end
        if zip_checkpoint_dir[-1] != '/':
            zip_checkpoint_dir += '/'

        # Get files inside the checkpoint directory
        orig_ns = list(filter(lambda n: n.startswith(zip_checkpoint_dir), ns))
        if len(orig_ns) == 0:
            raise RuntimeError(
                f"No directory {zip_checkpoint_dir} in zipfile!")

        # Get destination names
        dest_ns = list(map(lambda n: n[len(zip_checkpoint_dir):], orig_ns))

        ns = zip(orig_ns, dest_ns)

        checkpoint_file_re = re.compile(f"{checkpoint_name}.*\\.(index|data)")
        ns = filter(lambda t: checkpoint_file_re.search(t[0]) is not None, ns)

        # Extract files from zipfile to the temp directory
        for orig_n, dest_n in ns:
            zf.extract(orig_n, path=temp_checkpoint_dir)

        # Load the weights into the model with the load weights function
        checkpoint_path = os.path.join(
            temp_checkpoint_dir, 'checkpoints', checkpoint_name)

        # expect to load partial because we may
        # not have optimizer in place yet.
        tf.train.Checkpoint(model=mdl) \
                .restore(checkpoint_path) \
                .assert_existing_objects_matched() \
                .expect_partial()
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
            # Load the weights into the model with the load weights function
            checkpoint_path = os.path.join(d, checkpoint_name)

            # Create checkpoint object and save
            checkpoint = tf.train.Checkpoint(mdl)
            checkpoint.save(checkpoint_path)

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


def keras_save_checkpoint_to_zip_from_dir(
        mdl: tf.keras.Model,
        zf: zipfile.ZipFile,
        checkpoint_name: str,
        temp_checkpoint_dir: str,
        zip_checkpoint_dir: str = 'checkpoints') -> bool:
    if mdl is None:
        raise ValueError(
            "keras model can't be None! "
            "Did you run the object's compute_prepare method?")
    try:
        # Adjust checkpoint directory
        if zip_checkpoint_dir[-1] != '/':
            zip_checkpoint_dir += '/'

        # Get namelist
        ns = zf.namelist()

        # Get files inside the checkpoint directory
        ns = filter(lambda n: n.startswith(zip_checkpoint_dir), ns)
        checkpoint_file_re = re.compile(f"{checkpoint_name}\\.(index|data)")
        ns = list(filter(
            lambda n: checkpoint_file_re.search(n) is not None,
            ns))

        # clear existing checkpoint data
        for el_name in os.listdir(temp_checkpoint_dir):
            el_path = os.path.join(temp_checkpoint_dir, el_name)
            if os.path.isdir(el_path):
                shutil.rmtree(el_path)
            else:
                os.remove(el_path)

        # get the full checkpoint path
        checkpoint_path = os.path.join(temp_checkpoint_dir, checkpoint_name)

        # Create checkpoint object and save
        checkpoint = tf.train.Checkpoint(model=mdl)
        checkpoint.save(checkpoint_path)

        # Get list of saved files
        checkpoint_files = os.listdir(temp_checkpoint_dir)

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
                os.path.join(temp_checkpoint_dir, f_name),
                arcname=os.path.join(zip_checkpoint_dir, f_name))
    except Exception as e:
        print(f"Error saving keras weights! {e}")
        return False
    return True


class Model(TFModel):
    def __init__(
            self, *args, **kwargs):
        # It is subclass's responsibility to fill this
        # attribute with an actual keras class
        self.mdl = None

    def __call__(self, X, *args, target=True, index=False, **kwargs):
        return self.mdl(X, *args, **kwargs)

    def compute_cleanup_imp(self):
        # Delete the contained model
        del self.mdl
        self.mdl = None


class ModelWrapper(Model):
    @Meta.collect_args
    @Meta.collect_kwargs
    def __init__(self, cls, *args, **kwargs):
        if not isclass(cls):
            raise TypeError(
                f"Expected first argument to be type. Got {type(cls)}")
        self.cls = cls
        self.args = args
        self.kwargs = kwargs

    def compute_prepare_imp(self):
        self.mdl = self.cls(*self.args, **self.kwargs)


class TrainFunction(TFTrainFunction):
    pass


class BasicTraining(TrainFunction):
    def __init__(
            self, *args, val_split=0.2, val_num=None, num_total=None,
            shuffle_buffer=None, epochs=10000, **kwargs):
        self.val_split = val_split
        self.val_num = val_num
        self.shuffle_buffer = shuffle_buffer
        self.epochs = epochs
        self.num_total = num_total

    def __call__(
            self, trainable, data: Dataset, train_spec=None,
            train_callbacks=[]):

        # Pop the epoch to resume from
        start_epoch = 0
        if train_spec is not None:
            start_epoch = train_spec.level_step()

        # Type checking training data, and converting if necessary
        data = data.tf()

        # Check data is supervised.
        if not data.supervised:
            raise RuntimeError(
                "TFBasicTraining requires supervised data")

        # Make sure data is unbatched. We want the function to control this.
        data = data.unbatch()

        num_total = self.num_total
        if num_total is None:
            # Attempt to measure dataset size
            num_total = data.count()
        if self.val_num is not None:
            num_val = self.val_num
            if self.val_split is not None:
                print("Overridding val_split with val_num.")
            if self.val_num > num_total:
                raise ValueError("val_num cannot be larger than num_total!")
        else:
            num_val = int(num_total*self.val_split)

        num_train = num_total-num_val

        # Get tf.data.Dataset backing type
        ds = data.data()

        if self.shuffle_buffer is None:
            shuffle_buffer = num_train
        else:
            shuffle_buffer = self.shuffle_buffer

        batch_size = self.train_kwargs.pop('batch_size', 32)
        callbacks = self.train_kwargs.pop('callbacks', None)

        ds_val = ds.take(num_val)
        ds_val = ds_val.batch(batch_size)
        ds_val = ds_val.prefetch(tf.data.AUTOTUNE)

        ds_train = ds.skip(num_val)
        ds_train = ds_train.cache()
        ds_train = ds_train.shuffle(shuffle_buffer)
        ds_train = ds_train.batch(batch_size)
        ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

        if callbacks is None:
            callbacks = []

        if train_spec is not None:
            callbacks.append(
                keras_train_spec_updater(train_spec))

        for callback in train_callbacks:
            new_callback = keras_callback_wrapper(callback)
            callbacks.append(new_callback)

        # Fit model
        trainable.model.mdl.fit(
            ds_train, *self.train_args, validation_data=ds_val,
            callbacks=callbacks, epochs=self.epochs,
            initial_epoch=start_epoch, **self.train_kwargs)


class BasicEarlyStoppingTraining(TrainFunction):
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
            self, trainable, data, train_spec=None,
            train_callbacks=[]):

        # Pop the epoch to resume from
        start_epoch = 0
        if train_spec is not None:
            start_epoch = train_spec.level_step()

        # Type checking training data, and converting if necessary
        if type(data) is not TFDataset:
            if not hasattr(data, 'to_TFDataset'):
                raise TypeError(
                    f"Type {type(data)} can't be converted to TFDataset!")
            data = data.to_TFDataset()

        # Check data is supervised.
        if not data.supervised:
            raise RuntimeError(
                "TFBasicEarlyStoppingTraining requires supervised data")

        batch_size = self.train_kwargs.pop('batch_size', 32)
        callbacks = self.train_kwargs.pop('callbacks', None)

        # Make sure data is unbatched. We want the function to control this.
        data = data.unbatch()

        num_total = self.num_total
        if num_total is None:
            # Attempt to measure dataset size
            num_total = data.count()
        if self.val_num is not None:
            num_val = self.val_num
            if self.val_split is not None:
                print("Overridding val_split with val_num.")
            if self.val_num > num_total:
                raise ValueError("val_num cannot be larger than num_total!")
        else:
            num_val = int(num_total*self.val_split)

        num_train = num_total-num_val

        # Get tf.data.Dataset backing type
        ds = data.data()

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

        if callbacks is None:
            callbacks = []

        # Add Early Stopping Callback
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            patience=self.patience,
            monitor='val_loss',
            restore_best_weights=True))

        if train_spec is not None:
            callbacks.append(
                keras_train_spec_updater(train_spec))

        for callback in train_callbacks:
            new_callback = keras_callback_wrapper(callback)
            callbacks.append(new_callback)

        # Fit model
        trainable.model.mdl.fit(
            ds_train, validation_data=ds_val,
            initial_epoch=start_epoch,
            epochs=self.epochs, callbacks=callbacks,
            *self.train_args, **self.train_kwargs)


class SequentialFunctionalModel(Model):
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


def keras_sequential_functional_class(
        name, input_shape, output_shape, base_classes=(Model,)):

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

    # Create the new class
    new_cls = type(name, base_classes, {
        '__init__': __init__,
        'compute_prepare_imp': compute_prepare_imp,
    })

    adjust_class_module(new_cls)

    return new_cls


class Trainable(TFTrainable):
    def __init__(
            self, model: Model = None,
            optimizer: Wrapper = None,
            loss: Wrapper = None,
            metrics: List[Wrapper] = [],
            train_fn: TrainFunction = None):
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

        self.metrics = metrics

        if train_fn is None:
            raise ValueError(
                "You need to set the train_fn component of this trainable!")
        self.train_fn = train_fn

        self._temp_checkpoint_dir = None

    def compute_prepare_imp(self):
        self._temp_checkpoint_dir = get_temp_checkpoint_dir(self.dry_id)

    def compute_cleanup_imp(self):
        if self._temp_checkpoint_dir is not None:
            cleanup_checkpoint_dir(self._temp_checkpoint_dir)
            self._temp_checkpoint_dir = None

    def load_compute_imp(self, file: zipfile.ZipFile) -> bool:
        # Load Weights
        if not keras_load_checkpoint_from_zip_to_dir(
                self.model.mdl, file, 'ckpt-1', self._temp_checkpoint_dir):
            print("Error loading keras weights")
            return False

        return True

    def save_compute_imp(self, file: zipfile.ZipFile) -> bool:

        # Save Weights
        if not keras_save_checkpoint_to_zip_from_dir(
                self.model.mdl, file, 'ckpt', self._temp_checkpoint_dir):
            print("Error saving keras weights")
            return False

        return True

    def train(
            self, data, train_spec=None, train_callbacks=[],
            metrics=[]):
        self.train_fn(
            self, data, train_spec=train_spec,
            train_callbacks=train_callbacks)
        self.train_state = BaseTrainable.trained

    def prep_train(self):
        metric_list = []
        for metric in self.metrics:
            metric_list.append(metric.obj)

        self.model.mdl.compile(
            optimizer=self.optimizer.obj,
            loss=self.loss.obj,
            metrics=metric_list)

    def prep_eval(self):
        pass

# specifically for the setiment analysis
class TrainableVectorizer(TFTrainable):
    def __init__(
            self, model: Model = None, train_fn: TrainFunction = None):
        if model is None:
            raise ValueError(
                "You need to set the model component of this trainable!")
        self.model = model

        if train_fn is None:
            raise ValueError(
                "You need to set the train_fn component of this trainable!")
        self.train_fn = train_fn
        
        self._temp_checkpoint_dir = None

    def compute_prepare_imp(self):
        self._temp_checkpoint_dir = get_temp_checkpoint_dir(self.dry_id)

    def compute_cleanup_imp(self):
        if self._temp_checkpoint_dir is not None:
            cleanup_checkpoint_dir(self._temp_checkpoint_dir)
            self._temp_checkpoint_dir = None

    def load_compute_imp(self, file: zipfile.ZipFile) -> bool:
        vocab_filename = 'vectorizer_vocab.txt'
        try:
            # Extract vocabulary file
            with tempfile.TemporaryDirectory() as d:
                file.extract(vocab_filename, path=d)
                vocab_file_path = os.path.join(d, vocab_filename)
                with open(vocab_file_path, 'r', encoding='utf-8') as f:
                    vocab = [line.strip() for line in f if line.strip() != '']
            self.model.mdl.set_vocabulary(vocab)
        except Exception as e:
            print(f"Error loading vectorizer vocabulary! {e}")
            return False

        return True

    def save_compute_imp(self, file: zipfile.ZipFile) -> bool:
        try:
            vocab = self.model.mdl.get_vocabulary()
            vocab_filename = 'vectorizer_vocab.txt'
            with tempfile.TemporaryDirectory() as d:
                vocab_file_path = os.path.join(d, vocab_filename)
                with open(vocab_file_path, 'w', encoding='utf-8') as f:
                    for word in vocab:
                        f.write(word + '\n')
                file.write(vocab_file_path, arcname=vocab_filename)
        except Exception as e:
            print(f"Error saving vectorizer vocabulary! {e}")
            return False
        return True

    def train(self, data, train_spec=None, train_callbacks=[]):
        self.train_fn(
            self, data, train_spec=train_spec,
            train_callbacks=train_callbacks)
        self.train_state = BaseTrainable.trained


    def prep_train(self):
        pass

    def prep_eval(self):
        pass

    
