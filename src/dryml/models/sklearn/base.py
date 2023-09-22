import pickle
import zipfile
import numpy as np
from dryml.core.config import Meta
from dryml.models import Component, Trainable
from dryml.models import TrainFunction as BaseTrainFunction
from dryml.data import Dataset
from dryml.core.utils import validate_class


class Model(Component):
    @Meta.collect_kwargs
    def __init__(self, cls, **kwargs):
        # It is subclass's responsibility to fill this
        # attribute with an actual keras class
        self.mdl = None
        self.cls = validate_class(cls)
        self.mdl_kwargs = kwargs

    def compute_prepare_imp(self):
        self.mdl = self.cls(**self.mdl_kwargs)

    def compute_cleanup_imp(self):
        self.mdl = None

    def load_compute_imp(self, file: zipfile.ZipFile) -> bool:
        pkl_file_name = 'model.pkl'
        # Load Weights
        if pkl_file_name not in file.namelist():
            # No model pickle file right now
            return True
        else:
            with file.open(pkl_file_name, 'r') as f:
                self.mdl = pickle.loads(f.read())
        return True

    def save_compute_imp(self, file: zipfile.ZipFile) -> bool:
        # Save Weights
        if self.mdl is not None:
            pkl_file_name = 'model.pkl'
            with file.open(pkl_file_name, 'w') as f:
                f.write(pickle.dumps(self.mdl))

        return True

    def __call__(self, X, *args, target=True, index=False, **kwargs):
        raise NotImplementedError()


class ClassifierModel(Model):
    def __call__(self, X, *args, target=True, index=False, **kwargs):
        return self.mdl.predict_proba(X, *args, **kwargs)


class RegressionModel(Model):
    def __call__(self, X, *args, target=True, index=False, **kwargs):
        return self.mdl.predict(X, *args, **kwargs)


class BasicTraining(BaseTrainFunction):
    """
    The basic sklearn training method.
    """

    def __init__(
            self, num_examples=-1, shuffle=False,
            shuffle_seed=None, shuffle_buffer_size=None):
        self.num_examples = num_examples
        self.shuffle = shuffle
        self.shuffle_seed = shuffle_seed
        self.shuffle_buffer_size = shuffle_buffer_size
        self.train_args = ()
        self.train_kwargs = {}

    def __call__(
            self, trainable, train_data,
            train_spec=None, train_callbacks=[]):

        train_data = train_data.unbatch()

        total_examples = self.num_examples

        if not train_data.supervised:
            raise ValueError(
                "Dataset must be supervised for basic sklearn training")

        data_examples = total_examples
        if total_examples == -1:
            data_examples = train_data.count()
            if data_examples is np.nan or data_examples is np.inf:
                raise ValueError(
                    "Dataset has no known length and you must provide "
                    "an explicit number of examples for training.")

        if self.shuffle:
            train_data = train_data.shuffle(
                self.shuffle_buffer_size,
                seed=self.shuffle_seed)

        train_data = train_data.batch(batch_size=data_examples) \
                               .as_not_indexed() \
                               .numpy()

        x, y = train_data.peek()

        trainable.mdl.fit(x, y, *self.train_args, **self.train_kwargs)


class Trainable(Trainable):
    __dry_compute_context__ = 'default'

    def __init__(self, model=None, train_fn=None, **kwargs):
        # It is subclass's responsibility to fill this
        # attribute with an actual keras class
        if model is None:
            raise ValueError("Must give a model object")
        self.model = model
        if train_fn is None:
            raise ValueError("Must give a train_fn object")
        self.train_fn = train_fn

    def eval(self, data: Dataset, *args, eval_batch_size=32, **kwargs):
        if data.batched:
            # We can execute the method directly on the data
            return data.numpy().apply_X(
                func=lambda X: self.model(X, *args, **kwargs))
        else:
            # We first need to batch the data, then unbatch to leave
            # The dataset character unchanged.
            return data.numpy().batch(batch_size=eval_batch_size) \
                       .apply_X(
                            func=lambda X: self.model(X, *args, **kwargs)) \
                       .unbatch()

    def train(
            self, data, train_spec=None, train_callbacks=[]):

        self.train_fn(self.model, data, train_spec=train_spec)
