import pickle
import zipfile
import numpy as np
from dryml.dry_config import DryMeta
from dryml.models import DryComponent, DryTrainable
from dryml.data import DryData


class SklearnLikeModel(DryComponent):
    @DryMeta.collect_kwargs
    def __init__(self, cls, **kwargs):
        # It is subclass's responsibility to fill this
        # attribute with an actual keras class
        self.mdl = None
        self.cls = cls
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


class SklearnClassifierModel(SklearnLikeModel):
    def __call__(self, X, *args, target=True, index=False, **kwargs):
        return self.mdl.predict_proba(X, *args, **kwargs)


class SklearnRegressionModel(SklearnLikeModel):
    def __call__(self, X, *args, target=True, index=False, **kwargs):
        return self.mdl.predict(X, *args, **kwargs)


class SklearnLikeTrainFunction(DryComponent):
    def __init__(self):
        self.train_args = ()
        self.train_kwargs = {}

    def __call__(
            self, trainable, train_data, train_spec=None, train_callbacks=[]):
        raise NotImplementedError("method must be implemented in a subclass")


class SklearnBasicTraining(SklearnLikeTrainFunction):
    """
    The basic sklearn training method.
    """

    def __init__(
            self, num_examples=-1):
        self.num_examples = num_examples
        self.train_args = ()
        self.train_kwargs = {}

    def __call__(
            self, trainable, train_data,
            train_spec=None, train_callbacks=[]):

        train_data = train_data.unbatch()

        total_examples = self.num_examples

        data_examples = total_examples
        if total_examples == -1:
            data_examples = len(train_data)
            if data_examples is np.nan or data_examples is np.inf:
                raise ValueError(
                    "Dataset has no known length and you must provide "
                    "an explicit number of examples for training.")

        train_data = train_data.batch(batch_size=data_examples) \
                               .as_not_indexed() \
                               .numpy()

        if not train_data.supervised:
            raise ValueError(
                "Dataset must be supervised for basic sklearn training")

        x, y = train_data.peek()

        trainable.mdl.fit(x, y, *self.train_args, **self.train_kwargs)


class SklearnTrainable(DryTrainable):
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

    def eval(self, data: DryData, *args, eval_batch_size=32, **kwargs):
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
            self, data, *args, train_spec=None, train_callbacks=[], **kwargs):

        self.train_fn(self.model, data, train_spec=train_spec, **kwargs)
