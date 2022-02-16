import zipfile
import pickle
from dryml.dry_object import DryObject
from dryml.dry_collections import DryTuple


class DryTrainable(DryObject):
    untrained = 0
    trained = 2

    def __init__(self, *args, description="", **kwargs):
        self.train_state = DryTrainable.untrained

    def load_object_imp(self, file: zipfile.ZipFile) -> bool:
        # Load parent components first
        with file.open('component_data.pkl', 'r') as f:
            component_data = pickle.load(f)
        self.train_state = component_data['train_state']
        return True

    def save_object_imp(self, file: zipfile.ZipFile) -> bool:
        with file.open('component_data.pkl', 'w') as f:
            f.write(pickle.dumps({'train_state': self.train_state}))
        return True

    def prepare_data(self, data, *args, **kwargs):
        # Base component does nothing to input data.
        return data

    def train(self, *args, **kwargs):
        # Handle the setting of the train state flag
        self.train_state = DryTrainable.trained
        # This should be the last step in training so no more super is needed

    def eval(self, X, *args, **kwargs):
        raise RuntimeError("Method not defined for a base DryTrainable")


class DryPipe(DryTrainable, DryTuple):
    def __init__(self, *args, **kwargs):
        for obj in self.data:
            if not isinstance(obj, DryTrainable):
                raise ValueError("All stored objects must be DryTrainables")

    def train(self, train_data, *args, **kwargs):
        raise RuntimeError("Training of DryPipe not supported currently")

    def eval(self, X, *args, **kwargs):
        results = []
        for trainable in self:
            results.append(trainable.eval(X, *args, **kwargs))
        return results
