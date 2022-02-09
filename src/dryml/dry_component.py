import zipfile
import pickle
from dryml.dry_object import DryObject
from dryml.dry_collections import DryTuple


class DryComponent(DryObject):
    untrained = 0
    trained = 2

    def __init__(self, *args, description="", **kwargs):
        print("DryComponent constructor")
        self.train_state = DryComponent.untrained

    def load_object_imp(self, file: zipfile.ZipFile) -> bool:
        # Load parent components first
        if not super().load_object_imp(file):
            return False
        with file.open('component_data.pkl', 'r') as f:
            component_data = pickle.load(f)
        self.train_state = component_data['train_state']
        return True

    def save_object_imp(self, file: zipfile.ZipFile) -> bool:
        with file.open('component_data.pkl', 'w') as f:
            f.write(pickle.dumps({'train_state': self.train_state}))
        return super().save_object_imp(file)

    def prepare_data(self, data, *args, **kwargs):
        # Base component does nothing to input data.
        return data

    def train(self, *args, **kwargs):
        # Handle the setting of the train state flag
        self.train_state = DryComponent.trained
        # This should be the last step in training so no more super is needed

    def eval(self, X, *args, **kwargs):
        raise RuntimeError("Method not defined for a base DryComponent")


class DryPipe(DryComponent, DryTuple):
    def __init__(self, *args, **kwargs):
        for obj in self.data:
            if not isinstance(obj, DryComponent):
                raise ValueError("All stored objects must be DryComponents")

    def load_object_imp(self, file: zipfile.ZipFile):
        return super().load_object_imp(file)

    def save_object_imp(self, file: zipfile.ZipFile):
        return super().save_object_imp(file)

    def train(self, train_data, *args, **kwargs):
        raise RuntimeError("Training of DryPipe not supported currently")

    def eval(self, X, *args, **kwargs):
        results = []
        for component in self:
            results.append(component.eval(X, *args, **kwargs))
        return results
