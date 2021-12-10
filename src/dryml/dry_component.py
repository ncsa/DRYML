import zipfile
import pickle
from dryml.dry_object import DryObject

class DryComponent(DryObject):
    untrained = 0
    trained = 2
    def __init__(self, *args, description="", dry_args={}, dry_kwargs={}, **kwargs):
        dry_kwargs['description'] = description

        super().__init__(*args, dry_args=dry_args, dry_kwargs=dry_kwargs, **kwargs)
        self.train_state = DryComponent.untrained

    def load_object_imp(self, file: zipfile.ZipFile) -> bool:
        with file.open('component_data.pkl', 'r') as f:
            component_data = pickle.load(f)
        self.train_state = component_data['train_state']
        return super().load_object_imp(file)

    def save_object_imp(self, file: zipfile.ZipFile) -> bool:
        with file.open('component_data.pkl', 'r') as f:
            pickle.dump({'train_state': self.train_state}, f)
        return super().save_object_imp(file)

    def prepare_data(self, data, *args, **kwargs):
        raise RuntimeError("Method not defined for a base DryComponent")

    def train(self, train_data, *args, **kwargs):
        raise RuntimeError("Method not defined for a base DryComponent")

    def eval(self, X, *args, **kwargs):
        raise RuntimeError("Method not defined for a base DryComponent")
