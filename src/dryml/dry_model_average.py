import pickle
import zipfile
from dryml.dry_object import load_object
from dryml.dry_component import DryComponent
from dryml.utils import pickler


class DryModelAverage(DryComponent):
    def __init__(self, *args, dry_args=None, dry_kwargs=None, **kwargs):
        self.components = []
        super().__init__(
            *args,
            dry_args=dry_args, dry_kwargs=dry_kwargs,
            **kwargs)

    def add_component(self, component):
        self.components.append(component)

    def load_object_imp(self, file: zipfile.ZipFile) -> bool:
        # Load super classes information
        if not super().load_object_imp(file):
            return False

        # Load component list
        with file.open('component_list.pkl', mode='r') as f:
            component_filenames = pickle.loads(f.read())

        # Load individual components
        for filename in component_filenames:
            with file.open(filename, mode='r') as f:
                self.components.append(load_object(f))

        return True

    def save_object_imp(self, file: zipfile.ZipFile) -> bool:
        # We save each component inside the file first.
        component_filenames = []
        for component in self.components:
            filename = f"{component.get_individual_hash()}.dry"
            with file.open(filename, mode='w') as f:
                component.save_self(f)
            component_filenames.append(filename)

        with file.open('component_list.pkl', mode='w') as f:
            f.write(pickler(component_filenames))

        # Super classes should save their information
        return super().save_object_imp(file)

    def prepare_data(self, data, *args, **kwargs):
        # The model average doesn't do any data transformation itself.
        return data

    def train(self, train_data, *args, **kwargs):
        raise RuntimeError("All models should be trained individually")

    def eval(self, X, *args, **kwargs):
        results = []
        for mdl_component in self.components:
            # Transform input data to be fit for the model
            X_trans = mdl_component(X)
            # Evaluate model on data
            results.append(mdl_component.eval(X_trans))
