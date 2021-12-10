import zipfile
from dryml.dry_object import DryObject

class DryComponent(DryObject):
    untrained = 0
    trained = 2
    def __init__(self, *args, description="", dry_args={}, dry_kwargs={}, **kwargs):
        dry_kwargs['description'] = description

        super().__init__(*args, dry_args=dry_args, dry_kwargs=dry_kwargs, **kwargs)
        self.train_state = DryComponent.untrained

    def load_object_imp(self, file: zipfile.ZipFile) -> bool:
        # Helper function to load object specific data should return a boolean indicating if loading was successful
        return super().load_object_imp(file)

    def save_object_imp(self, file: zipfile.ZipFile) -> bool:
        # Helper function to save object specific data should return a boolean indicating if loading was successful
        return super().save_object_imp(file)

    def prepare_data(self, *args, **kwargs):
        raise RuntimeError("Method not defined for a base DryComponent")

    def train(self, train_data, *args, **kwargs):
        raise RuntimeError("Method not defined for a base DryComponent")

    def eval(self, X, *args, **kwargs):
        raise RuntimeError("Method not defined for a base DryComponent")
