from dryml.dry_config import DryConfig
from dryml.dry_object import DryObject

class DryModel(DryObject):
    untrained = 0
    ready_to_train = 1
    trained = 2
    def __init__(self, *args, description="", dry_args={}, dry_kwargs={}, **kwargs):
        dry_kwargs['description'] = description

        super().__init__(*args, dry_args=dry_args, dry_kwargs=dry_kwargs, **kwargs)
        self.train_state = DryModel.untrained

    def prepare_data(self):
        raise RuntimeError("Method not defined for a base DryModel")

    def train(self, train_data):
        raise RuntimeError("Method not defined for a base DryModel")

    def eval(self, x):
        raise RuntimeError("Method not defined for a base DryModel")

