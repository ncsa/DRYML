import os
from dryml.dry_object import DryObject, DryObjectFactory
from dryml.dry_repo import DryRepo
from dryml.dry_component import DryComponent

class Workshop(object):
    def __init__(self, *args, work_dir=None, create_work_dir=True, **kwargs):
        super().__init__(**kwargs)
        self.repo = DryRepo(work_dir, create=create_work_dir)

    def data_prep(self):
        raise RuntimeError("Not implemented for base workshop")

    def train_single_object(self, obj, *args, **kwargs):
        raise RuntimeError("Not implemented for base workshop")

    def train_models(self, *args, selector=None, sel_args=None, sel_kwargs=None, **kwargs):
        for obj in self.repo.get(selector=selector, sel_args=sel_args, sel_kwargs=sel_kwargs):
            if obj.train_state == DryComponent.untrained:
                self.train_single_object(obj, *args, **kwargs)

# toy usage:
#
# Create workshop object
# shop = Workshop()
#
# Load models related to this workshop from a directory
# Options to restrict?
# shop.load_models(directory, **kwargs)
# OR create new models and add them to the workshop
# for model in models:
#     shop.add_model(model)
#
# Method where central data repository is initialized
# shop.data_prep()
#
# Train models (with training options??
# shop.train_models(**kwargs)
#
# Measure model performance
# shop.model_performance()


