import os
from dryml.dry_object import DryObject, DryObjectFactory
from dryml.dry_component import DryComponent

class Workshop(object):
    def __init__(self, work_dir, *args, create_work_dir=True, **kwargs):
        super().__init__(**kwargs)
        if not os.path.exists(work_dir):
            if create_work_dir:
                os.mkdir(work_dir)
            else:
                raise RuntimeError(f"Workshop directory {work_dir} doesn't exist!")

        self.work_dir = work_dir
        self.single_models = []

    def data_prep(self):
        return

    def add_model_from_factory(self, factory: DryObjectFactory, label: str):
        self.single_models.append(
            {'model': factory(),
             'label': label}
        )

    def train_single_models(self, **kwargs):
        for model_dict in self.single_models:
            model = model_dict['model']
            if model.train_state == DryComponent.untrained:
                model.train(self.train_ds, **kwargs)

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


