from dryml.core.repo import Repo
from dryml.models.trainable import Trainable


class Workshop(object):
    def __init__(self, *args, work_dir=None, create_work_dir=True, **kwargs):
        super().__init__(**kwargs)
        self.repo = Repo(directory=work_dir, create=create_work_dir)

    def data_prep(self):
        raise RuntimeError("Not implemented for base workshop")

    def train_trainable(self, obj: Trainable, *args, **kwargs):
        obj.train(self.train_data, *args, **kwargs)

    def train_trainable_outer(self, obj: Trainable, *args, **kwargs):
        if obj.train_state == Trainable.untrained:
            self.train_trainable(obj, *args, **kwargs)

    def train_models(self, *args,
                     selector=None,
                     sel_args=None, sel_kwargs=None,
                     load_objects: bool = True,
                     only_loaded: bool = False,
                     **kwargs):
        self.repo.apply(self.train_trainable_outer,
                        func_args=args,
                        func_kwargs=kwargs,
                        selector=selector,
                        sel_args=sel_args,
                        sel_kwargs=sel_kwargs,
                        load_objects=load_objects,
                        only_loaded=only_loaded)
