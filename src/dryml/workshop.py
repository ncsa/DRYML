from dryml.dry_repo import DryRepo
from dryml.dry_component import DryComponent


class Workshop(object):
    def __init__(self, *args, work_dir=None, create_work_dir=True, **kwargs):
        super().__init__(**kwargs)
        self.repo = DryRepo(directory=work_dir, create=create_work_dir)

    def data_prep(self):
        raise RuntimeError("Not implemented for base workshop")

    def train_component(self, obj: DryComponent, *args, **kwargs):
        obj.train(self.train_data, *args, **kwargs)

    def train_component_outer(self, obj: DryComponent, *args, **kwargs):
        if obj.train_state == DryComponent.untrained:
            self.train_component(obj, *args, **kwargs)

    def train_models(self, *args,
                     selector=None,
                     sel_args=None, sel_kwargs=None,
                     load_objects: bool = True,
                     only_loaded: bool = False,
                     **kwargs):
        self.repo.apply(self.train_component_outer,
                        func_args=args,
                        func_kwargs=kwargs,
                        selector=selector,
                        sel_args=sel_args,
                        sel_kwargs=sel_kwargs,
                        load_objects=load_objects,
                        only_loaded=only_loaded)
