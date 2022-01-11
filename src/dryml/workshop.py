from dryml.dry_repo import DryRepo
from dryml.dry_component import DryComponent


class Workshop(object):
    def __init__(self, *args, work_dir=None, create_work_dir=True, **kwargs):
        super().__init__(**kwargs)
        self.repo = DryRepo(work_dir, create=create_work_dir)

    def data_prep(self):
        raise RuntimeError("Not implemented for base workshop")

    def train_component(self, obj: DryComponent, *args, **kwargs):
        if obj.train_state == DryComponent.untrained:
            obj.train(*args, **kwargs)

    def train_models(self, *args,
                     selector=None,
                     sel_args=None, sel_kwargs=None,
                     **kwargs):
        self.repo.apply(self.train_component,
                        func_args=args,
                        func_kwargs=kwargs,
                        selector=selector,
                        sel_args=sel_args,
                        sel_kwargs=sel_kwargs)
