from .model import Model


class Pipe(Model):
    """A Sequential processing pipeline modelled after sklearn pipe"""
    def __init__(self, *args):
        self.components = args

    def __iter__(self):
        return iter(self.components)

    def __getitem__(self, key):
        return self.components[key]

    def __call__(self, X):
        last_val = X
        for step in self:
            last_val = step(last_val)

        return last_val

    def prep_train(self):
        for step in self:
            if hasattr(step, 'prep_train'):
                step.prep_train()

    def prep_eval(self):
        for step in self:
            if hasattr(step, 'prep_eval'):
                step.prep_eval()

    def infer_output_spec(self, input_spec):
        spec = input_spec
        for step in self:
            spec = step.infer_output_spec(spec)
        return spec
