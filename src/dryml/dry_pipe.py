from dryml.dry_component import DryComponent
from dryml.dry_collections import DryList


class DryPipe(DryComponent, DryList):
    """A Sequential processing pipeline modelled after sklearn pipe"""
    def __init__(self, *args, **kwargs):
        pass

    def train(self, data, *args, **kwargs):
        last_val = data
        for i in range(len(self)):
            step = self[i]
            # Prepare data for this step
            last_val = step.prepare_data(last_val)
            # Train step if needed
            if step.train_state == DryComponent.untrained:
                step.train(last_val, *args, **kwargs)
            # Eval data for next step in training
            if i < len(self)-1:
                last_val = step.eval(last_val)

        # Call DryComponent Train
        super().train(data, *args, **kwargs)

    def eval(self, X, *args, **kwargs):
        last_val = X
        for step in self:
            last_val = step.prepare_data(last_val)
            last_val = step.eval(last_val, *args, **kwargs)

        return last_val
