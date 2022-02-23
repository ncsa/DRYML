from dryml.data.dry_data import DryData
from dryml.models.dry_trainable import DryTrainable
from dryml.dry_collections import DryTuple


class DryPipe(DryTrainable, DryTuple):
    """A Sequential processing pipeline modelled after sklearn pipe"""
    def __init__(self, *args, **kwargs):
        pass

    def train(self, data, *args, **kwargs):
        last_val = data
        for i in range(len(self)):
            step = self[i]
            # Train step if needed
            if step.train_state == DryTrainable.untrained:
                step.train(last_val, *args, **kwargs)
            # Eval data for next step in training
            if i < len(self)-1:
                last_val = step.eval(last_val)

        # Call DryTrainable Train
        super().train(data, *args, **kwargs)

    def eval(self, X: DryData, *args, **kwargs):
        last_val = X
        for step in self:
            last_val = step.eval(last_val, *args, **kwargs)

        return last_val