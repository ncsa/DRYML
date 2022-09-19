from dryml.data.dataset import Dataset
from dryml.models.trainable import Trainable
from dryml.collections import Tuple


class Pipe(Trainable, Tuple):
    """A Sequential processing pipeline modelled after sklearn pipe"""
    def __init__(self, *args, **kwargs):
        # May need to test that elements of the pipes are trainable too
        pass

    def train(
            self, data, *args, train_spec=None,
            train_callbacks=[], **kwargs):
        # get start epoch (only way to do this is to pass a train_spec.
        # Non-zero start epoch can only happen with no train_spec?
        start_epoch = 0
        if train_spec is not None:
            start_epoch = train_spec.level_step()

        last_val = data
        for i in range(len(self)):
            # Get pipe step
            step = self[i]

            if start_epoch > i or step.train_state == Trainable.trained:
                # This part of the training has already been completed.
                pass
            else:
                # We need to train this step.

                # Manage train spec
                if train_spec is not None:
                    train_spec.descend()

                # Perform training
                step.train(
                    last_val, *args, train_spec=train_spec,
                    train_callbacks=train_callbacks, **kwargs)

                # manage train spec
                if train_spec is not None:
                    train_spec.elevate()

            # Eval data for next step in training
            if i < len(self)-1:
                last_val = step.eval(last_val)

            if train_spec is not None and i >= start_epoch:
                train_spec.advance()

        # Call Trainable Train
        super().train(data, *args, **kwargs)

    def eval(self, X: Dataset, *args, **kwargs):
        last_val = X
        for step in self:
            last_val = step.eval(last_val, *args, **kwargs)

        return last_val

    def prep_train(self):
        for step in self:
            step.prep_train()

    def prep_eval(self):
        for step in self:
            step.prep_eval()
