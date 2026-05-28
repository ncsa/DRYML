from __future__ import annotations

from dryml.core2.object import Object
from dryml.core2.utils.general import pickle_load, pickle_save, revision_path

from .train_spec import TrainState


class Experiment(Object):
    """Runtime training aggregate centered on a model, data, and train function."""

    def __init__(self, model, train_fn, train_data=None, val_data=None, metrics=None, **capabilities):
        super().__init__()
        self.model = model
        self.train_fn = train_fn
        self.train_data = train_data
        self.val_data = val_data
        self.metrics = dict(metrics or {})
        self.capabilities = dict(capabilities)
        self.state = TrainState()

    def train(self):
        return self.train_fn(self)

    def predict(self, x, *args, **kwargs):
        return self.model(x, *args, **kwargs)

    def save_state_to_dir_imp(self, dest_dir: str, revision: str | None = None):
        pickle_save(self.state, revision_path("experiment_state", "pkl", dest_dir, revision=revision))

    def restore_state_from_dir_imp(self, src_dir: str, revision: str | None = None):
        self.state = pickle_load(revision_path("experiment_state", "pkl", src_dir, revision=revision))


__all__ = ["Experiment"]
