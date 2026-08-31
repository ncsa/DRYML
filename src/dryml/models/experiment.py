from __future__ import annotations

from dryml.core.object import Serializable
from dryml.core.utils.general import pickle_load, pickle_save

from .train_spec import TrainState


class Experiment(Serializable):
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
        self.state.phase = TrainState.training
        try:
            result = self.train_fn(self)
        except Exception:
            self.state.phase = TrainState.failed
            raise
        if self.state.phase == TrainState.training:
            self.state.phase = TrainState.trained
        return result

    def save_state_to_dir_imp(self, dest_dir: str, *, codec: str) -> None:
        pickle_save(self.state, f"{dest_dir}/experiment_state.pkl")

    def restore_state_from_dir_imp(self, src_dir: str, *, codec: str) -> None:
        self.state = pickle_load(f"{src_dir}/experiment_state.pkl")


__all__ = ["Experiment"]
