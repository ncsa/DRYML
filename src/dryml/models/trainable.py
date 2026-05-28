from .model import Model
from ..data.dataset import Dataset


class Trainable(Model):
    untrained = 0
    trained = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_state = Trainable.untrained

    def save_state_to_dir_imp(self, dest_dir: str, revision: str|None=None):
        # Save the entire object as a pickle
        pickle_save(
            self.train_state,
            revision_path("train_state", "pkl", dest_dir, revision=revision))

    def restore_state_from_dir_imp(self, src_dir: str, revision: str|None):
        # heavy-state data is stored in heavy.pkl
        self.train_state = pickle_load(
            revision_path("train_state", "pkl", src_dir, revision=revision))

    def prep_train(self):
        # Configure the model for training
        pass

    def prep_eval(self):
        # Configure the model for evaluation
        pass

    def train(self, data: dict[str,Dataset] | Dataset, train_spec=None, train_callbacks=[], **kwargs):
        # Handle the setting of the train state flag
        self.train_state = Trainable.trained
        # This should be the last step in training so no more super is needed
