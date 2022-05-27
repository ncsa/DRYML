import os
import dryml
import pathlib

class dryml_object_saver(object):
    def __init__(self, obj, checkpoint_dir_base, train_state):
        self.obj = obj
        self.checkpoint_dir_base = checkpoint_dir_base
        self.train_state = train_state

    def __call__(self):
        checkpoint_dir = f"{self.checkpoint_dir_base}_{self.train_state.global_step()}"
        pathlib.Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        self.obj.save_self(f"{checkpoint_dir}/model.dry")
        self.train_state.save(f"{checkpoint_dir}/train_state.pkl")
