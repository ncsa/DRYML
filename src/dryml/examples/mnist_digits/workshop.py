import dryml
from dryml.data import TFDSAdapter


class MNISTDigitsWorkshop(dryml.Workshop):
    @dryml.env.req(packages={"tensorflow-datasets": None})
    @dryml.world.req(cpus={"min": 1})
    @dryml.world.default(cpus=1)
    @dryml.runtime.default(mode="worker", device_visibility={"policy": "assigned"})
    def data_prep(self):
        self.train_ds = TFDSAdapter("mnist", split="train", as_supervised=True)
        self.test_ds = TFDSAdapter("mnist", split="test", as_supervised=True)
