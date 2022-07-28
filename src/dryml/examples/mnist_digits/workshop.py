import dryml
import tensorflow_datasets as tfds
from dryml.data.tf import TFDataset
from dryml.context import context_check


class MNISTDigitsWorkshop(dryml.Workshop):
    def data_prep(self):
        context_check({'tf': {}})

        (ds_train, ds_test), ds_info = tfds.load(
            'mnist',
            split=['train', 'test'],
            shuffle_files=True,
            as_supervised=True,
            with_info=True)

        self.train_ds = TFDataset(
            ds_train,
            supervised=True)
        self.test_ds = TFDataset(
            ds_test,
            supervised=True)
        self.info_ds = ds_info
