import dryml
import tensorflow_datasets as tfds
import torch
from torch.utils.data import Dataset
from dryml.data.torch import TorchDataset
from dryml.models.torch.text import TextVectorizer
from dryml.models import Pipe
from dryml.metrics.scalar import binary_accuracy

class IMDBTorchDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        return text.decode('utf-8'), torch.tensor(float(label), dtype=torch.float)

class SentimentWorkshop(dryml.Workshop):
    def data_prep(self):
        (ds_train, ds_test), ds_info = tfds.load(
            'imdb_reviews',
            split=['train', 'test'],
            as_supervised=True,
            with_info=True
        )
        train_data = list(ds_train.as_numpy_iterator())
        test_data = list(ds_test.as_numpy_iterator())

        train_torch_ds = IMDBTorchDataset(train_data)
        test_torch_ds = IMDBTorchDataset(test_data)
        self.train_ds = TorchDataset(train_torch_ds, supervised=True)
        self.test_ds = TorchDataset(test_torch_ds, supervised=True)

def train(trainable):
    ws = SentimentWorkshop()
    ws.data_prep()
    
    trainable.prep_train()

    trainable.train(ws.train_ds)
    return trainable


