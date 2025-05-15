from dryml.models.torch.generic import Trainable
import torch
from collections import Counter
import re
import numpy as np

# Text Vectorizer for Sentiment Analysis Model
class TextVectorizer(Trainable):
    def __init__(self, max_tokens=10000, sequence_length=250, pad_token="<PAD>", unk_token="<UNK>"):
        super().__init__()
        self.max_tokens = max_tokens
        self.sequence_length = sequence_length
        self.pad_token = pad_token
        self.unk_token = unk_token

        self.vocab = {}
        self.id_to_token = []
        self.trained = False

    def simple_tokenize(self, text):
        return re.findall(r"\w+", text.lower())

    def build_vocab(self, dataset):
        counter = Counter()
        for x, _ in dataset:
            tokens = self.simple_tokenize(x)
            counter.update(tokens)
        most_common = counter.most_common(self.max_tokens - 2)
        self.id_to_token = [self.pad_token, self.unk_token] + [w for w, _ in most_common]
        self.vocab = {w: i for i, w in enumerate(self.id_to_token)}

    def vectorize_text(self, text):
        tokens = self.simple_tokenize(text)
        ids = []
        for t in tokens:
            ids.append(self.vocab.get(t, self.vocab[self.unk_token]))
        if len(ids) < self.sequence_length:
            ids += [self.vocab[self.pad_token]] * (self.sequence_length - len(ids))
        else:
            ids = ids[:self.sequence_length]
        return np.array(ids, dtype=np.int64)

    def train(self, ds, **kwargs):
        if not self.trained:
            self.build_vocab(ds)
            self.trained = True

    def predict(self, x, **kwargs):
        if isinstance(x, str):
            return torch.tensor(self.vectorize_text(x), dtype=torch.long)
        elif isinstance(x, np.ndarray):
            return self.predict(x.tolist(), **kwargs)
        elif isinstance(x, (list, tuple)):
            return torch.stack([
                torch.tensor(self.vectorize_text(xx), dtype=torch.long)
                for xx in x
            ])
        raise ValueError("Unsupported input type for vectorizer.")


    def eval(self, dataset, **kwargs):
        return dataset.map(lambda x: (self.predict(x[0]), x[1]))

    def prep_train(self, **kwargs):
        pass

    def save_model(self, *args, **kwargs):
        return {
            'vocab': self.vocab,
            'id_to_token': self.id_to_token,
            'trained': self.trained
        }

    def load_model(self, dct, *args, **kwargs):
        self.vocab = dct['vocab']
        self.id_to_token = dct['id_to_token']
        self.trained = dct['trained']
