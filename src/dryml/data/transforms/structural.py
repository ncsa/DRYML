from ..dataset import Dataset
from copy import copy
from dryml.core2.backend import Backend
from dryml.core2.utils.recurse import map_leaf_groups, first_leaf
from dryml.core2.tensor_spec import as_tensor_spec
import numpy as np

# Structural transformations that aren't 'simple'
# batch, unbatch, window, shuffle, repeat, take, skip

class Batch(Dataset):
    def __init__(self, src: Dataset, batch_size: int):
        self.batch_size = batch_size
        self.src = src
        self._spec = copy(src.spec)
        self._spec.batch = batch_size

    def __iter__(self):
        backend = None
        if self.spec.batch is None:
            # The dataset is element wise.
            finished = False
            while not finished:
                elements = []
                gen = iter(self.src)
                for _ in range(self.batch_size):
                    try:
                        x = next(gen)
                        if backend is None:
                            backend = as_tensor_spec(first_leaf(x)).backend
                        elements.append(x)
                    except StopIteration:
                        finished = True
                        break

                if len(elements) > 0:
                    if backend == Backend.numpy:
                        yield map_leaf_groups(np.array, elements)
                    elif backend == Backend.tf:
                        yield map_leaf_groups(tf.tensor, elements)
                    else:
                        raise ValueError(f"Batching not available with this backend. {backend}")
