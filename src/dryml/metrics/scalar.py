from dryml.data import Dataset
from dryml.models import DryTrainable
from dryml.context import compute_context
import numpy as np


@compute_context(ctx_dont_create_context=True)
def mean_squared_error(trainable: DryTrainable, test_data: Dataset):
    if not test_data.supervised:
        raise ValueError("Dataset is unsupervised!")

    # Prepare input data, expects a dataset
    data = trainable.eval(test_data) \
                    .as_not_indexed()

    if not data.batched:
        data = data.batch()

    data = data.numpy()

    total_loss = 0.
    num_examples = 0
    for batch_e_y, batch_y in data:
        total_loss += np.sum((batch_e_y-batch_y)**2)
        num_examples += batch_e_y.shape[0]

    return total_loss/num_examples


@compute_context(ctx_dont_create_context=True)
def categorical_accuracy(trainable: DryTrainable, test_data: Dataset):
    if not test_data.supervised:
        raise ValueError("Dataset is unsupervised!")

    # Prepare the input data, expects a dataset
    data = trainable.eval(test_data) \
                    .as_not_indexed()

    if not data.batched:
        data = data.batch()

    num_correct = 0
    num_total = 0

    for Y_eval, Y in data.numpy():
        num_correct += (Y_eval == Y).sum()
        num_total += len(Y)

    return num_correct/num_total
