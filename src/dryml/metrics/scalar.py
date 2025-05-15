from dryml.data import Dataset
from dryml.models import Trainable
from dryml.context import compute_context
import numpy as np


@compute_context(ctx_dont_create_context=True)
def mean_squared_error(trainable: Trainable, test_data: Dataset):
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
def categorical_accuracy(trainable: Trainable, test_data: Dataset):
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


@compute_context(ctx_dont_create_context=True)
def binary_f1_score(trainable: Trainable, test_data: Dataset):
    if not test_data.supervised:
        raise ValueError("Dataset is unsupervised!")
    if not test_data.batched:
        data = test_data.batch()

    # Prepare the input data, expects a dataset
    data = trainable.eval(data) \
                    .as_not_indexed()
    true_neg = 0
    true_pos = 0
    false_neg = 0
    false_pos = 0
    for y_pred, y in data.numpy():
        for i in range(y.shape[0]):
            for j in range(y.shape[1]):
                if y[i][j] == 0:
                    if y_pred[i][j] == 0:
                        true_neg += 1
                    else:
                        false_pos += 1
                else:
                    if y_pred[i][j] == 0:
                        false_neg += 1
                    else:
                        true_pos += 1
    if true_pos == 0 and false_pos == 0:
        recall = 0
    else:
        recall = true_pos / (true_pos + false_pos)
    if true_pos == 0 and false_neg == 0:
        precision = 0
    else:
        precision = true_pos / (true_pos + false_neg)
    if precision == 0 and recall == 0:
        f1_score = 0
    else:
        f1_score = 2*precision*recall / (precision + recall)
    return f1_score

@compute_context(ctx_dont_create_context=True)
def binary_accuracy(trainable: Trainable, test_data: Dataset, threshold=0.5):
    if not test_data.supervised:
        raise ValueError("Dataset is unsupervised!")

    data = trainable.eval(test_data).as_not_indexed()

    if not data.batched:
        data = data.batch()

    total_correct = 0
    total_samples = 0

    for batch in data.numpy():
        batch_pred, batch_true = batch

        batch_pred = batch_pred
        batch_true = batch_true

        # Convert predictions to binary
        batch_pred_binary = (batch_pred >= threshold).astype(np.int32)
        batch_pred_binary = batch_pred_binary.flatten()
        batch_true = batch_true.flatten().astype(np.int32)

        total_correct += np.sum(batch_pred_binary == batch_true)
        total_samples += len(batch_true)

    return total_correct / total_samples
