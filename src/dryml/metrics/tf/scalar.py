from dryml.data import DryData
from dryml.context import compute_context


@compute_context(ctx_dont_create_context=True)
def category_accuracy(trainable, test_data: DryData):
    if not test_data.supervised():
        raise ValueError("Dataset is unsupervised!")

    # Prepare the input data, expects a dataset
    data = trainable.eval(test_data) \
                    .as_not_indexed()

    if not data.batched():
        data = data.batch(batch_size=32)

    num_correct = 0
    num_total = 0

    for Y_eval, Y in data:
        num_correct += (Y_eval == Y).numpy().sum()
        num_total += len(Y)

    return num_correct/num_total
