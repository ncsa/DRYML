import numpy as np
from dryml.data import NumpyDataset


def test_numpy_dataset_1():
    batch_size = 10

    data = np.random.random((batch_size, 20))

    dataset = NumpyDataset(data, batch_size=batch_size)

    i = 0
    for el in dataset.unbatch():
        assert np.all(el == data[i])
        i += 1


def test_numpy_dataset_2():
    batch_size = 10

    data = np.random.random((batch_size, 20))

    dataset = NumpyDataset(data)

    i = 0
    for el in dataset.unbatch():
        assert np.all(el == data[i])
        i += 1


def test_numpy_dataset_3():
    batch_size = 10

    data = np.random.random((batch_size, 20))

    dataset = NumpyDataset(data, batch_size=batch_size)

    v = next(iter(dataset))

    assert np.all(v == data)


def test_numpy_dataset_4():
    batch_size = 10

    data = np.random.random((batch_size, 20))

    dataset = NumpyDataset(data)

    v = next(iter(dataset))
    assert np.all(v == data)


def test_numpy_dataset_5():
    batch_size = 32

    batch_rows = []
    for i in range(batch_size):
        batch_rows.append(np.random.random((20,)))

    dataset = NumpyDataset(batch_rows)

    i = 0
    for el in dataset:
        assert np.all(el == batch_rows[i])
        i += 1


def test_numpy_dataset_6():
    batch_size = 32

    batch_rows = []
    for i in range(batch_size):
        batch_rows.append(np.random.random((20,)))

    dataset = NumpyDataset(batch_rows)

    v = next(iter(dataset.batch(batch_size=batch_size)))

    all_data = np.stack(batch_rows, axis=0)

    assert np.all(v == all_data)


def test_numpy_dataset_7():
    batch_size = 32

    batch_rows = []
    for i in range(batch_size):
        batch_rows.append(np.random.random((20,)))

    dataset = NumpyDataset(batch_rows)

    sub_batch_size = batch_size//4

    i = 0
    for el in dataset.batch(batch_size=sub_batch_size):
        sub_batch = np.stack(
            batch_rows[i*sub_batch_size:(i+1)*sub_batch_size], axis=0)
        assert np.all(el == sub_batch)
        i += 1


def test_numpy_dataset_8():
    batch_size = 32

    batch_rows = []
    for i in range(batch_size):
        batch_rows.append(np.random.random((20,)))

    dataset = NumpyDataset(batch_rows)

    dataset2 = dataset.batch(batch_size//4).unbatch()

    for el, el2 in zip(dataset, dataset2):
        assert np.all(el == el2)
