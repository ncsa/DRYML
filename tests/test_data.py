import numpy as np
from dryml.data import NumpyDataset
from dryml.data import util


def test_data_util_slicer_1():
    array_shape = (10, 20)
    data1 = np.random.random(array_shape)
    data2 = np.random.random(array_shape)

    slice1 = slice(None, 5)
    slice2 = slice(5, None)

    data_tup = (data1, data2)

    data_tup_slice1 = util.nested_slice(data_tup, slice1)

    assert np.all(data_tup_slice1[0] == data1[slice1])
    assert np.all(data_tup_slice1[1] == data2[slice1])

    data_tup_slice2 = util.nested_slice(data_tup, slice2)

    assert np.all(data_tup_slice2[0] == data1[slice2])
    assert np.all(data_tup_slice2[1] == data2[slice2])


def test_data_util_slicer_2():
    array_shape = (10, 20)
    data1 = np.random.random(array_shape)
    data2 = np.random.random(array_shape)

    slice1 = slice(None, 5)
    slice2 = slice(5, None)

    data_dict = {'key1': data1, 'key2': data2}

    data_dict_slice1 = util.nested_slice(data_dict, slice1)

    assert np.all(data_dict_slice1['key1'] == data1[slice1])
    assert np.all(data_dict_slice1['key2'] == data2[slice1])

    data_dict_slice2 = util.nested_slice(data_dict, slice2)

    assert np.all(data_dict_slice2['key1'] == data1[slice2])
    assert np.all(data_dict_slice2['key2'] == data2[slice2])


def test_data_util_slicer_3():
    array_shape = (10, 20)
    data1 = np.random.random(array_shape)
    data2 = np.random.random(array_shape)
    data3 = np.random.random(array_shape)

    slice1 = slice(None, 5)
    slice2 = slice(5, None)

    data = (data1, {'key1': data2, 'key2': data3})

    data_slice1 = util.nested_slice(data, slice1)

    assert np.all(data_slice1[0] == data1[slice1])
    assert np.all(data_slice1[1]['key1'] == data2[slice1])
    assert np.all(data_slice1[1]['key2'] == data3[slice1])

    data_slice2 = util.nested_slice(data, slice2)

    assert np.all(data_slice2[0] == data1[slice2])
    assert np.all(data_slice2[1]['key1'] == data2[slice2])
    assert np.all(data_slice2[1]['key2'] == data3[slice2])


def test_data_util_slicer_4():
    array_shape = (10, 20)
    data1 = np.random.random(array_shape)
    data2 = np.random.random(array_shape)
    data3 = np.random.random(array_shape)

    slice1 = slice(None, 5)
    slice2 = slice(5, None)

    data = {'key1': (data1, data2), 'key2': data3}

    data_slice1 = util.nested_slice(data, slice1)

    assert np.all(data_slice1['key1'][0] == data1[slice1])
    assert np.all(data_slice1['key1'][1] == data2[slice1])
    assert np.all(data_slice1['key2'] == data3[slice1])

    data_slice2 = util.nested_slice(data, slice2)

    assert np.all(data_slice2['key1'][0] == data1[slice2])
    assert np.all(data_slice2['key1'][1] == data2[slice2])
    assert np.all(data_slice2['key2'] == data3[slice2])


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
