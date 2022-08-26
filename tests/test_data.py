import numpy as np
from dryml.data import NumpyDataset
from dryml.data import util
import dryml
import pytest


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

    # assert len(dataset) == batch_size

    i = 0
    for el in dataset.unbatch():
        assert np.all(el == data[i])
        i += 1


def test_numpy_dataset_2():
    batch_size = 10

    data = np.random.random((batch_size, 20))

    dataset = NumpyDataset(data)

    # assert len(dataset) == batch_size

    i = 0
    for el in dataset.unbatch():
        assert np.all(el == data[i])
        i += 1


def test_numpy_dataset_3():
    batch_size = 10

    data = np.random.random((batch_size, 20))

    dataset = NumpyDataset(data, batch_size=batch_size)

    # assert len(dataset) == batch_size

    v = next(iter(dataset))

    assert np.all(v == data)


def test_numpy_dataset_4():
    batch_size = 10

    data = np.random.random((batch_size, 20))

    dataset = NumpyDataset(data)

    # assert len(dataset) == batch_size

    v = next(iter(dataset))
    assert np.all(v == data)


def test_numpy_dataset_5():
    batch_size = 32

    batch_rows = []
    for i in range(batch_size):
        batch_rows.append(np.random.random((20,)))

    dataset = NumpyDataset(batch_rows)

    # assert len(dataset) == batch_size

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

    # assert len(dataset) == batch_size

    v = next(iter(dataset.batch(batch_size=batch_size)))

    all_data = np.stack(batch_rows, axis=0)

    assert np.all(v == all_data)


def test_numpy_dataset_7():
    batch_size = 32

    batch_rows = []
    for i in range(batch_size):
        batch_rows.append(np.random.random((20,)))

    dataset = NumpyDataset(batch_rows)

    # assert len(dataset) == batch_size

    sub_batch_size = batch_size//4

    dataset = dataset.batch(batch_size=sub_batch_size)

    # assert len(dataset) == batch_size

    i = 0
    for el in dataset:
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

    # assert len(dataset) == batch_size

    dataset2 = dataset.batch(batch_size//4).unbatch()

    # assert len(dataset2) == batch_size

    for el, el2 in zip(dataset, dataset2):
        assert np.all(el == el2)


def test_numpy_dataset_9():
    batch_size = 32

    batch_rows = []
    for i in range(batch_size):
        batch_rows.append(np.random.random((20,)))

    dataset = NumpyDataset(batch_rows)

    # assert len(dataset) == batch_size

    dataset = dataset.as_indexed()

    # assert len(dataset) == batch_size

    i = 0
    for idx, el in dataset:
        assert idx == i
        assert np.all(el == batch_rows[i])
        i += 1


def test_numpy_dataset_10():
    batch_size = 32

    data = np.random.random((batch_size, 20))

    dataset = NumpyDataset(data)

    # assert len(dataset) == batch_size

    dataset = dataset.as_indexed().unbatch()

    # assert len(dataset) == batch_size

    i = 0
    for idx, el in dataset:
        assert idx == i
        assert np.all(el == data[i])
        i += 1


def test_numpy_dataset_11():
    batch_size = 32

    data = np.random.random((batch_size, 20))

    dataset = NumpyDataset(data)

    # assert len(dataset) == batch_size

    dataset = dataset.as_indexed().as_not_indexed().unbatch()

    # assert len(dataset) == batch_size

    i = 0
    for el in dataset:
        assert np.all(el == data[i])
        i += 1


def test_numpy_dataset_12():
    batch_size = 32

    data1 = np.random.random((batch_size, 20))
    data2 = np.random.random((batch_size, 20))

    dataset = NumpyDataset([(data1, data2)], batch_size=batch_size)

    # assert len(dataset) == batch_size

    dataset = dataset.as_indexed().unbatch()

    # assert len(dataset) == batch_size

    i = 0
    for idx, el in dataset:
        assert idx == i
        assert np.all(el[0] == data1[i])
        assert np.all(el[1] == data2[i])
        i += 1


def test_numpy_dataset_13():
    batch_size = 32

    data1 = np.random.random((batch_size, 20))
    data2 = np.random.random((batch_size, 20))

    dataset = NumpyDataset([(data1, data2)], batch_size=batch_size)

    # assert len(dataset) == batch_size

    dataset_2 = dataset.batch(batch_size=2)

    # assert len(dataset_2) == batch_size

    assert dataset is not dataset_2


def test_numpy_dataset_14():
    batch_size = 32

    batch_rows = []
    for i in range(batch_size):
        batch_rows.append(np.random.random((20,)))

    dataset = NumpyDataset(batch_rows).batch(batch_size=batch_size)

    # assert len(dataset) == batch_size

    assert dataset is dataset.batch(batch_size=batch_size)


def test_numpy_dataset_15():
    batch_size = 32
    data_block = np.random.random((batch_size, 20))

    dataset = NumpyDataset(data_block, batch_size=batch_size)

    # assert len(dataset) == batch_size

    take_num = 10
    dataset = dataset.unbatch().take(take_num)

    # assert len(dataset) == take_num

    count = 0
    for el in dataset:
        count += 1
        assert el.shape == (20,)

    assert count == take_num


def test_numpy_dataset_16():
    batch_size = 32
    data_block = np.random.random((batch_size, 20))

    dataset = NumpyDataset(data_block, batch_size=batch_size)

    # assert len(dataset) == batch_size

    el_a = dataset.peek()
    el_b = dataset.peek()

    assert np.all(el_a == el_b)


def test_numpy_dataset_17():
    batch_size = 32
    data_block = np.random.random((batch_size, 20))

    dataset = NumpyDataset(data_block, batch_size=batch_size)

    # assert len(dataset) == batch_size

    for el_a, el_b in zip(dataset, dataset):
        assert np.all(el_a == el_b)


def test_numpy_dataset_18():
    batch_size = 32
    data_block_x = np.random.random((batch_size, 20))
    data_block_y = np.random.random((batch_size, 20))

    dataset = NumpyDataset(
        (data_block_x, data_block_y),
        batch_size=batch_size,
        supervised=True)

    # assert len(dataset) == batch_size

    el_a = dataset.peek()
    el_b = dataset.peek()

    assert type(el_a) is tuple
    assert type(el_b) is tuple

    assert np.all(el_a[0] == el_b[0])
    assert np.all(el_a[1] == el_b[1])


def test_chain_transforms_9():
    batch_size = 32
    data_block = np.random.random((batch_size, 5))

    result = dryml.data.NumpyDataset(data_block) \
                       .batch() \
                       .apply_X(func=lambda X: X**2) \
                       .numpy() \
                       .peek()

    assert type(result) is np.ndarray
    assert np.all(data_block**2 == result)


# Define equality functions which will be needed
def np_eq(el1, el2):
    return np.all(el1 == el2)


def tf_eq(el1, el2):
    tf = pytest.importorskip('tensorflow')
    try:
        tf.debugging.assert_equal(el1, el2)
        return True
    except Exception:
        return False


def torch_eq(el1, el2):
    torch = pytest.importorskip('torch')
    try:
        torch.testing.assert_close(el1, el2, rtol=0, atol=0)
        return True
    except Exception:
        return False


datasets_to_test = []


def append_dataset_gen(f):
    datasets_to_test.append(f)
    return f


@append_dataset_gen
def np_dataset_1():
    dataset = np.random.random((50, 50))
    dataset = dryml.data.NumpyDataset(dataset)
    return dataset, np_eq


@append_dataset_gen
def np_dataset_2():
    dataset = np.random.random((50, 50))
    dataset = dryml.data.NumpyDataset(dataset).unbatch()
    return dataset, np_eq


@append_dataset_gen
def np_dataset_3():
    dataset = np.random.random((50, 50))
    dataset = dryml.data.NumpyDataset(dataset) \
                        .apply_X(lambda x: x**2)
    return dataset, np_eq


@append_dataset_gen
def tf_dataset_1():
    tf = pytest.importorskip('tensorflow')
    data = np.random.random((50, 50))
    tf_data = tf.data.Dataset.from_tensor_slices(data)
    from dryml.data.tf import TFDataset
    dataset = TFDataset(tf_data)
    return dataset, tf_eq


@append_dataset_gen
def tf_dataset_2():
    tf = pytest.importorskip('tensorflow')
    data = np.random.random((50, 50))
    tf_data = tf.data.Dataset.from_tensor_slices(data)
    from dryml.data.tf import TFDataset
    dataset = TFDataset(tf_data).unbatch()
    return dataset, tf_eq


@append_dataset_gen
def tf_dataset_3():
    tf = pytest.importorskip('tensorflow')
    data = np.random.random((50, 50))
    tf_data = tf.data.Dataset.from_tensor_slices(data)
    from dryml.data.tf import TFDataset
    dataset = TFDataset(tf_data).apply_X(lambda x: tf.multiply(x, x))
    return dataset, tf_eq


@append_dataset_gen
def np_tf_dataset_1():
    pytest.importorskip('tensorflow')
    data = np.random.random((50, 50))
    ds = dryml.data.NumpyDataset(data)
    ds = ds.tf()
    return ds, tf_eq


@append_dataset_gen
def np_tf_dataset_2():
    tf = pytest.importorskip('tensorflow')
    data = np.random.random((50, 50))
    ds = dryml.data.NumpyDataset(data)
    ds = ds.tf().apply_X(
        lambda x: tf.multiply(x, x))
    return ds, tf_eq


@append_dataset_gen
def np_tf_dataset_3():
    tf = pytest.importorskip('tensorflow')
    data = np.random.random((50, 50))
    ds = dryml.data.NumpyDataset(data)

    def mult(x):
        return tf.multiply(x, x)

    ds = ds.tf().apply_X(mult)

    return ds, tf_eq


@pytest.mark.parametrize(
    'dataset_gen', datasets_to_test)
def test_double_collect_1(dataset_gen):
    dataset, eq_func = dataset_gen()
    collect_elements = dataset.collect()
    collect_elements_2 = dataset.collect()

    assert len(collect_elements) > 0
    assert len(collect_elements) == len(collect_elements_2)
    for el1, el2 in zip(collect_elements, collect_elements_2):
        assert eq_func(el1, el2)


@pytest.mark.parametrize(
    'dataset_gen', datasets_to_test)
def test_double_collect_2(dataset_gen):
    dataset, eq_func = dataset_gen()
    collect_elements = dataset.take(1).collect()
    collect_elements_2 = dataset.take(1).collect()

    assert len(collect_elements) > 0
    assert len(collect_elements) == len(collect_elements_2)
    for el1, el2 in zip(collect_elements, collect_elements_2):
        assert eq_func(el1, el2)
