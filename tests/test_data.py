import numpy as np
from dryml.data import NumpyDataset
from dryml.data import util
from dryml.data.tf import TFDataset
import dryml
import tensorflow as tf


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


def test_numpy_dataset_9():
    batch_size = 32

    batch_rows = []
    for i in range(batch_size):
        batch_rows.append(np.random.random((20,)))

    dataset = NumpyDataset(batch_rows)

    i = 0
    for idx, el in dataset.as_indexed():
        assert idx == i
        assert np.all(el == batch_rows[i])
        i += 1


def test_numpy_dataset_10():
    batch_size = 32

    data = np.random.random((batch_size, 20))

    dataset = NumpyDataset(data)

    i = 0
    for idx, el in dataset.as_indexed().unbatch():
        assert idx == i
        assert np.all(el == data[i])
        i += 1


def test_numpy_dataset_11():
    batch_size = 32

    data = np.random.random((batch_size, 20))

    dataset = NumpyDataset(data)

    i = 0
    for el in dataset.as_indexed().as_not_indexed().unbatch():
        assert np.all(el == data[i])
        i += 1


def test_numpy_dataset_12():
    batch_size = 32

    data1 = np.random.random((batch_size, 20))
    data2 = np.random.random((batch_size, 20))

    dataset = NumpyDataset([(data1, data2)], batch_size=batch_size)

    i = 0
    for idx, el in dataset.as_indexed().unbatch():
        assert idx == i
        assert np.all(el[0] == data1[i])
        assert np.all(el[1] == data2[i])
        i += 1


def test_numpy_dataset_13():
    batch_size = 32

    data1 = np.random.random((batch_size, 20))
    data2 = np.random.random((batch_size, 20))

    dataset = NumpyDataset([(data1, data2)], batch_size=batch_size)

    assert dataset is not dataset.batch(batch_size=2)


def test_numpy_dataset_14():
    batch_size = 32

    batch_rows = []
    for i in range(batch_size):
        batch_rows.append(np.random.random((20,)))

    dataset = NumpyDataset(batch_rows).batch(batch_size=batch_size)

    assert dataset is dataset.batch(batch_size=batch_size)


def test_numpy_dataset_15():
    batch_size = 32
    data_block = np.random.random((batch_size, 20))

    dataset = NumpyDataset(data_block, batch_size=batch_size)

    take_num = 10
    dataset = dataset.unbatch().take(take_num)

    count = 0
    for el in dataset:
        count += 1
        assert el.shape == (20,)

    assert count == take_num


def test_numpy_dataset_16():
    batch_size = 32
    data_block = np.random.random((batch_size, 20))

    dataset = NumpyDataset(data_block, batch_size=batch_size)

    el_a = dataset.peek()
    el_b = dataset.peek()

    assert np.all(el_a == el_b)


def test_numpy_dataset_17():
    batch_size = 32
    data_block = np.random.random((batch_size, 20))

    dataset = NumpyDataset(data_block, batch_size=batch_size)

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

    el_a = dataset.peek()
    el_b = dataset.peek()

    assert type(el_a) is tuple
    assert type(el_b) is tuple

    assert np.all(el_a[0] == el_b[0])
    assert np.all(el_a[1] == el_b[1])


def test_tf_dataset_1():
    with dryml.context.ContextManager({'tf': {}}):
        batch_size = 10

        data = np.random.random((batch_size, 20))

        tf_ds = tf.data.Dataset.from_tensor_slices(data)

        dataset = TFDataset(tf_ds)

        i = 0
        for el in dataset.unbatch():
            assert np.all(el.numpy() == data[i])
            i += 1


def test_tf_dataset_2():
    with dryml.context.ContextManager({'tf': {}}):
        batch_size = 10

        data = np.random.random((batch_size, 20))

        tf_data = tf.data.Dataset.from_tensor_slices(data)

        dataset = TFDataset(tf_data)

        dataset = dataset.batch(batch_size=batch_size)

        v = next(iter(dataset))

        assert np.all(v.numpy() == data)


def test_tf_dataset_3():
    with dryml.context.ContextManager({'tf': {}}):
        batch_size = 32

        batch_rows = []
        for i in range(batch_size):
            batch_rows.append(np.random.random((20,)))

        tf_data = tf.data.Dataset.from_tensors(batch_rows)

        dataset = TFDataset(tf_data, batch_size=len(batch_rows))

        i = 0
        for el in dataset.unbatch():
            assert np.all(el.numpy() == batch_rows[i])
            i += 1


def test_tf_dataset_4():
    with dryml.context.ContextManager({'tf': {}}):
        batch_size = 32

        batch_rows = []
        for i in range(batch_size):
            batch_rows.append(np.random.random((20,)))

        tf_data = tf.data.Dataset.from_tensors(batch_rows)

        dataset = TFDataset(tf_data, batch_size=len(batch_rows))

        v = next(iter(dataset))

        all_data = np.stack(batch_rows, axis=0)

        assert np.all(v.numpy() == all_data)


def test_tf_dataset_5():
    with dryml.context.ContextManager({'tf': {}}):
        batch_size = 32

        batch_rows = []
        for i in range(batch_size):
            batch_rows.append(np.random.random((20,)))

        tf_data = tf.data.Dataset.from_tensors(batch_rows)

        dataset = TFDataset(tf_data, batch_size=len(batch_rows))

        sub_batch_size = batch_size//4

        i = 0
        for el in dataset.batch(batch_size=sub_batch_size):
            sub_batch = np.stack(
                batch_rows[i*sub_batch_size:(i+1)*sub_batch_size], axis=0)
            assert np.all(el.numpy() == sub_batch)
            i += 1


def test_tf_dataset_6():
    with dryml.context.ContextManager({'tf': {}}):
        batch_size = 32

        batch_rows = []
        for i in range(batch_size):
            batch_rows.append(np.random.random((20,)))

        tf_data = tf.data.Dataset.from_tensors(batch_rows)

        dataset = TFDataset(tf_data, batch_size=len(batch_rows)).unbatch()

        dataset2 = dataset.batch(batch_size//4).unbatch()

        for el, el2 in zip(dataset, dataset2):
            assert np.all(el.numpy() == el2.numpy())


def test_tf_dataset_7():
    with dryml.context.ContextManager({'tf': {}}):
        batch_size = 32

        batch_rows = []
        for i in range(batch_size):
            batch_rows.append(np.random.random((20,)))

        tf_data = tf.data.Dataset.from_tensors(batch_rows)

        dataset = TFDataset(tf_data, batch_size=len(batch_rows)).unbatch()

        i = 0
        for idx, el in dataset.as_indexed():
            assert idx.numpy() == i
            assert np.all(el.numpy() == batch_rows[i])
            i += 1


def test_tf_dataset_8():
    with dryml.context.ContextManager({'tf': {}}):
        batch_size = 32

        data = np.random.random((batch_size, 20))

        tf_data = tf.data.Dataset.from_tensor_slices(data)

        dataset = TFDataset(tf_data)

        i = 0
        for idx, el in dataset.as_indexed():
            assert idx.numpy() == i
            assert np.all(el.numpy() == data[i])
            i += 1


def test_tf_dataset_9():
    with dryml.context.ContextManager({'tf': {}}):
        batch_size = 32

        data = np.random.random((batch_size, 20))

        tf_data = tf.data.Dataset.from_tensor_slices(data)

        dataset = TFDataset(tf_data)

        i = 0
        for el in dataset.as_indexed().as_not_indexed():
            assert np.all(el.numpy() == data[i])
            i += 1


def test_tf_dataset_10():
    with dryml.context.ContextManager({'tf': {}}):
        batch_size = 32

        data1 = np.random.random((batch_size, 20))
        data2 = np.random.random((batch_size, 20))

        tf_data = tf.data.Dataset.from_tensor_slices((data1, data2))

        dataset = TFDataset(tf_data)

        i = 0
        for idx, el in dataset.as_indexed():
            assert idx.numpy() == i
            assert np.all(el[0].numpy() == data1[i])
            assert np.all(el[1].numpy() == data2[i])
            i += 1
        assert i == batch_size


def test_tf_dataset_11():
    with dryml.context.ContextManager({'tf': {}}):
        batch_size = 32

        data1 = np.random.random((batch_size, 20))
        data2 = np.random.random((batch_size, 20))

        tf_data = tf.data.Dataset.from_tensor_slices((data1, data2))

        dataset = TFDataset(tf_data).batch(batch_size=32)

        assert dataset is not dataset.batch(batch_size=2)


def test_tf_dataset_12():
    with dryml.context.ContextManager({'tf': {}}):
        batch_size = 32

        batch_rows = []
        for i in range(batch_size):
            batch_rows.append(np.random.random((20,)))

        tf_data = tf.data.Dataset.from_tensors(batch_rows)

        dataset = TFDataset(tf_data).batch(batch_size=batch_size)

        assert dataset is dataset.batch(batch_size=batch_size)


def test_tf_dataset_13():
    with dryml.context.ContextManager({'tf': {}}):
        batch_size = 32
        data_block = np.random.random((batch_size, 20))

        tf_data = tf.data.Dataset.from_tensor_slices(data_block)

        dataset = TFDataset(tf_data).batch(batch_size=batch_size)

        take_num = 10
        dataset = dataset.unbatch().take(take_num)

        count = 0
        for el in dataset:
            count += 1
            assert el.shape == (20,)

        assert count == take_num


def test_tf_dataset_14():
    with dryml.context.ContextManager({'tf': {}}):
        batch_size = 32
        data_block = np.random.random((batch_size, 20))

        tf_data = tf.data.Dataset.from_tensor_slices(data_block)

        dataset = TFDataset(tf_data)

        el1 = dataset.peek()
        el2 = dataset.peek()

        assert np.all(el1.numpy() == el2.numpy())


def test_tf_dataset_15():
    with dryml.context.ContextManager({'tf': {}}):
        batch_size = 32
        data_block = np.random.random((batch_size, 20))

        tf_data = tf.data.Dataset.from_tensor_slices(data_block)

        dataset = TFDataset(tf_data)

        for el_a, el_b in zip(dataset, dataset):
            assert np.all(el_a.numpy() == el_b.numpy())


def test_tf_to_numpy_dataset_1():
    with dryml.context.ContextManager({'tf': {}}):
        batch_size = 32
        data_block = np.random.random((batch_size, 20))

        tf_data = tf.data.Dataset.from_tensor_slices(data_block)

        dataset = TFDataset(tf_data)

        numpy_dataset = dataset.numpy()
        i = 0
        for tf_el, numpy_el in zip(dataset, numpy_dataset):
            assert np.all(tf_el.numpy() == numpy_el)
            i += 1
        assert i == batch_size


def test_numpy_to_tf_dataset_2():
    with dryml.context.ContextManager({'tf': {}}):
        batch_size = 32
        data_block = np.random.random((batch_size, 20))

        dataset = NumpyDataset(data_block, batch_size=batch_size)

        tf_dataset = dataset.tf()

        numpy_data = dataset.peek()
        tf_data = tf_dataset.peek()

        assert np.all(numpy_data == tf_data.numpy())


def test_chain_transforms_1():
    with dryml.context.ContextManager({'tf': {}}):
        batch_size = 32
        data_block = np.random.random((batch_size, 5))

        dataset = NumpyDataset(data_block, batch_size=batch_size)

        dataset = dataset.unbatch().tf()

        i = 0
        for el_a, el_b in zip(dataset, dataset):
            assert np.all(el_a.numpy() == data_block[i])
            assert np.all(el_b.numpy() == data_block[i])
            assert np.all(el_a.numpy() == el_b.numpy())
            i += 1


def test_chain_transforms_1a():
    with dryml.context.ContextManager({'tf': {}}):
        batch_size = 32
        data_block = np.random.random((batch_size, 5))

        dataset = NumpyDataset(data_block, batch_size=batch_size)

        dataset = dataset.unbatch().tf()

        it1 = iter(dataset)
        it2 = iter(dataset)

        el_a = next(it1)
        el_b = next(it2)

        assert np.all(el_a.numpy() == data_block[0])
        assert np.all(el_b.numpy() == data_block[0])
        assert np.all(el_a.numpy() == el_b.numpy())


def test_chain_transforms_2():
    with dryml.context.ContextManager({'tf': {}}):
        batch_size = 32
        data_block = np.random.random((batch_size, 5))

        tf_data = tf.data.Dataset.from_tensor_slices(data_block)

        dataset = TFDataset(tf_data)

        dataset = dataset.numpy()

        i = 0
        for el_a, el_b in zip(dataset, dataset):
            assert np.all(el_a == data_block[i])
            assert np.all(el_b == data_block[i])
            assert np.all(el_a == el_b)
            i += 1


def test_chain_transforms_3():
    with dryml.context.ContextManager({'tf': {}}):
        batch_size = 32
        data_block = np.random.random((batch_size, 5))

        tf_data = tf.data.Dataset.from_tensor_slices(data_block)

        dataset_orig = TFDataset(tf_data)

        dataset = dataset_orig.numpy().tf()

        i = 0
        for el1, el2 in zip(dataset_orig, dataset):
            assert np.all(el1.numpy() == el2.numpy())
            i += 1
        assert i == batch_size


def test_chain_transforms_4():
    with dryml.context.ContextManager({'tf': {}}):
        batch_size = 32
        data_block = np.random.random((batch_size, 5))

        tf_data = tf.data.Dataset.from_tensor_slices(data_block)

        dataset_orig = TFDataset(tf_data)

        dataset = dataset_orig.numpy().apply_X(lambda X: X**2).tf()

        i = 0
        for el1, el2 in zip(dataset_orig, dataset):
            assert np.all(el1.numpy()**2 == el2.numpy())
            i += 1
        assert i == batch_size


def test_chain_transforms_5():
    with dryml.context.ContextManager({'tf': {}}):
        batch_size = 32
        data_block_x = np.random.random((batch_size, 5))
        data_block_y = np.random.random((batch_size, 5))

        tf_data = tf.data.Dataset.from_tensor_slices(
            (data_block_x, data_block_y))

        dataset_orig = TFDataset(tf_data, supervised=True)

        dataset = dataset_orig.numpy().apply_X(lambda X: X**2).tf()

        i = 0
        for el1, el2 in zip(dataset_orig, dataset):
            assert np.all(el1[0].numpy()**2 == el2[0].numpy())
            assert np.all(el1[1].numpy() == el2[1].numpy())
            i += 1
        i == batch_size
