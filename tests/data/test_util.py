import numpy as np

from dryml.data.util import nested_slice


def test_nested_slice_slices_tuple_and_dict_leaves():
    data = (
        np.array([0, 1, 2, 3], dtype=np.int32),
        {
            "x": np.array([[0, 1], [2, 3], [4, 5], [6, 7]], dtype=np.int32),
            "y": (np.array([10, 11, 12, 13], dtype=np.int32),),
        },
    )

    out = nested_slice(data, slice(1, 3))

    assert out[0].tolist() == [1, 2]
    assert out[1]["x"].tolist() == [[2, 3], [4, 5]]
    assert out[1]["y"][0].tolist() == [11, 12]


def test_nested_slice_slices_single_index_across_leaves():
    data = {
        "left": np.array([0, 1, 2], dtype=np.int32),
        "right": (np.array([[3], [4], [5]], dtype=np.int32),),
    }

    out = nested_slice(data, 2)

    assert int(out["left"]) == 2
    assert out["right"][0].tolist() == [5]
