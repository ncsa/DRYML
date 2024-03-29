import numpy as np


def gen_dataset_1(
        num_examples=1000, x_l=-1., x_h=1.,
        a=-0.5, b=0., c=10., eps=0.5):
    x_data = np.random.random((num_examples, 1))
    x_data = x_data*(x_h-x_l)
    x_data = x_data+x_l

    eps = np.random.normal(loc=0., scale=1., size=(num_examples, 1))

    y_data = a*np.multiply(x_data, x_data)+b*x_data+c+eps

    return (x_data, y_data)


def gen_dataset_2(
        num_examples=5000,
        centers: np.ndarray = np.array([[-0.5, 5.], [2., 2.], [-3., -1.]]),
        widths: np.ndarray = np.array([[1, 1], [1, 1], [1, 1]]),
        supervised=True):

    # Check that centers and widths shapes are equal
    if centers.shape != widths.shape:
        raise ValueError(
            "Need to pass equal numbers of centers and widths, "
            "of same dimension")

    # Detect number of classes
    num_classes = centers.shape[0]
    dim = centers.shape[1]

    # Generate classes
    classes = np.random.choice(num_classes, size=num_examples)

    points = []

    for i in range(dim):
        # For each dimension, generate data
        cs = centers[classes, i]
        ws = widths[classes, i]
        points.append(np.random.normal(loc=cs, scale=ws))

    points = np.stack(points, axis=1)

    if supervised:
        return (points, classes)
    else:
        return points
