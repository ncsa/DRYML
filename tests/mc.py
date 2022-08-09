import numpy as np


def gen_dataset(
        num_examples=1000, x_l=-1., x_h=1.,
        a=-0.5, b=0., c=10., eps=0.5):
    x_data = np.random.random((num_examples, 1))
    x_data = x_data*(x_h-x_l)
    x_data = x_data+x_l

    eps = np.random.normal(loc=0., scale=1., size=(num_examples, 1))

    y_data = a*np.multiply(x_data, x_data)+b*x_data+c+eps

    return (x_data, y_data)
