from dryml.data.transforms import StaticTransform
from dryml.data.util import nestize


class ImageNormalization(StaticTransform):
    def numpy_eval(self, data, *args, **kwargs):
        import numpy as np
        return data.apply_X(
            nestize(lambda image: np.cast(image, np.float32) / 255.))

    def tf_eval(self, data, *args, **kwargs):
        import tensorflow as tf
        return data.apply_X(
            nestize(lambda image: tf.cast(image, tf.float32) / 255.))

    def torch_eval(self, data, *args, **kwargs):
        import torch
        return data.apply_X(
            nestize(lambda image: image.to(torch.float32) / 255.))
