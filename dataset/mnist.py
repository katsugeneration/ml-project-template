from typing import Any
import tensorflow as tf
import numpy as np
from dataset.base import BinaryImageClassifierDataset


class MnistDataset(BinaryImageClassifierDataset):
    """Mnist dataset loader."""

    input_shape = (28, 28, 1)
    category_nums = 10

    def __init__(self, **kwargs: Any) -> None:
        """Load data and setup preprocessing."""
        super(MnistDataset, self).__init__(**kwargs)
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        x_train = np.expand_dims(x_train, axis=3)
        x_test = np.expand_dims(x_test, axis=3)

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
