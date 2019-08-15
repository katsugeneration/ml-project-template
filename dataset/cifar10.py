from typing import Any
import tensorflow as tf
from dataset.base import BinaryImageClassifierDataset


class Cifar10Dataset(BinaryImageClassifierDataset):
    """Cifar10 dataset loader."""

    input_shape = (32, 32, 3)
    category_nums = 10

    def __init__(self, **kwargs: Any) -> None:
        """Load data and setup preprocessing."""
        super(Cifar10Dataset, self).__init__(**kwargs)
        cifar10 = tf.keras.datasets.cifar10
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
