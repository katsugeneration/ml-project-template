from typing import Any
import tensorflow as tf
from dataset.base import BinaryImageClassifierDataset


class Cifar100Dataset(BinaryImageClassifierDataset):
    """Cifar10 dataset loader.

    Args:
        data_normalize_style (str): data noramlization style. Allowed valus are 0to1 ot -1to1

    """

    input_shape = (32, 32, 3)
    category_nums = 100

    def __init__(
            self,
            data_normalize_style: str = '0to1',
            **kwargs: Any) -> None:
        """Load data and setup preprocessing."""
        super(Cifar100Dataset, self).__init__(**kwargs)
        cifar100 = tf.keras.datasets.cifar100
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()

        if data_normalize_style == '0to1':
            x_train, x_test = x_train / 255.0, x_test / 255.0
        elif data_normalize_style == '-1to1':
            x_train, x_test = (x_train - 127.5) / 127.5, (x_test - 127.5) / 127.5
        else:
            raise ValueError('Data normaliztion style: {} is not supported.'.format(data_normalize_style))

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
