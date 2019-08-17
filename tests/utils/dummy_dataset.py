# Copyright 2019 Katsuya Shimabukuro. All rights reserved.
# Licensed under the MIT License.

from typing import Any
import tensorflow as tf
import numpy as np
from dataset.base import BinaryImageClassifierDataset


class DummyDataset(BinaryImageClassifierDataset):
    """Dummy dataset class for test."""

    input_shape = (1, 1, 1)
    category_nums = 1

    def __init__(
            self,
            **kwargs: Any) -> None:
        """Load data and setup preprocessing."""
        super(DummyDataset, self).__init__(**kwargs)

        self.x_train = np.zeros((self.batch_size, ) + self.input_shape)
        self.y_train = [0] * self.batch_size
        self.x_test = np.zeros((self.batch_size, ) + self.input_shape)
        self.y_test = [0] * self.batch_size
