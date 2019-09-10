# Copyright 2019 Katsuya Shimabukuro. All rights reserved.
# Licensed under the MIT License.

from typing import Any, Union, Generator, Tuple
import tensorflow as tf
import numpy as np
from dataset.base import BinaryImageClassifierDataset, ImageSegmentationDatasetBase


class DummyDataset(BinaryImageClassifierDataset):
    """Dummy dataset class for test."""

    input_shape = (1, 1, 1)
    category_nums = 2

    def __init__(
            self,
            **kwargs: Any) -> None:
        """Load data and setup preprocessing."""
        super(DummyDataset, self).__init__(**kwargs)

        self.x_train = np.zeros((self.batch_size, ) + self.input_shape)
        self.y_train = np.array([0, 1] * (self.batch_size // 2))
        self.x_test = np.zeros((self.batch_size, ) + self.input_shape)
        self.y_test = np.array([0, 1] * (self.batch_size // 2))


class SegmentationDummyDataset(ImageSegmentationDatasetBase):
    """Dummy dataset class for test."""

    input_shape = (256, 256, 3)
    category_nums = 2

    def __init__(
            self,
            **kwargs: Any) -> None:
        """Load data and setup preprocessing."""
        super(SegmentationDummyDataset, self).__init__(**kwargs)

        self.x_train = np.zeros((self.batch_size, ) + self.input_shape)
        self.y_train = np.zeros((self.batch_size, ) + self.input_shape[:2])
        self.x_test = np.ones((self.batch_size, ) + self.input_shape)
        self.y_test = np.ones((self.batch_size, ) + self.input_shape[:2])

    def training_data_generator(self) -> Union[tf.keras.utils.Sequence, Generator]:
        """Return training dataset.

        Return:
            dataset (Union[tf.keras.utils.Sequence, Generator]): dataset generator

        """
        y_train = tf.keras.utils.to_categorical(self.y_train, num_classes=self.category_nums)
        self.steps_per_epoch = len(self.x_train) // self.batch_size
        return self.train_data_gen.flow(self.x_train, y=y_train, batch_size=self.batch_size)

    def eval_data_generator(self) -> Union[tf.keras.utils.Sequence, Generator]:
        """Return evaluation dataset.

        Return:
            dataset (Union[tf.keras.utils.Sequence, Generator]): dataset generator

        """
        y_test = tf.keras.utils.to_categorical(self.y_test, num_classes=self.category_nums)
        self.eval_steps_per_epoch = len(self.x_test) // self.batch_size
        return self.eval_data_gen.flow(self.x_test, y=y_test, batch_size=self.batch_size)

    def eval_data(self) -> Tuple[np.array, np.array]:
        """Return evaluation dataset.

        Return:
            dataset (Union[tf.keras.utils.Sequence, Generator]): dataset generator

        """
        y_test = tf.keras.utils.to_categorical(self.y_test, num_classes=self.category_nums)
        return self.x_test, y_test
