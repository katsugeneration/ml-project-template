from typing import Tuple, Any, Iterator
import numpy as np
import tensorflow as tf


class DatasetBase(object):
    """Dataset loader base class."""

    def __init__(self) -> None:
        """Load data and setup preprocessing."""
        pass

    def training_data(self) -> Tuple[np.array, np.array]:
        """Return training dataset.

        Return:
            dataset (Tuple[np.array, np.array]): training dataset pair

        """
        pass

    def training_data_generator(self) -> Iterator:
        """Return training dataset iterator.

        Return:
            dataset (Iterator): dataset iterator

        """
        pass

    def eval_data(self) -> Tuple[np.array, np.array]:
        """Return evaluation dataset.

        Return:
            dataset (Tuple[np.array, np.array]): evaluation dataset pair

        """
        pass

    def eval_data_generator(self) -> Iterator:
        """Return evaluation dataset.

        Return:
            dataset (Iterator): dataset generator

        """
        pass


class ImageClassifierDatasetBase(DatasetBase):
    """Image classification dataset loader base class.

    Args:
        batch_size (int): training batch size.

    """

    def __init__(
            self,
            batch_size: int = 32,
            **kwargs: Any
            ) -> None:
        """Initialize data generator."""
        self.batch_size = batch_size
        self.train_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(**kwargs)
        self.eval_data_gen = tf.keras.preprocessing.image.ImageDataGenerator()
        self.input_shape: Tuple[int, int, int]
        self.category_nums: int
        self.steps_per_epoch: int
        self.eval_steps_per_epoch: int


class BinaryImageClassifierDataset(ImageClassifierDatasetBase):
    """Memory loaded classifier dataset."""

    def __init__(self, **kwargs: Any) -> None:
        super(BinaryImageClassifierDataset, self).__init__(**kwargs)
        self.x_train: np.array
        self.x_test: np.array
        self.y_train: np.array
        self.y_test: np.array

    def training_data(self) -> Tuple[np.array, np.array]:
        """Return training dataset.

        Return:
            dataset (Tuple[np.array, np.array]): training dataset pair

        """
        return (self.train_data_gen.random_transform(self.x_train), self.y_train)

    def training_data_generator(self) -> Iterator:
        """Return training dataset.

        Return:
            dataset (Iterator): dataset generator

        """
        self.steps_per_epoch = len(self.x_train) // self.batch_size
        return self.train_data_gen.flow(self.x_train, y=self.y_train, batch_size=self.batch_size)

    def eval_data(self) -> Tuple[np.array, np.array]:
        """Return evaluation dataset.

        Return:
            dataset (Tuple[np.array, np.array]): evaluation dataset pair

        """
        return (self.x_test, self.y_test)

    def eval_data_generator(self) -> Iterator:
        """Return evaluation dataset.

        Return:
            dataset (Iterator): dataset generator

        """
        self.eval_steps_per_epoch = len(self.x_test) // self.batch_size
        return self.eval_data_gen.flow(self.x_test, y=self.y_test, batch_size=self.batch_size)
