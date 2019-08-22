# Copyright 2019 Katsuya Shimabukuro. All rights reserved.
# Licensed under the MIT License.
from typing import Tuple, Any, Generator, Union
import numpy as np
import tensorflow as tf
from dataset.mixup import MixupGenerator


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

    def training_data_generator(self) -> Union[tf.keras.utils.Sequence, Generator]:
        """Return training dataset iterator.

        Return:
            dataset (Union[tf.keras.utils.Sequence, Generator]): dataset iterator

        """
        pass

    def eval_data(self) -> Tuple[np.array, np.array]:
        """Return evaluation dataset.

        Return:
            dataset (Tuple[np.array, np.array]): evaluation dataset pair

        """
        pass

    def eval_data_generator(self) -> Union[tf.keras.utils.Sequence, Generator]:
        """Return evaluation dataset.

        Return:
            dataset (Union[tf.keras.utils.Sequence, Generator]): dataset generator

        """
        pass


class ImageClassifierDatasetBase(DatasetBase):
    """Image classification dataset loader base class.

    Args:
        batch_size (int): training batch size.
        use_mixup (bool): whether to use mixup augmentation.

    """

    def __init__(
            self,
            batch_size: int = 32,
            use_mixup: bool = False,
            **kwargs: Any
            ) -> None:
        """Initialize data generator."""
        self.batch_size = batch_size
        self.use_mixup = use_mixup
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
        y_train = tf.keras.utils.to_categorical(self.y_train)
        return (self.train_data_gen.random_transform(self.x_train), y_train)

    def training_data_generator(self) -> Union[tf.keras.utils.Sequence, Generator]:
        """Return training dataset.

        Return:
            dataset (Union[tf.keras.utils.Sequence, Generator]): dataset generator

        """
        y_train = tf.keras.utils.to_categorical(self.y_train)
        self.steps_per_epoch = len(self.x_train) // self.batch_size
        if self.use_mixup:
            return MixupGenerator(datagen=self.train_data_gen).flow(self.x_train, y_train, batch_size=self.batch_size)
        else:
            return self.train_data_gen.flow(self.x_train, y=y_train, batch_size=self.batch_size)

    def eval_data(self) -> Tuple[np.array, np.array]:
        """Return evaluation dataset.

        Return:
            dataset (Tuple[np.array, np.array]): evaluation dataset pair

        """
        y_test = tf.keras.utils.to_categorical(self.y_test)
        return (self.x_test, y_test)

    def eval_data_generator(self) -> Union[tf.keras.utils.Sequence, Generator]:
        """Return evaluation dataset.

        Return:
            dataset (Union[tf.keras.utils.Sequence, Generator]): dataset generator

        """
        y_test = tf.keras.utils.to_categorical(self.y_test)
        self.eval_steps_per_epoch = len(self.x_test) // self.batch_size
        return self.eval_data_gen.flow(self.x_test, y=y_test, batch_size=self.batch_size)


class ImageSegmentationDatasetBase(ImageClassifierDatasetBase):
    """Image segmentation dataset loader base class."""

    pass
