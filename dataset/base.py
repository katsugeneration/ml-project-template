# Copyright 2019 Katsuya Shimabukuro. All rights reserved.
# Licensed under the MIT License.
from typing import Tuple, Any, Generator, Union
import pathlib
import pandas as pd
import numpy as np
from PIL import Image
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


class DirectoryImageSegmentationDataset(ImageSegmentationDatasetBase):
    """Directory loaded classifier dataset.

    Args:
        directory (str): path to image directory.
                         this dirctory contains `train`, `train_labels`, `val` and `val_labels` directories,
                         which have correspondence input images or label images.
        class_csv (str): path to class color definition csv. each rows has `name`, `r`, `g` and `b` columns.

    """

    def __init__(
            self,
            directory: str,
            class_csv: str,
            **kwargs: Any) -> None:
        """Initilize params."""
        super(DirectoryImageSegmentationDataset, self).__init__(**kwargs)
        self.directory = pathlib.Path(directory)
        self.class_dict: pd.DataFrame = pd.read_csv(class_csv)

    def _label_image_to_category(
            self,
            image: np.array) -> np.array:
        """Convert label image to category number list.

        Args:
            image (np.array): image array. shape is H x W x 3

        Return:
            label (np.array): label array. shape is H x W x category_nums

        """
        label = np.ones(image.shape[:2]) * len(self.class_dict)
        for i, r in self.class_dict.iterrows():
            equality = np.equal(image, [r['r'], r['g'], r['b']]).all(axis=-1)
            label[equality] = i
        return label

    def training_data(self) -> Tuple[np.array, np.array]:
        """Return training dataset.

        Return:
            dataset (Tuple[np.array, np.array]): training dataset pair

        """
        train_images = []
        for path in sorted(self.directory.joinpath('train').glob('*')):
            try:
                train_images.append(np.array(Image.open(path)))
            except OSError:
                pass

        train_labels = []
        for path in sorted(self.directory.joinpath('train_labels').glob('*')):
            try:
                train_labels.append(self._label_image_to_category(np.array(Image.open(path))))
            except OSError:
                pass
        train_labels = tf.keras.utils.to_categorical(train_labels)

        return (train_images, train_labels)
