# Copyright 2019 Katsuya Shimabukuro. All rights reserved.
# Licensed under the MIT License.
from typing import Tuple, Any, Generator, Union, List
import pathlib
import multiprocessing
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
        train_image_directory (str): path to training image directory.
        train_label_directory (str): path to training label directory.
        test_image_directory (str): path to test image directory.
        test_image_directory (str): path to test label directory.
        class_csv (str): path to class color definition csv. each rows has `name`, `r`, `g` and `b` columns.
        crop_width (int): crop width. if value is 0, we do not crop.
        crop_height (int): crop width. if value is 0, we do not crop.
        window_size (int): pre images and pot images scope size.

    """

    def __init__(
            self,
            train_image_directory: str,
            train_label_directory: str,
            test_image_directory: str,
            test_label_directory: str,
            class_csv: str,
            crop_width: int = 0,
            crop_height: int = 0,
            window_size: int = 0,
            **kwargs: Any) -> None:
        """Initilize params."""
        super(DirectoryImageSegmentationDataset, self).__init__(**kwargs)
        self.train_image_directory = pathlib.Path(train_image_directory)
        self.train_label_directory = pathlib.Path(train_label_directory)
        self.test_image_directory = pathlib.Path(test_image_directory)
        self.test_label_directory = pathlib.Path(test_label_directory)
        self.class_dict: pd.DataFrame = pd.read_csv(class_csv)
        self.category_nums = len(self.class_dict)

        for path in self.train_image_directory.glob('*'):
            try:
                image = np.array(Image.open(path))
            except OSError:
                continue

            self.input_shape = image.shape
            break

        self.input_shape = (self.input_shape[0], self.input_shape[1], self.input_shape[2] * (2 * window_size + 1))
        if crop_height != 0 and crop_width != 0:
            self.input_shape = (crop_width, crop_height, self.input_shape[2])
        self.crop_width = crop_width
        self.crop_height = crop_height
        self.window_size = window_size

    def _random_crop(
            self,
            image: np.array,
            label: np.array) -> Tuple[np.array, np.array]:
        """Crop image and label to correspondence position.

        Args:
            image (np.array): target image.
            label (np.array): target label.

        Return:
            cropped_image (np.array): cropped image
            cropped_label (np.array): cropped label

        """
        if (image.shape[0] != label.shape[0]) or (image.shape[1] != label.shape[1]):
            raise ValueError('Image and label must have the same dimensions!')

        if self.crop_width == 0 or self.crop_height == 0:
            return image, label

        if (self.crop_width <= image.shape[0]) and (self.crop_height <= image.shape[1]):
            x = np.random.randint(0, image.shape[0]-self.crop_width)
            y = np.random.randint(0, image.shape[1]-self.crop_height)

            return image[x:x+self.crop_width, y:y+self.crop_height], label[x:x+self.crop_width, y:y+self.crop_height]
        else:
            raise Exception(
                'Crop shape (%d, %d) exceeds image dimensions (%d, %d)!'.format(
                    (self.crop_width, self.crop_height, image.shape[0], image.shape[1])))

    def _label_image_to_category(
            self,
            image: np.array) -> np.array:
        """Convert label image to category number list.

        Args:
            image (np.array): image array. shape is H x W x 3

        Return:
            label (np.array): label array. shape is H x W x category_nums

        """
        label = np.ones(image.shape[:2]) * self.category_nums
        for i, r in self.class_dict.iterrows():
            equality = np.equal(image, [r['r'], r['g'], r['b']]).all(axis=-1)
            label[equality] = i
        return tf.keras.utils.to_categorical(label, num_classes=self.category_nums)

    def _load_data(
            self,
            image_path: str,
            label_path: str) -> Tuple[np.array, np.array]:
        """Load one data.

        Args:
            image_path (str): path to image.
            label_path (str): path to correspondence label.

        Return:
            image (np.array): image object.
            label (mp.array): correspondence label object.
        """
        try:
            image = np.array(Image.open(image_path))
            label = self._label_image_to_category(np.array(Image.open(label_path).convert('RGB')))
            image, label = self._random_crop(image, label)
            train_data = self.train_data_gen.random_transform(np.concatenate([image, label], axis=-1))
            return train_data[:, :, :3], train_data[:, :, 3:]
        except OSError:
            return None, None

    def _data(
            self,
            image_path: pathlib.Path,
            label_path: pathlib.Path) -> Tuple[np.array, np.array]:
        """Return training dataset.

        Args:
            image_path (pathlib.Path): image directory path
            label_path (pathlib.Path): label directory path

        Return:
            dataset (Tuple[np.array, np.array]): training dataset pair

        """
        image_paths = sorted(image_path.glob('*'))
        label_paths = sorted(label_path.glob('*'))
        assert len(label_paths) == len(label_paths)

        pool = multiprocessing.pool.ThreadPool()
        results = []
        for image, label in zip(image_paths, label_paths):
            results.append(pool.apply_async(self._load_data, (image, label)))

        pool.close()
        pool.join()

        X: List = []
        y: List = []
        for res in results:
            image, label = res.get()
            if image is not None and label is not None:
                X.append(image)
                y.append(label)

        X_new = [np.concatenate(X[j-self.window_size:j+self.window_size+1], axis=2)
                 for j in range(self.window_size, len(y)-self.window_size)]
        y_new = [y[j] for j in range(self.window_size, len(y)-self.window_size)]
        return np.array(X_new), np.array(y_new)

    def training_data(self) -> Tuple[np.array, np.array]:
        """Return training dataset.

        Return:
            dataset (Tuple[np.array, np.array]): training dataset pair

        """
        return self._data(self.train_image_directory, self.train_label_directory)

    def eval_data(self) -> Tuple[np.array, np.array]:
        """Return evaluation dataset.

        Return:
            dataset (Tuple[np.array, np.array]): evaluation dataset pair

        """
        return self._data(self.test_image_directory, self.test_label_directory)

    def _data_generator(
            self,
            image_path: pathlib.Path,
            label_path: pathlib.Path) -> Union[tf.keras.utils.Sequence, Generator]:
        """Return training dataset.

        Args:
            image_path (pathlib.Path): image directory path
            label_path (pathlib.Path): label directory path

        Return:
            dataset (Union[tf.keras.utils.Sequence, Generator]): dataset generator

        """
        image_paths = sorted(image_path.glob('*'))
        label_paths = sorted(label_path.glob('*'))
        assert len(label_paths) == len(label_paths)
        sample_num = len(image_paths)
        size = self.batch_size + 2 * self.window_size

        while True:
            indexes = np.arange(sample_num)
            np.random.shuffle(indexes)
            itr_num = int(len(indexes) // (size))

            for i in range(itr_num):
                pool = multiprocessing.pool.ThreadPool()
                batch_ids = indexes[i * size:(i + 1) * size]
                results = []
                X: List = []
                y: List = []

                for j in batch_ids:
                    results.append(
                        pool.apply_async(self._load_data, (image_paths[j], label_paths[j])))

                for res in results:
                    image, label = res.get()
                    if image is not None and label is not None:
                        X.append(image)
                        y.append(label)

                pool.close()
                pool.join()

                X_new = [np.concatenate(X[j-self.window_size:j+self.window_size+1], axis=2)
                         for j in range(self.window_size, len(y)-self.window_size)]
                y_new = [y[j] for j in range(self.window_size, len(y)-self.window_size)]
                yield np.array(X_new), np.array(y_new)

    def training_data_generator(self) -> Union[tf.keras.utils.Sequence, Generator]:
        """Return training dataset.

        Return:
            dataset (Union[tf.keras.utils.Sequence, Generator]): dataset generator

        """
        self.steps_per_epoch = len(list(self.train_image_directory.glob('*'))) // self.batch_size
        return self._data_generator(self.train_image_directory, self.train_label_directory)

    def eval_data_generator(self) -> Union[tf.keras.utils.Sequence, Generator]:
        """Return evaluation dataset.

        Return:
            dataset (Union[tf.keras.utils.Sequence, Generator]): dataset generator

        """
        self.eval_steps_per_epoch = len(list(self.test_image_directory.glob('*'))) // self.batch_size
        return self._data_generator(self.test_image_directory, self.test_label_directory)
