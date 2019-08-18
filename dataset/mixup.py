# Copyright 2019 Katsuya Shimabukuro. All rights reserved.
# Licensed under the MIT License.
from typing import Generator
import numpy as np
import tensorflow as tf


class MixupGenerator(tf.keras.utils.Sequence):
    """Mixup image generator.

    source cord references to https://qiita.com/yu4u/items/70aa007346ec73b7ff05.

    Args:
        alpha (float): mix ratio.
        datagen (tf.keras.preprocessing.image.ImageDataGenerator): base data generator

    """

    def __init__(
            self,
            alpha: float = 1.0,
            datagen: tf.keras.preprocessing.image.ImageDataGenerator = None):
        """Initialize parameters."""
        self.alpha = alpha
        self.datagen = datagen

    def flow(
            self,
            X_train: np.array,
            y_train: np.array,
            batch_size: int = 32,
            shuffle: bool = True) -> Generator:
        """Return images and labels appled mixup.

        Args:
            X_train (np.array): input images.
            y_train (np.array): input labels.
            batch_size (int): iteration batch size.
            shuffle (bool): whether to shuffle data.

        Return:
            generator (Generator): images and labels appled mixup.

        """
        self.X_train = X_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sample_num = len(X_train)

        while True:
            indexes = self.__get_exploration_order()
            itr_num = int(len(indexes) // (self.batch_size * 2))

            for i in range(itr_num):
                batch_ids = indexes[i * self.batch_size * 2:(i + 1) * self.batch_size * 2]
                X, y = self.__data_generation(batch_ids)

                yield X, y

    def __get_exploration_order(self):
        indexes = np.arange(self.sample_num)

        if self.shuffle:
            np.random.shuffle(indexes)

        return indexes

    def __data_generation(self, batch_ids):
        _, h, w, c = self.X_train.shape
        _, class_num = self.y_train.shape
        X1 = self.X_train[batch_ids[:self.batch_size]]
        X2 = self.X_train[batch_ids[self.batch_size:]]
        y1 = self.y_train[batch_ids[:self.batch_size]]
        y2 = self.y_train[batch_ids[self.batch_size:]]
        l = np.random.beta(self.alpha, self.alpha, self.batch_size)
        X_l = l.reshape(self.batch_size, 1, 1, 1)
        y_l = l.reshape(self.batch_size, 1)

        X = X1 * X_l + X2 * (1 - X_l)
        y = y1 * y_l + y2 * (1 - y_l)

        if self.datagen:
            for i in range(self.batch_size):
                X[i] = self.datagen.random_transform(X[i])

        return X, y
