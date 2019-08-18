# Copyright 2019 Katsuya Shimabukuro. All rights reserved.
# Licensed under the MIT License.
import numpy as np
import tensorflow as tf


class MixupGenerator(tf.keras.utils.Sequence):
    """Mixup image generator.

    source cord references to https://qiita.com/yu4u/items/70aa007346ec73b7ff05.

    Args:
        X_train (np.array): input images.
        y_train (np.array): input labels.
        batch_size (int): iteration batch size.
        alpha (float): mix ratio.
        shuffle (bool): whether to shuffle data.
        datagen (tf.keras.preprocessing.image.ImageDataGenerator): base data generator

    """

    def __init__(
            self,
            X_train: np.array,
            y_train: np.array,
            batch_size: int = 32,
            alpha: float = 0.1,
            shuffle: bool = True,
            datagen: tf.keras.preprocessing.image.ImageDataGenerator = None):
        """Initialize parameters."""
        self.X_train = X_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.alpha = alpha
        self.shuffle = shuffle
        self.sample_num = len(X_train)
        self.datagen = datagen
        self.__update_indexes()

    def __len__(self):
        """Return item steps."""
        return self.max_itr_num

    def __getitem__(self, idx):
        """Return images and labels appled mixup.

        Return:
            X (np.array): images appled mixup.
            y (np.array): labels appled mixup.

        """
        if idx >= self.max_itr_num:
            raise ValueError('{} is over index'.format(idx))
        batch_ids = self.indexes[idx * self.batch_size * 2:(idx + 1) * self.batch_size * 2]
        X, y = self.__data_generation(batch_ids)

        return X, y

    def __update_indexes(self):
        self.indexes = self.__get_exploration_order()
        self.max_itr_num = int(len(self.indexes) // (self.batch_size * 2))
        self.itr_num = 0

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
        rate = np.random.beta(self.alpha, self.alpha, self.batch_size)
        X_l = rate.reshape(self.batch_size, 1, 1, 1)
        y_l = rate.reshape(self.batch_size, 1)

        X = X1 * X_l + X2 * (1 - X_l)
        y = y1 * y_l + y2 * (1 - y_l)

        if self.datagen:
            for i in range(self.batch_size):
                X[i] = self.datagen.random_transform(X[i])

        return X, y
