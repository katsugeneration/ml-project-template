# Copyright 2019 Katsuya Shimabukuro. All rights reserved.
# Licensed under the MIT License.
from typing import Any
import tensorflow as tf
from model.base import KerasImageClassifierBase


class ConvolutionalNet(KerasImageClassifierBase):
    """CNN implementation with batch normalization use VGG style.

    Args:
        block_nums (int): number of block units.

    """

    def __init__(
            self,
            block_nums: int = 2,
            **kwargs: Any) -> None:
        """Intialize parameter and build model."""
        # initialize params
        super(ConvolutionalNet, self).__init__(**kwargs)
        self.block_nums = block_nums

        self.channels = [64, 128, 256]

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Input(self.dataset.input_shape))
        for _ in range(self.block_nums):
            model.add(tf.keras.layers.Conv2D(self.channels[0], (3, 3), padding='same', kernel_initializer='he_normal'))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.MaxPool2D(strides=(2, 2)))
        for _ in range(self.block_nums):
            model.add(tf.keras.layers.Conv2D(self.channels[1], (3, 3), padding='same', kernel_initializer='he_normal'))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.MaxPool2D(strides=(2, 2)))
        for _ in range(self.block_nums):
            model.add(tf.keras.layers.Conv2D(self.channels[2], (3, 3), padding='same', kernel_initializer='he_normal'))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.MaxPool2D(strides=(2, 2)))
        model.add(tf.keras.layers.GlobalAveragePooling2D())
        model.add(tf.keras.layers.Dense(self.dataset.category_nums, activation='softmax'))

        self.model = model
        self.setup()
