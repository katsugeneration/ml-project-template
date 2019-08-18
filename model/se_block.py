# Copyright 2019 Katsuya Shimabukuro. All rights reserved.
# Licensed under the MIT License.
from typing import Any
import tensorflow as tf


class SEBlock(tf.keras.layers.Layer):
    """Squeeze-and-Excitation block implementation.

    Args:
        channels (int): size of input channels.
        r (int): bottleneck ratio.

    """

    def __init__(
            self,
            channels: int,
            r: int = 16,
            **kwargs: Any):
        """Initilize parameters and layers."""
        super(SEBlock, self).__init__(**kwargs)
        self.channels = channels
        self.r = r

        # Squeeze
        self.se_path = [
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(
                channels//r,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                kernel_initializer="he_normal"),
            tf.keras.layers.Dense(
                channels,
                activation="sigmoid",
                kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                kernel_initializer="he_normal"),
        ]

    def call(self, inputs):
        x1 = inputs
        for l in self.se_path:
            x1 = l(x1)
        outputs = tf.keras.layers.Add()([inputs, tf.keras.layers.Multiply()([inputs, x1])])
        return outputs
