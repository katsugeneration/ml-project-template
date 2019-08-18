# Copyright 2019 Katsuya Shimabukuro. All rights reserved.
# Licensed under the MIT License.
from typing import Any, List
import tensorflow as tf
from model.base import KerasImageClassifierBase
from model.se_block import SEBlock


class ResNet(KerasImageClassifierBase):
    """ResNet implementation bottleneck and pre-ctivation style.

    Args:
        block_nums (int): number of block units.
        use_se (bool): whether to use squeeze-and-excitation block

    """

    def __init__(
            self,
            block_nums: int = 3,
            use_se: bool = False,
            **kwargs: Any) -> None:
        """Intialize parameter and build model."""
        # initialize params
        super(ResNet, self).__init__(**kwargs)
        self.block_nums = block_nums

        self.channels = [16, 32, 64]

        self.start_conv = tf.keras.layers.Conv2D(
                                self.channels[0],
                                kernel_size=(3, 3),
                                padding="same",
                                kernel_initializer="he_normal",
                                kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                                input_shape=self.dataset.input_shape)

        self.blocks: List = []
        for c in self.channels:
            for i in range(self.block_nums):
                subsampling = i == 0 and c > 16
                out_c = c*4
                strides = (2, 2) if subsampling else (1, 1)

                initial_path = [
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Activation(tf.nn.relu)
                ]

                residual_path = [
                    tf.keras.layers.Conv2D(
                            c,
                            kernel_size=(1, 1),
                            padding="same",
                            strides=strides,
                            kernel_initializer="he_normal",
                            kernel_regularizer=tf.keras.regularizers.l2(1e-4)),

                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Activation(tf.nn.relu),
                    tf.keras.layers.Conv2D(
                            c,
                            kernel_size=(3, 3),
                            padding="same",
                            kernel_initializer="he_normal",
                            kernel_regularizer=tf.keras.regularizers.l2(1e-4)),

                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Activation(tf.nn.relu),
                    tf.keras.layers.Conv2D(
                            out_c,
                            kernel_size=(1, 1),
                            padding="same",
                            kernel_initializer="he_normal",
                            kernel_regularizer=tf.keras.regularizers.l2(1e-4)),

                ]

                if use_se:
                    residual_path.append(SEBlock(out_c))

                if i == 0:
                    identity_path = [
                            tf.keras.layers.Conv2D(
                                out_c,
                                kernel_size=(1, 1),
                                strides=strides,
                                padding="same",
                                kernel_initializer="he_normal",
                                kernel_regularizer=tf.keras.regularizers.l2(1e-4))
                    ]
                else:
                    identity_path = []

                self.blocks.append({
                    "initial_path": initial_path,
                    "residual_path": residual_path,
                    "identity_path": identity_path,
                })

        self.end_block = [
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.nn.relu),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(
                self.dataset.category_nums,
                activation=tf.nn.softmax,
                kernel_initializer="he_normal",
                kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        ]

        # build model
        inputs = tf.keras.layers.Input(self.dataset.input_shape)
        x = self.start_conv(inputs)

        for b in self.blocks:
            for l in b["initial_path"]:
                x = l(x)
            residual = x
            for l in b["residual_path"]:
                residual = l(residual)
            for l in b["identity_path"]:
                x = l(x)
            x = tf.keras.layers.Add()([x, residual])

        for l in self.end_block:
            x = l(x)

        self.model = tf.keras.Model(inputs=inputs, outputs=x)
        self.setup()
