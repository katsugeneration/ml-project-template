# Copyright 2019 Katsuya Shimabukuro. All rights reserved.
# Licensed under the MIT License.
from typing import Any, List
import tensorflow as tf
from model.base import KerasImageClassifierBase


class ResNet101(KerasImageClassifierBase):
    """ResNet implementation bottleneck and pre-ctivation style with 101 layers."""

    def __init__(
            self,
            **kwargs: Any) -> None:
        """Intialize parameter and build model."""
        # initialize params
        super(ResNet101, self).__init__(**kwargs)
        self.channels = [64, 128, 256, 512]
        self.block_nums = [3, 4, 23, 3]

        self.start_block = [
            tf.keras.layers.ZeroPadding2D(padding=(3, 3)),
            tf.keras.layers.Conv2D(
                        self.channels[0],
                        kernel_size=(7, 7),
                        padding="valid",
                        strides=(2, 2),
                        kernel_initializer="he_normal",
                        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                        input_shape=self.dataset.input_shape),
            tf.keras.layers.ZeroPadding2D(padding=(1, 1)),
            tf.keras.layers.MaxPooling2D(
                        pool_size=(3, 3),
                        padding="valid",
                        strides=(2, 2))
        ]

        self.blocks: List = []
        for k, c in enumerate(self.channels):
            for i in range(self.block_nums[k]):
                subsampling = i == 0 and k < 3
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

                if i == 0:
                    identity_path = [
                            tf.keras.layers.Conv2D(
                                out_c,
                                kernel_size=(1, 1),
                                strides=strides,
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
        x = inputs

        for l in self.start_block:
            x = l(x)

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
