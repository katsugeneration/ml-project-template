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
        use_xt (bool): whether to use grouped convolution block
        group_num (int): ResNetXt block group number.

    """

    def __init__(
            self,
            block_nums: int = 3,
            weight_decay: float = 1e-4,
            use_se: bool = False,
            use_xt: bool = False,
            group_num: int = 2,
            **kwargs: Any) -> None:
        """Intialize parameter and build model."""
        # initialize params
        super(ResNet, self).__init__(**kwargs)
        self.block_nums = block_nums

        self.channels = [16, 32, 64]

        self.start_conv = tf.keras.layers.Conv2D(
                                self.channels[0] if not use_xt else self.channels[0] * 8,
                                kernel_size=(3, 3),
                                padding="same",
                                kernel_initializer="he_normal",
                                kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                input_shape=self.dataset.input_shape,
                                use_bias=False)

        self.blocks: List = []
        for c in self.channels:
            for i in range(self.block_nums):
                subsampling = i == 0 and c > 16
                out_c = c*4
                in_c = c
                strides = (2, 2) if subsampling else (1, 1)

                initial_path = [
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Activation(tf.nn.relu)
                ]

                intermidiate = tf.keras.layers.Conv2D(
                                        in_c,
                                        kernel_size=(3, 3),
                                        padding="same",
                                        kernel_initializer="he_normal",
                                        kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                        use_bias=False)
                if use_xt:
                    in_c = c * 8
                    out_c = c * 16
                    intermidiate = []
                    for j in range(group_num):
                        intermidiate.append(
                            tf.keras.layers.Conv2D(
                                        in_c // group_num,
                                        kernel_size=(3, 3),
                                        padding="same",
                                        kernel_initializer="he_normal",
                                        kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                        use_bias=False)
                        )

                residual_path = [
                    initial_path[0],
                    initial_path[1],
                    tf.keras.layers.Conv2D(
                            in_c,
                            kernel_size=(1, 1),
                            padding="same",
                            strides=strides,
                            kernel_initializer="he_normal",
                            kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                            use_bias=False),

                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Activation(tf.nn.relu),
                    intermidiate,

                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Activation(tf.nn.relu),
                    tf.keras.layers.Conv2D(
                            out_c,
                            kernel_size=(1, 1),
                            padding="same",
                            kernel_initializer="he_normal",
                            kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                            use_bias=False),

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
                                kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                use_bias=False)
                    ]
                else:
                    identity_path = []

                self.blocks.append({
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
                kernel_regularizer=tf.keras.regularizers.l2(weight_decay))
        ]

        # build model
        inputs = tf.keras.layers.Input(self.dataset.input_shape)
        x = self.start_conv(inputs)

        for b in self.blocks:
            residual = x
            for idx, l in enumerate(b["residual_path"]):
                if use_xt and isinstance(l, list):
                    c = residual.shape[-1] // group_num
                    residual = tf.concat([
                        l[j](residual[:, :, :, j*c:(j+1)*c])
                        for j in range(group_num)], axis=-1)
                else:
                    residual = l(residual)
                    if idx == 1:
                        top_node = residual
            for l in b["identity_path"]:
                x = l(top_node)
            x = tf.keras.layers.Add()([x, residual])

        for l in self.end_block:
            x = l(x)

        self.model = tf.keras.Model(inputs=inputs, outputs=x)
        self.setup()
