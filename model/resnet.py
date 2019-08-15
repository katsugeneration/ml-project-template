from typing import Any
import tensorflow as tf
from model.base import KerasImageClassifierBase


class ResNet(KerasImageClassifierBase):
    """ResNet implementation bottleneck and pre-ctivation style.

    Args:
        block_nums (int): number of block units.

    """

    def __init__(
            self,
            block_nums: int,
            **kwargs: Any) -> None:
        """Intialize parameter and build model."""
        # initialize params
        super(ResNet, self).__init__(**kwargs)
        self.block_nums = block_nums

        channels = [16, 32, 64]

        inputs = tf.keras.layers.Input(shape=self.dataset.input_shape)
        if len(self.dataset.input_shape) == 2:
            x = tf.expand_dims(inputs, axis=2)
        else:
            x = inputs
        x = tf.keras.layers.Conv2D(
            channels[0],
            kernel_size=(3, 3),
            padding="same",
            kernel_initializer="he_normal",
            kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)

        for c in channels:
            for i in range(self.block_nums):
                subsampling = i == 0 and c > 16
                out_c = c*4
                strides = (2, 2) if subsampling else (1, 1)

                x = tf.keras.layers.BatchNormalization()(x)
                x = tf.keras.layers.Activation(tf.nn.relu)(x)
                y = tf.keras.layers.Conv2D(
                        c,
                        kernel_size=(1, 1),
                        padding="same",
                        strides=strides,
                        kernel_initializer="he_normal",
                        kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)

                y = tf.keras.layers.BatchNormalization()(y)
                y = tf.keras.layers.Activation(tf.nn.relu)(y)
                y = tf.keras.layers.Conv2D(
                        c,
                        kernel_size=(3, 3),
                        padding="same",
                        kernel_initializer="he_normal",
                        kernel_regularizer=tf.keras.regularizers.l2(1e-4))(y)

                y = tf.keras.layers.BatchNormalization()(y)
                y = tf.keras.layers.Activation(tf.nn.relu)(y)
                y = tf.keras.layers.Conv2D(
                        out_c,
                        kernel_size=(1, 1),
                        padding="same",
                        kernel_initializer="he_normal",
                        kernel_regularizer=tf.keras.regularizers.l2(1e-4))(y)

                if i == 0:
                    x = tf.keras.layers.Conv2D(
                        out_c,
                        kernel_size=(1, 1),
                        strides=strides,
                        padding="same",
                        kernel_initializer="he_normal",
                        kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
                x = tf.keras.layers.Add()([x, y])

        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(tf.nn.relu)(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Flatten()(x)
        outputs = tf.keras.layers.Dense(
                    self.dataset.category_nums,
                    activation=tf.nn.softmax,
                    kernel_initializer="he_normal",
                    kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.type = "resnet" + str(9 * self.block_nums + 2)
        self.model = model

        self.compile()
