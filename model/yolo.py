# Copyright 2020 Katsuya Shimabukuro. All rights reserved.
# Licensed under the MIT License.
from typing import Any, Tuple, Optional
import tensorflow as tf
from model.base import KerasObjectDetectionBase


class YoloConvBlock(tf.keras.layers.Layer):
    """Yolo convolution block implementaton."""

    def __init__(
            self,
            filters: int,
            kernel: int,
            stride: int,
            weight_decay: float,
            activation_cls: Optional[Any] = tf.keras.layers.LeakyReLU) -> None:
        """Initialize yolo convolution block.

        Args:
            filters (int): convolution output filter size.
            kernel (int): convolution kernel size.
            stride (int): convolution stride size.
            weight_decay (float): weight decay's weight.
            activation_cls (Optional[str]): activation class.

        """
        super(YoloConvBlock, self).__init__()

        self._conv = tf.keras.layers.Conv2D(
                        filters,
                        kernel_size=kernel,
                        strides=stride,
                        padding="same",
                        kernel_initializer="he_normal",
                        kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                        use_bias=False)
        self._bn = tf.keras.layers.BatchNormalization()
        self._activation = None
        if activation_cls is not None:
            self._activation = activation_cls()

    def call(
            self,
            inputs: tf.Variable) -> tf.Variable:
        """Layer forward process.

        Args:
            inputs (tf.Variable): input array whose shape is (BatchSize, Height, Width, Channel)

        Return:
            outputs (tf.Variable): output array whose shape is (BatchSize, Height, Width, Channel)

        """
        x = self._conv(inputs)
        x = self._bn(x)
        if self._activation is not None:
            x = self._activation(x)
        return x


class YoloBottleneckBlock(tf.keras.layers.Layer):
    """Yolo convolution block implementaton."""

    def __init__(
            self,
            filters: int,
            bottleneck_filters: int,
            weight_decay: float) -> None:
        """Initialize yolo convolution block.

        Args:
            filters (int): convolution output filter size.
            bottleneck_filters (int): bottleneck convolution output filter size.
            weight_decay (float): weight decay's weight.

        """
        super(YoloBottleneckBlock, self).__init__()

        self._conv1 = YoloConvBlock(
                            filters=filters,
                            kernel=3,
                            stride=1,
                            weight_decay=weight_decay)

        self._conv2 = YoloConvBlock(
                            filters=bottleneck_filters,
                            kernel=1,
                            stride=1,
                            weight_decay=weight_decay)

        self._conv3 = YoloConvBlock(
                            filters=filters,
                            kernel=3,
                            stride=1,
                            weight_decay=weight_decay)

    def call(
            self,
            inputs: tf.Variable) -> tf.Variable:
        """Layer forward process.

        Args:
            inputs (tf.Variable): input array whose shape is (BatchSize, Height, Width, Channel)

        Return:
            outputs (tf.Variable): output array whose shape is (BatchSize, Height, Width, Channel)

        """
        x = self._conv1(inputs)
        x = self._conv2(x)
        x = self._conv3(x)
        return x


class YoloV2(KerasObjectDetectionBase):
    """YOLO v2 implementation with darknet-19."""

    def __init__(
            self,
            weight_decay: float = 5e-4,
            resize_shape: Tuple[int, int] = (416, 416),
            **kwargs: Any) -> None:
        """Intialize parameter and build model.

        Args:
            weight_decay (float): weight decay's weight.
            resize_shape (Tuple[int]): model input shape.

        """
        super(YoloV2, self).__init__(**kwargs)

        self.anchors = [(1.08, 1.19), (3.42, 4.41), (6.63, 11.38), (9.42, 5.11), (16.62, 10.52)]
        filters = [32, 64, 128, 256, 512, 1024]

        inputs = tf.keras.layers.Input(self.dataset.input_shape)
        x = tf.image.resize(inputs, resize_shape, method=tf.image.ResizeMethod.BILINEAR)

        # darknet body
        for i in range(2):
            x = YoloConvBlock(
                    filters=filters[i],
                    kernel=3,
                    stride=1,
                    weight_decay=weight_decay)(x)
            x = tf.keras.layers.MaxPool2D()(x)

        for i in range(2, 4):
            x = YoloBottleneckBlock(
                    filters=filters[i],
                    bottleneck_filters=filters[i-1],
                    weight_decay=weight_decay)(x)
            x = tf.keras.layers.MaxPool2D()(x)

        for i in range(4, 6):
            x = YoloBottleneckBlock(
                    filters=filters[i],
                    bottleneck_filters=filters[i-1],
                    weight_decay=weight_decay)(x)
            x = YoloConvBlock(
                    filters=filters[i-1],
                    kernel=1,
                    stride=1,
                    weight_decay=weight_decay)(x)
            x = YoloConvBlock(
                    filters=filters[i],
                    kernel=3,
                    stride=1,
                    weight_decay=weight_decay)(x)

            if i == 4:
                _fine_grained = x
                x = tf.keras.layers.MaxPool2D()(x)

        # normal path
        x = YoloConvBlock(
                filters=filters[-1],
                kernel=3,
                stride=1,
                weight_decay=weight_decay)(x)
        x = YoloConvBlock(
                filters=filters[-1],
                kernel=3,
                stride=1,
                weight_decay=weight_decay)(x)

        # fine grained path
        _fine_grained = YoloConvBlock(
                filters=filters[1],
                kernel=1,
                stride=1,
                weight_decay=weight_decay)(_fine_grained)
        _fine_grained = tf.nn.space_to_depth(_fine_grained, 2)

        x = tf.concat([x, _fine_grained], axis=-1)
        x = YoloConvBlock(
                filters=filters[-1],
                kernel=3,
                stride=1,
                weight_decay=weight_decay)(x)
        outputs = tf.keras.layers.Conv2D(
                filters=len(self.anchors) * (self.dataset.category_nums + 5),
                kernel_size=1,
                strides=1,
                padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                use_bias=True)(x)

        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        self.setup()
