# Copyright 2019 Katsuya Shimabukuro. All rights reserved.
# Licensed under the MIT License.
from typing import Any, List
import functools
import tensorflow as tf
import numpy as np
from model.base import KerasClassifierBase


def drop_connect(inputs, is_training, survival_prob):
    """Drop the entire conv with given survival probability."""
    # "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
    if not is_training:
        return inputs

    # Compute tensor.
    batch_size = tf.shape(inputs)[0]
    random_tensor = survival_prob
    random_tensor += tf.random.uniform([batch_size, 1, 1, 1], dtype=inputs.dtype)
    binary_tensor = tf.floor(random_tensor)
    # Unlike conventional way that multiply survival_prob at test time, here we
    # divide survival_prob at training time, such that no addition compute is
    # needed at test time.
    output = tf.math.divide(inputs, survival_prob) * binary_tensor
    return output


def dense_kernel_initializer(shape, dtype=None, partition_info=None):
    """Initialization for dense kernels.
    This initialization is equal to
        tf.variance_scaling_initializer(scale=1.0/3.0, mode='fan_out',
                                        distribution='uniform').
    It is written out explicitly here for clarity.
    Args:
        shape: shape of variable
        dtype: dtype of variable
        partition_info: unused
    Returns:
        an initialization for the variable
    """
    del partition_info
    init_range = 1.0 / np.sqrt(shape[1])
    return tf.random.uniform(shape, -init_range, init_range, dtype=dtype)


class MBConvBlock(tf.keras.layers.Layer):
    """A class of MBConv: Mobile Inverted Residual Bottleneck.
    Attributes:
        endpoints: dict. A list of internal tensors.
    """

    def __init__(self, input_filters, output_filters, expand_ratio, kernel_size, strides, se_ratio):
        """Initializes a MBConv block.

        Args:
            block_args: BlockArgs, arguments to create a Block.
            global_params: GlobalParams, a set of global parameters.
        """
        super(MBConvBlock, self).__init__()
        self.input_filters = input_filters
        self.output_filters = output_filters
        self.expand_ratio = expand_ratio
        self.kernel_size = kernel_size
        self.strides = strides
        self.se_ratio = se_ratio
        self._kernel_initializer = tf.keras.initializers.he_normal()

        self._relu_fn = tf.nn.swish

        self.conv_cls = tf.keras.layers.Conv2D
        self.depthwise_conv_cls = tf.keras.layers.DepthwiseConv2D

        # Builds the block accordings to arguments.
        self._build()

    def _build(self):
        """Builds block according to the arguments."""

        filters = self.input_filters * self.expand_ratio
        kernel_size = self.kernel_size

        # Expansion phase. Called if not using fused convolutions and expansion
        # phase is necessary.
        self._expand_conv = self.conv_cls(
            filters=filters,
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=self._kernel_initializer,
            padding='same',
            use_bias=False)
        self._bn0 = tf.keras.layers.BatchNormalization()

        # Depth-wise convolution phase. Called if not using fused convolutions.
        self._depthwise_conv = self.depthwise_conv_cls(
            kernel_size=[kernel_size, kernel_size],
            strides=self.strides,
            depthwise_initializer=self._kernel_initializer,
            padding='same',
            use_bias=False)
        self._bn1 = tf.keras.layers.BatchNormalization()

        num_reduced_filters = max(
            1, int(self.input_filters * self.se_ratio))
        # Squeeze and Excitation layer.
        self._se_reduce = tf.keras.layers.Conv2D(
            num_reduced_filters,
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=self._kernel_initializer,
            padding='same',
            use_bias=True)
        self._se_expand = tf.keras.layers.Conv2D(
            filters,
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=self._kernel_initializer,
            padding='same',
            use_bias=True)

        # Output phase.
        filters = self.output_filters
        self._project_conv = self.conv_cls(
            filters=filters,
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=self._kernel_initializer,
            padding='same',
            use_bias=False)
        self._bn2 = tf.keras.layers.BatchNormalization()

    def _call_se(self, input_tensor):
        """Call Squeeze and Excitation layer.
        Args:
            input_tensor: Tensor, a single input tensor for Squeeze/Excitation layer.
        Returns:
            A output tensor, which should have the same shape as input.
        """
        se_tensor = tf.reduce_mean(input_tensor, (1, 2), keepdims=True)
        se_tensor = self._se_expand(self._relu_fn(self._se_reduce(se_tensor)))
        return tf.sigmoid(se_tensor) * input_tensor

    def call(self, inputs, training=True, survival_prob=None):
        """Implementation of call().
        Args:
            inputs: the inputs tensor.
            training: boolean, whether the model is constructed for training.
            survival_prob: float, between 0 to 1, drop connect rate.

        Returns:
            A output tensor.
        """
        x = inputs

        expand_conv_fn = self._expand_conv
        depthwise_conv_fn = self._depthwise_conv
        project_conv_fn = self._project_conv

        # Expand
        if self.expand_ratio != 1:
            x = self._relu_fn(self._bn0(expand_conv_fn(x), training=training))

        # Depthwise
        x = self._relu_fn(self._bn1(depthwise_conv_fn(x), training=training))

        # SE Block
        x = self._call_se(x)

        # Projection
        x = self._bn2(project_conv_fn(x), training=training)

        # Add identity so that quantization-aware training can insert quantization
        # ops correctly.
        x = tf.identity(x)
        if all(s == 1 for s in self.strides) and self.input_filters == self.output_filters:
            x = drop_connect(x, training, survival_prob)
            x = tf.add(x, inputs)
        return x


class EfficientNet(KerasClassifierBase):
    """EfficientNet implementation bottleneck and pre-ctivation style.

    Args:
        weight_decay (float): L2 regurarization weight parameter.

    """

    def __init__(
            self,
            weight_decay: float = 1e-4,
            width_coefficient: float = 1.0,
            depth_coefficient: float = 1.0,
            depth_divisor: int = 8,
            min_depth: float = None,
            **kwargs: Any) -> None:
        """Intialize parameter and build model."""
        # initialize params
        super(EfficientNet, self).__init__(**kwargs)
        self.channels = [32, 16, 24, 40, 80, 112, 192, 320, 1280]
        self.blocks = [1, 1, 2, 2, 3, 3, 4, 1, 1]
        self.kernels = [3, 3, 3, 5, 3, 5, 5, 3, 1]
        self.strides = [2, 1, 2, 2, 2, 1, 2, 1, 1]
        self.dropout_rate = 0.2
        self.survival_prob = 0.8
        self._kernel_initializer = tf.keras.initializers.he_normal()

        self.start_conv = tf.keras.layers.Conv2D(
                                self.round_filters(
                                    self.channels[0],
                                    width_coefficient,
                                    depth_divisor,
                                    min_depth),
                                kernel_size=(self.kernels[0], self.kernels[0]),
                                padding="same",
                                strides=(self.strides[0], self.strides[0]),
                                kernel_initializer=self._kernel_initializer,
                                kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                use_bias=False)
        self.start_bn = tf.keras.layers.BatchNormalization()

        self.hiddens: List = []

        for i in range(1, len(self.blocks) - 1):
            input_filters = self.round_filters(self.channels[i - 1], width_coefficient, depth_divisor, min_depth)
            output_filters = self.round_filters(self.channels[i], width_coefficient, depth_divisor, min_depth)
            kernel_size = self.kernels[i]
            strides = (self.strides[i], self.strides[i])

            for j in range(self.blocks[i]):
                if j > 0:
                    strides = (1, 1)
                    input_filters = output_filters
                self.hiddens.append(MBConvBlock(input_filters, output_filters, 6, kernel_size, strides, 0.25))

        self.end_conv = tf.keras.layers.Conv2D(
                                self.round_filters(
                                    self.channels[-1],
                                    width_coefficient,
                                    depth_divisor,
                                    min_depth),
                                kernel_size=(self.kernels[-1], self.kernels[-1]),
                                padding="same",
                                strides=(self.strides[-1], self.strides[-1]),
                                kernel_initializer=self._kernel_initializer,
                                kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                use_bias=False)
        self.end_bn = tf.keras.layers.BatchNormalization()

        self.end_block = [
            self.end_conv,
            self.end_bn,
            tf.keras.layers.Activation(tf.nn.swish),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(self.dropout_rate),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(
                self.dataset.category_nums,
                activation=tf.nn.softmax,
                kernel_initializer=dense_kernel_initializer,
                kernel_regularizer=tf.keras.regularizers.l2(weight_decay))
        ]

        # build model
        inputs = tf.keras.layers.Input(self.dataset.input_shape)
        x = tf.image.resize(inputs, (224, 224))
        x = self.start_conv(x)
        x = self.start_bn(x)
        x = tf.keras.layers.Activation(tf.nn.swish)(x)

        for idx, h in enumerate(self.hiddens):
            drop_rate = 1.0 - self.survival_prob
            survival_prob = 1.0 - drop_rate * float(idx) / len(self.hiddens)
            x = h(x, survival_prob=survival_prob)

        for l in self.end_block:
            x = l(x)

        self.model = tf.keras.Model(inputs=inputs, outputs=x)
        self.setup()

    def round_filters(self, filters, width_coefficient=None, depth_divisor=None, min_depth=None):
        """Round number of filters based on depth multiplier."""
        multiplier = width_coefficient
        divisor = depth_divisor
        min_depth = min_depth
        if not multiplier:
            return filters

        filters *= multiplier
        min_depth = min_depth or divisor
        new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_filters < 0.9 * filters:
            new_filters += divisor
        return int(new_filters)
