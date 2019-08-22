from typing import Any
import tensorflow as tf
import numpy as np
from dataset.base import ImageSegmentationDatasetBase
from model.base import KerasModelBase, KerasImageClassifierBase


def Upsampling(inputs, feature_map_shape):
    return tf.image.resize_bilinear(inputs, size=feature_map_shape)


def ConvUpscaleBlock(inputs, n_filters, kernel_size=[3, 3], scale=2):
    """
    Basic conv transpose block for Encoder-Decoder upsampling
    Apply successivly Transposed Convolution, BatchNormalization, ReLU nonlinearity
    """
    inputs = tf.keras.layers.BatchNormalization(fused=True)(inputs)
    inputs = tf.keras.layers.Activation(tf.nn.relu)(inputs)
    inputs = tf.keras.layers.Conv2DTranspose(n_filters, kernel_size=kernel_size, strides=[scale, scale], padding='same')(inputs)
    return inputs


def ConvBlock(inputs, n_filters, kernel_size=[3, 3]):
    """
    Basic conv block for Encoder-Decoder
    Apply successivly Convolution, BatchNormalization, ReLU nonlinearity
    """
    inputs = tf.keras.layers.BatchNormalization(fused=True)(inputs)
    inputs = tf.keras.layers.Activation(tf.nn.relu)(inputs)
    inputs = tf.keras.layers.Conv2D(n_filters, kernel_size, padding='same')(inputs)
    return inputs


def InterpBlock(inputs, level, feature_map_shape):
    # Compute the kernel and stride sizes according to how large the final feature map will be
    # When the kernel size and strides are equal, then we can compute the final feature map size
    # by simply dividing the current size by the kernel or stride size
    # The final feature map sizes are 1x1, 2x2, 3x3, and 6x6. We round to the closest integer
    kernel_size = [int(np.round(float(feature_map_shape[0]) / float(level))), int(np.round(float(feature_map_shape[1]) / float(level)))]
    stride_size = kernel_size

    inputs = tf.keras.layers.MaxPooling2D(kernel_size, strides=stride_size)(inputs)
    inputs = tf.keras.layers.Conv2D(512, [1, 1])(inputs)
    inputs = tf.keras.layers.BatchNormalization(fused=True)(inputs)
    inputs = tf.keras.layers.Activation(tf.nn.relu)(inputs)
    inputs = Upsampling(inputs, feature_map_shape)
    return inputs


def PyramidPoolingModule(inputs, feature_map_shape):
    """
    Build the Pyramid Pooling Module.
    """

    interp_block1 = InterpBlock(inputs, 1, feature_map_shape)
    interp_block2 = InterpBlock(inputs, 2, feature_map_shape)
    interp_block3 = InterpBlock(inputs, 3, feature_map_shape)
    interp_block6 = InterpBlock(inputs, 6, feature_map_shape)

    res = tf.concat([inputs, interp_block6, interp_block3, interp_block2, interp_block1], axis=-1)
    return res


class PSPNet(KerasModelBase):
    """PSPNet implementation.

    Args:
        dataset (ImageSegmentationDatasetBase): dataset object.
        frontend (KerasImageClassifierBase): base classifier object.

    """

    def __init__(
            self,
            dataset: ImageSegmentationDatasetBase,
            frontend: KerasImageClassifierBase,
            **kwargs: Any) -> None:
        """Intialize parameter and build model."""
        # initialize params
        self.dataset = dataset

        inputs = frontend.model.inputs
        hiddens = frontend.model.layers[35].output

        # extract final feature maps
        feature_map_shape = [int(x / 8.0) for x in dataset.input_shape[:2]]
        print(feature_map_shape)
        x = PyramidPoolingModule(hiddens, feature_map_shape=feature_map_shape)

        x = tf.keras.layers.Conv2D(512, [3, 3], padding='same')(x)
        x = tf.keras.layers.BatchNormalization(fused=True)(x)
        x = tf.keras.layers.Activation(tf.nn.relu)(x)

        # upscale featire maps
        x = ConvUpscaleBlock(x, 256, kernel_size=[3, 3], scale=2)
        x = ConvBlock(x, 256)
        x = ConvUpscaleBlock(x, 128, kernel_size=[3, 3], scale=2)
        x = ConvBlock(x, 128)
        x = ConvUpscaleBlock(x, 64, kernel_size=[3, 3], scale=2)
        x = ConvBlock(x, 64)

        # calc likelihood
        outputs = tf.keras.layers.Conv2D(dataset.category_nums, [1, 1])(x)

        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
