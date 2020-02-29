# Copyright 2020 Katsuya Shimabukuro. All rights reserved.
# Licensed under the MIT License.
import tensorflow as tf


class MaskedSparseCategoricalCrossentropy():
    """SparseCategoricalCrossentropy without padding mask."""

    def __call__(self, label, pred, **kwargs):
        """Calculate loss.

        Args:
            label (tf.Tensor): sequence label with shape (B, Seq).
            pred (tf.Tensor): sequence label prediction likelihood with shape (B, Seq, Token) in [0, 1].

        Return:
            loss (tf.Tensor): mean loss float value without padding mask.

        """
        mask = tf.math.logical_not(tf.math.equal(label, 0))
        loss = tf.keras.losses.sparse_categorical_crossentropy(label, pred)

        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask

        return tf.reduce_mean(tf.reduce_sum(loss, axis=1) / tf.reduce_sum(mask, axis=1))
