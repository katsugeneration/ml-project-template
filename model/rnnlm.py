# Copyright 2019 Katsuya Shimabukuro. All rights reserved.
# Licensed under the MIT License.
from typing import Any, List
import tensorflow as tf
from model.base import KerasImageClassifierBase


class RNNLM(KerasImageClassifierBase):
    """RNN Language Model."""

    def __init__(
            self,
            hidden_nums: int = 512,
            embedding_dim: int = 100,
            **kwargs: Any) -> None:
        """Intialize parameter and build model."""
        super(RNNLM, self).__init__(**kwargs)

        self._embdeeing = tf.keras.layers.Embedding(
                                self.dataset.vocab_size,
                                embedding_dim,
                                mask_zero=True)

        self._rnn = tf.keras.layers.GRU(
                            hidden_nums,
                            return_sequences=True,
                            stateful=False,
                            recurrent_initializer='glorot_uniform')

        self._dense = tf.keras.layers.Dense(
                                        self.dataset.vocab_size,
                                        kernel_initializer='he_normal')

        # build model
        inputs = tf.keras.Input((self.dataset.seq_length, ))
        x = inputs
        x = self._embdeeing(x)
        x = self._rnn(x)
        x = self._dense(x)

        self.model = tf.keras.Model(inputs=inputs, outputs=x)
        self.setup()
