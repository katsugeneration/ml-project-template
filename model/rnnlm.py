# Copyright 2019 Katsuya Shimabukuro. All rights reserved.
# Licensed under the MIT License.
from typing import Any, List
import tensorflow as tf
from model.base import KerasLanguageModelBase


class RNNLM(KerasLanguageModelBase):
    """RNN Language Model."""

    def __init__(
            self,
            hidden_nums: int = 512,
            embedding_dim: int = 100,
            weight_decay: float = 1e-4,
            dropout_rate: float = 0.5,
            layer_nums: int = 2,
            **kwargs: Any) -> None:
        """Intialize parameter and build model."""
        super(RNNLM, self).__init__(**kwargs)

        self._embdeeing = tf.keras.layers.Embedding(
                                self.dataset.vocab_size,
                                embedding_dim,
                                mask_zero=True)

        self._rnn: List[tf.keras.layers.Layer] = []
        for _ in range(layer_nums):
            self._rnn.append(tf.keras.layers.LSTM(
                                hidden_nums,
                                return_sequences=True,
                                stateful=False,
                                dropout=dropout_rate,
                                recurrent_initializer='glorot_uniform',
                                kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))

        self._dense = tf.keras.layers.Dense(
                                        self.dataset.vocab_size,
                                        activation=tf.nn.softmax,
                                        kernel_initializer='he_normal',
                                        kernel_regularizer=tf.keras.regularizers.l2(weight_decay))

        # build model
        inputs = tf.keras.Input((self.dataset.seq_length, ))
        x = inputs
        x = self._embdeeing(x)
        for i in range(layer_nums):
            x = self._rnn[i](x)
        x = self._dense(x)

        self.model = tf.keras.Model(inputs=inputs, outputs=x)
        self.setup()
