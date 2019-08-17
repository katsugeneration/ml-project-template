from typing import Any
import tensorflow as tf
from model.base import KerasImageClassifierBase


class FCNNClassifier(KerasImageClassifierBase):
    """Fully connected neural network classifier.

    Args:
        hidden_nums (int): number of hidden layer units.
        dropout_rate (float): dropout ratio.

    """

    def __init__(
            self,
            hidden_nums: int,
            dropout_rate: float,
            **kwargs: Any) -> None:
        """Intialize parameter and build model."""
        # initialize params
        super(FCNNClassifier, self).__init__(**kwargs)
        self.hidden_nums = hidden_nums
        self.dropout_rate = dropout_rate

        self.flatten = tf.keras.layers.Flatten(input_shape=self.dataset.input_shape)
        self.dense1 = tf.keras.layers.Dense(self.hidden_nums, activation=tf.nn.relu)
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.dense2 = tf.keras.layers.Dense(self.dataset.category_nums, activation=tf.nn.softmax)

        # build model
        inputs = tf.keras.layers.Input(self.dataset.input_shape)
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dropout(x)
        outputs = self.dense2(x)

        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        self.setup()
