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

        # buid model
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=self.dataset.input_shape),
            tf.keras.layers.Dense(self.hidden_nums, activation=tf.nn.relu),
            tf.keras.layers.Dropout(self.dropout_rate),
            tf.keras.layers.Dense(self.dataset.category_nums, activation=tf.nn.softmax)
        ])
        self.compile()
