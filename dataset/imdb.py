# Copyright 2019 Katsuya Shimabukuro. All rights reserved.
# Licensed under the MIT License.
from typing import Any
import tensorflow as tf
from dataset.base import BinaryTextDataset


class ImdbDataset(BinaryTextDataset):
    """IMDB dataset loader."""

    def __init__(
            self,
            data_normalize_style: str = 'standardization',
            **kwargs: Any) -> None:
        """Load data and setup preprocessing."""
        super(ImdbDataset, self).__init__(**kwargs)
        imdb = tf.keras.datasets.imdb
        (x_train, y_train), (x_test, y_test) = imdb.load_data()
        word_index = imdb.get_word_index()
        words = {idx: word for word, idx in word_index.items()}

        self.x_train = [" ".join([words[idx-3] if idx >= 3 else '<UNK>' for idx in line if idx >= 2])
                        for line in x_train]
        self.y_train = y_train
        self.x_test = [" ".join([words[idx-3] if idx >= 3 else '<UNK>' for idx in line if idx >= 2])
                       for line in x_test]
        self.y_test = y_test
