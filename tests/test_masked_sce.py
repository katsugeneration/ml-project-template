# Copyright 2019 Katsuya Shimabukuro. All rights reserved.
# Licensed under the MIT License.
from nose.tools import ok_, eq_
import numpy as np
import tensorflow as tf
from model.utils.losses import MaskedSparseCategoricalCrossentropy
from tests.utils.dummy_dataset import TextDummyDataset


class TestMaskedSparseCategoricalCrossentropy(object):
    def test_loss(self):
        loss = MaskedSparseCategoricalCrossentropy()
        _loss = loss(
            tf.constant(np.array([[1, 2, 0, 0, 0], [2, 1, 0, 0, 0]])),
            tf.constant(np.array([[[0.1, 0.1, 0.8]] * 5, [[0.1, 0.3, 0.6]] * 5])))
        expected_loss = -(np.log(0.1) + np.log(0.8) + np.log(0.6) + np.log(0.3)) / 4.
        ok_(abs(_loss.numpy().sum() - expected_loss) < 1e-5)
