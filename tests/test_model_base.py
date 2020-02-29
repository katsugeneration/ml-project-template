# Copyright 2019 Katsuya Shimabukuro. All rights reserved.
# Licensed under the MIT License.
from nose.tools import ok_, eq_
import numpy as np
import tensorflow as tf
from model.base import KerasLanguageModelBase
from tests.utils.dummy_dataset import TextDummyDataset


class TestKerasLanguageModelBase(object):
    def test_loss(self):
        model = KerasLanguageModelBase(TextDummyDataset([[]]))
        loss = model._loss(
            tf.constant(np.array([[1, 2, 0, 0, 0], [2, 1, 0, 0, 0]])),
            tf.constant(np.array([[[0.1, 0.1, 0.8]] * 5, [[0.1, 0.3, 0.6]] * 5])))
        expected_loss = -(np.log(0.1) + np.log(0.8) + np.log(0.6) + np.log(0.3))
        ok_(abs(loss.numpy().sum() - expected_loss) < 1e-5)
