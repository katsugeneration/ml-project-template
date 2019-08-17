# Copyright 2019 Katsuya Shimabukuro. All rights reserved.
# Licensed under the MIT License.
from nose.tools import ok_, eq_
from model.fcnn import FCNNClassifier
from tests.utils.dummy_dataset import DummyDataset


class TestFCNN(object):
    def test_init(self):
        fcnn = FCNNClassifier(
                    dataset=DummyDataset(),
                    hidden_nums=16,
                    dropout_rate=0.8)
        eq_(fcnn.dense1.kernel.shape[1], 16)
        eq_(fcnn.dropout.rate, 0.8)

    def test_train(self):
        fcnn = FCNNClassifier(
                    dataset=DummyDataset(batch_size=1),
                    hidden_nums=16,
                    dropout_rate=0.8)
        history = fcnn.train()

        ok_('acc' in history)
        ok_('loss' in history)
