# Copyright 2019 Katsuya Shimabukuro. All rights reserved.
# Licensed under the MIT License.
from nose.tools import ok_, eq_
from model.efficientnet import EfficientNet
from model.se_block import SEBlock
from tests.utils.dummy_dataset import DummyDataset


class TestEfficientNet(object):
    def test_init(self):
        efficientnet = EfficientNet(
                    dataset=DummyDataset(),
                    block_nums=1)
        
    def test_train(self):
        efficientnet = EfficientNet(
                    dataset=DummyDataset(),
                    block_nums=1)
        history = efficientnet.train()

        ok_('accuracy' in history)
        ok_('loss' in history)
