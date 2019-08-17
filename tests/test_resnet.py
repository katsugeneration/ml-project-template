# Copyright 2019 Katsuya Shimabukuro. All rights reserved.
# Licensed under the MIT License.
from nose.tools import ok_, eq_
from model.resnet import ResNet
from tests.utils.dummy_dataset import DummyDataset


class TestResnet(object):
    def test_init(self):
        resnet = ResNet(
                    dataset=DummyDataset(),
                    block_nums=1)
        eq_(len(resnet.blocks), 3)

    def test_train(self):
        resnet = ResNet(
                    dataset=DummyDataset(batch_size=1),
                    block_nums=1,
                    epochs=1)
        history = resnet.train()

        ok_('acc' in history)
        ok_('loss' in history)
