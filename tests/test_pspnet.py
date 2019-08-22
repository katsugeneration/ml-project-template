# Copyright 2019 Katsuya Shimabukuro. All rights reserved.
# Licensed under the MIT License.
from nose.tools import ok_, eq_
from model.resnet101 import ResNet101
from model.pspnet import PSPNet
from tests.utils.dummy_dataset import SegmentationDummyDataset


class TestResnet(object):
    def test_init(self):
        dataset = SegmentationDummyDataset()
        resnet = ResNet101(dataset=dataset)
        PSPNet(
            dataset=dataset,
            frontend=resnet)

    def test_train(self):
        dataset = SegmentationDummyDataset(batch_size=32)
        resnet = ResNet101(dataset=dataset)
        pspnet = PSPNet(
            epochs=1,
            dataset=dataset,
            frontend=resnet)
        history = pspnet.train()

        ok_('acc' in history)
        ok_('loss' in history)
