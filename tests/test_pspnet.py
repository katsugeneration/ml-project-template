# Copyright 2019 Katsuya Shimabukuro. All rights reserved.
# Licensed under the MIT License.
from nose.tools import ok_, eq_
from model.pspnet import PSPNet
from tests.utils.dummy_dataset import SegmentationDummyDataset


class TestPspnet(object):
    def test_init(self):
        dataset = SegmentationDummyDataset()
        PSPNet(
            dataset=dataset,
            frontend_name='resnet101')

    def test_train(self):
        dataset = SegmentationDummyDataset(batch_size=32)
        pspnet = PSPNet(
            epochs=1,
            dataset=dataset,
            frontend_name='resnet101')
        history = pspnet.train()

        ok_('loss' in history)

    def test_train_gdl(self):
        dataset = SegmentationDummyDataset(batch_size=32)
        pspnet = PSPNet(
            epochs=1,
            dataset=dataset,
            frontend_name='resnet101',
            generarized_dice_loss={'alpha': 0.01})
        history = pspnet.train()

        ok_('loss' in history)
