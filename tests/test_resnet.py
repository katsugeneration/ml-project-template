# Copyright 2019 Katsuya Shimabukuro. All rights reserved.
# Licensed under the MIT License.
from nose.tools import ok_, eq_
from model.resnet import ResNet
from model.se_block import SEBlock
from tests.utils.dummy_dataset import DummyDataset


class TestResnet(object):
    def test_init(self):
        resnet = ResNet(
                    dataset=DummyDataset(),
                    block_nums=1)
        eq_(len(resnet.blocks), 3)

    def test_init_usese(self):
        resnet = ResNet(
                    dataset=DummyDataset(),
                    block_nums=1,
                    use_se=True)
        ok_(isinstance(resnet.blocks[0]['residual_path'][-1], SEBlock))

    def test_init_usext(self):
        resnet = ResNet(
                    dataset=DummyDataset(),
                    block_nums=1,
                    use_xt=True)
        ok_(isinstance(resnet.blocks[0]['residual_path'][-4], list))
        
    def test_train(self):
        resnet = ResNet(
                    dataset=DummyDataset(batch_size=2),
                    block_nums=1,
                    epochs=1)
        history = resnet.train()

        ok_('loss' in history)
        
    def test_train_usext(self):
        resnet = ResNet(
                    dataset=DummyDataset(batch_size=2),
                    block_nums=1,
                    epochs=1,
                    use_xt=True)
        history = resnet.train()

        ok_('loss' in history)
