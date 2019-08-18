# Copyright 2019 Katsuya Shimabukuro. All rights reserved.
# Licensed under the MIT License.
from nose.tools import ok_, eq_
from model.se_block import SEBlock
from tests.utils.dummy_dataset import DummyDataset


class TestSEBlock(object):
    def test_init(self):
        se = SEBlock(
                channels=8,
                r=2)
        eq_(se.se_path[1].units, 4)
        eq_(se.se_path[2].units, 8)
