# Copyright 2019 Katsuya Shimabukuro. All rights reserved.
# Licensed under the MIT License.
from nose.tools import ok_, eq_
import numpy as np
from model.yolo import YoloV2
from tests.utils.dummy_dataset import OjbjectDetectionDummyDataset


class TestYoloV2(object):
    def test_init(self):
        yolo = YoloV2(dataset=OjbjectDetectionDummyDataset())
        eq_(yolo.model.outputs[0].shape.as_list(), [None, 13, 13, 5 * (2 + 5)])

    def test_train(self):
        yolo = YoloV2(dataset=OjbjectDetectionDummyDataset())
        history = yolo.train()
        ok_('loss' in history)
