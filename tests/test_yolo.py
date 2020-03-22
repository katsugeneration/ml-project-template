# Copyright 2019 Katsuya Shimabukuro. All rights reserved.
# Licensed under the MIT License.
from nose.tools import ok_, eq_
import numpy as np
from model.yolo import YoloV2
from tests.utils.dummy_dataset import OjbjectDetectionDummyDataset, DummyDataset


class TestYoloV2(object):
    def test_init(self):
        yolo = YoloV2(dataset=OjbjectDetectionDummyDataset())
        eq_(yolo.model.outputs[0].shape.as_list(), [None, 13, 13, 5 * (2 + 5)])
        yolo = YoloV2(dataset=OjbjectDetectionDummyDataset(), tiny=False)
        eq_(yolo.model.outputs[0].shape.as_list(), [None, 13, 13, 5 * (2 + 5)])
        yolo = YoloV2(dataset=DummyDataset(), classification=True)
        eq_(yolo.model.outputs[0].shape.as_list(), [None, DummyDataset.category_nums])
        yolo = YoloV2(dataset=OjbjectDetectionDummyDataset(), tiny=False, classification=True)
        eq_(yolo.model.outputs[0].shape.as_list(), [None, DummyDataset.category_nums])

    def test_train(self):
        yolo = YoloV2(dataset=OjbjectDetectionDummyDataset())
        history = yolo.train()
        ok_('loss' in history)

    def test_train_classification(self):
        dataset = DummyDataset()
        DummyDataset()
        yolo = YoloV2(dataset=DummyDataset(), classification=True)
        history = yolo.train()
        ok_('loss' in history)

    def test_preprocess_gt_boxes(self):
        yolo = YoloV2(dataset=OjbjectDetectionDummyDataset())
        boxes = np.array([[
            [0.14666, 0.6672933, 0.194, 0.44109333, 1],
            [0.5164922, 0.46938825, 0.91251564, 0.7482824, 5]]])
        _, ret = yolo.preprocess_gt_boxes(boxes)
        ok_((np.abs(ret[0, 8, 1, 2] - np.array([0.90658, 0.6748133, -0.28044838, 0.04637885, 1.])) <= 1e-3).all())
        ok_((np.abs(ret[0, 6, 6, 4] - np.array([0.7143984, 0.10204697, 0.19402957, 0.05922477, 5.])) <= 1e-3).all())
