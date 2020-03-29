# Copyright 2019 Katsuya Shimabukuro. All rights reserved.
# Licensed under the MIT License.
from nose.tools import ok_, eq_
import subprocess
import pathlib
import shutil
import time
import numpy as np
from PIL import Image, ImageDraw
from model.yolo import YoloV2
from dataset.mscoco import MSCococDatectionDataset, convert_tfrecord
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
        yolo = YoloV2(dataset=OjbjectDetectionDummyDataset(), tiny=False)
        history = yolo.train()
        ok_('loss' in history)

    def test_train_classification(self):
        yolo = YoloV2(dataset=DummyDataset(), classification=True)
        history = yolo.train()
        ok_('loss' in history)

    def test_load_pretrained_model(self):
        path = pathlib.Path(__file__).parent.joinpath('test_path')
        path.mkdir(parents=True, exist_ok=True)
        yolo = YoloV2(dataset=DummyDataset(), classification=True, epochs=1)
        history = yolo.train()
        yolo.save(path.joinpath('model'))
        weight = yolo.model.layers[2].weights[0]

        yolo = YoloV2(dataset=OjbjectDetectionDummyDataset(), restore_path=path.joinpath('model'))
        ok_((np.abs(weight - yolo.model.layers[2].weights[0]) <= 1e-5).all())
        shutil.rmtree(path)

    def test_preprocess_gt_boxes(self):
        yolo = YoloV2(dataset=OjbjectDetectionDummyDataset())
        boxes = np.array([[
            [0.14666, 0.6672933, 0.194, 0.44109333, 1],
            [0.5164922, 0.46938825, 0.91251564, 0.7482824, 5]]])
        _, ret = yolo.preprocess_gt_boxes(boxes)
        ok_((np.abs(ret[0, 8, 1, 2] - np.array([0.90658, 0.6748133, -0.28044838, 0.04637885, 1.])) <= 1e-3).all())
        ok_((np.abs(ret[0, 6, 6, 4] - np.array([0.7143984, 0.10204697, 0.19402957, 0.05922477, 5.])) <= 1e-3).all())

    def test_preprocess_gt_boxes_with_real_data(self):
        path = pathlib.Path(__file__).parent.joinpath('after_path')
        path.mkdir(parents=True, exist_ok=True)

        convert_tfrecord(path, pathlib.Path(__file__).parent.joinpath('data/mscoco/'))
        dataset = MSCococDatectionDataset(
                train_image_directory=pathlib.Path(__file__).parent.joinpath('data/mscoco/train2014'),
                test_image_directory=pathlib.Path(__file__).parent.joinpath('data/mscoco/val2014'),
                train_data_directory=path.joinpath('train'),
                test_data_directory=path.joinpath('test'),
                batch_size=2)
        
        image, label = next(dataset.training_data_generator())
        image = Image.fromarray((image.numpy()[0] * 255.0).astype(np.uint8))
        
        yolo = YoloV2(dataset=dataset)
        detect_masks, matching_boxes = yolo.preprocess_gt_boxes(label)
        non_zeroes = np.argwhere(detect_masks.numpy()[0][..., 0])

        draw = ImageDraw.Draw(image)
        for box in non_zeroes:
            x, y, w, h, category = matching_boxes.numpy()[0, box[0], box[1], box[2]]
            x = (x + box[1]) * 32
            w = np.exp(w) * yolo.anchors[box[2]][0] * 32
            y = (y + box[0]) * 32
            h = np.exp(h) * yolo.anchors[box[2]][1] * 32
            draw.rectangle(((x - w / 2, y - h / 2), (x + w / 2, y + h / 2)), outline='red', width=5)
        image.save('test.png', "PNG")
        subprocess.run(('open test.png'), shell=True)
        time.sleep(1)
        pathlib.Path('test.png').unlink()
        shutil.rmtree(path)
