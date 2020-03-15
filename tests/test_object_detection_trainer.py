# Copyright 2019 Katsuya Shimabukuro. All rights reserved.
# Licensed under the MIT License.
import pathlib
import shutil
from nose.tools import ok_, eq_
from dataset.mscoco import convert_tfrecord
from runner.object_detection_trainer import ObjectDetectionTrainer


class TestObjectDetectionTrainer(object):
    def test_init(self):
        runner = ObjectDetectionTrainer()

    def test_run(self):
        path = pathlib.Path('test_log')
        if not path.exists():
            path.mkdir(parents=True)
        runner = ObjectDetectionTrainer()

        data_path = pathlib.Path(__file__).parent.joinpath('after_path')
        data_path.mkdir(parents=True, exist_ok=True)
        convert_tfrecord(data_path, pathlib.Path(__file__).parent.joinpath('data/mscoco/'))
        history = runner.run(
                    'yolo',
                    'mscoco',
                    {'epochs': 1},
                    {
                        'batch_size': 2,
                        'train_image_directory': pathlib.Path(__file__).parent.joinpath('data/mscoco/train2014'),
                        'test_image_directory': pathlib.Path(__file__).parent.joinpath('data/mscoco/val2014'),
                        'train_data_directory': data_path.joinpath('train'),
                        'test_data_directory': data_path.joinpath('test')
                    },
                    path)

        ok_(path.joinpath('model.h5').exists())
        shutil.rmtree(data_path)
