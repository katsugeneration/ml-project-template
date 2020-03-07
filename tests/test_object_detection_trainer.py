# Copyright 2019 Katsuya Shimabukuro. All rights reserved.
# Licensed under the MIT License.
import pathlib
from nose.tools import ok_, eq_
from runner.object_detection_trainer import ObjectDetectionTrainer


class TestObjectDetectionTrainer(object):
    def test_init(self):
        runner = ObjectDetectionTrainer()

    def test_run(self):
        path = pathlib.Path('test_log')
        if not path.exists():
            path.mkdir(parents=True)
        runner = ObjectDetectionTrainer()
        history = runner.run(
                    'yolo',
                    'mscoco',
                    {'epochs': 1},
                    {
                        'batch_size': 1,
                        'train_image_directory': pathlib.Path(__file__).parent.joinpath('data/mscoco/image'),
                        'train_label_path': pathlib.Path(__file__).parent.joinpath('data/mscoco/annotations.json'),
                        'test_image_directory': pathlib.Path(__file__).parent.joinpath('data/mscoco/image'),
                        'test_label_path': pathlib.Path(__file__).parent.joinpath('data/mscoco/annotations.json')
                    },
                    path)

        ok_(path.joinpath('model.h5').exists())
