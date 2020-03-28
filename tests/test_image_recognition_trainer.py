# Copyright 2019 Katsuya Shimabukuro. All rights reserved.
# Licensed under the MIT License.
import pathlib
from nose.tools import ok_, eq_
from runner.image_recognition_trainer import ImageRecognitionTrainer


class TestImageRecognitionTrainer(object):
    def test_init(self):
        runner = ImageRecognitionTrainer()

    def test_run(self):
        path = pathlib.Path('test_log')
        if not path.exists():
            path.mkdir(parents=True)
        runner = ImageRecognitionTrainer()
        history = runner.run('fcnn', 'mnist', {'epochs': 1}, {'batch_size': 128}, path)

        ok_(path.joinpath('model').exists())
