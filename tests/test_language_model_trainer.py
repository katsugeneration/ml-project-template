# Copyright 2019 Katsuya Shimabukuro. All rights reserved.
# Licensed under the MIT License.
import pathlib
from nose.tools import ok_, eq_
from runner.language_model_trainer import LanguageModelTrainer


class TestLanguageModelTrainer(object):
    def test_init(self):
        runner = LanguageModelTrainer()

    def test_run(self):
        path = pathlib.Path('test_log')
        if not path.exists():
            path.mkdir(parents=True)
        runner = LanguageModelTrainer()
        history = runner.run('rnnlm', 'imdb', {'epochs': 1}, {'batch_size': 128}, path)

        ok_(path.joinpath('model.h5').exists())
