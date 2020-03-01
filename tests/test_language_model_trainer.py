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
        history = runner.run(
            'rnnlm',
            'ptb',
            {'epochs': 1},
            {
                'path': pathlib.Path(__file__).parent.joinpath('data/ptb'),
                'seq_length': 20,
                'batch_size': 5
            },
            path)

        ok_(path.joinpath('model.h5').exists())
