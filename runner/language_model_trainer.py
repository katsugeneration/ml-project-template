# Copyright 2019 Katsuya Shimabukuro. All rights reserved.
# Licensed under the MIT License.
from typing import Dict, List, Any
import pathlib
import numpy as np
from runner.base import RunnerBase
from dataset.base import DatasetBase
from dataset.imdb import ImdbDataset
from dataset.ptb import PtbDataset
from model.base import ModelBase
from model.rnnlm import RNNLM


class LanguageModelTrainer(RunnerBase):
    """Image recognition task trainning runner."""

    def __init__(self):
        """Initilize parameters."""
        self.datasets = {
            'imdb': ImdbDataset,
            'ptb': PtbDataset,
        }

        self.models = {
            'rnnlm': RNNLM,
        }

    def _run(
            self,
            dataset: DatasetBase,
            model: ModelBase,
            log_path: pathlib.Path) -> Dict[str, List[Any]]:
        """Run task.

        Args:
            dataset (DatasetBase): dataset object.
            model (ModelBase): model object.
            log_path (pathlib.Path): log path object.

        Return:
            history (Dict[str, List[Any]]): task running history.

        """
        # run learning
        history = model.train()
        model.save(log_path.joinpath('model.h5'))

        # save results
        (x_test, y_test) = dataset.eval_data()

        end = dataset.decode(['<E>'])[0]
        for i in range(10):
            predicts = list(x_test[i][:5])
            ret = predicts[-1]
            while ret != end and len(predicts) != dataset.seq_length:
                sequences = dataset.uniformed_sequences([predicts])
                y_pred = model.model.predict(sequences)
                ret = np.random.choice(range(dataset.vocab_size), p=y_pred[0][len(predicts)-1])
                predicts.append(ret)
            print(dataset.decode([y_test[i]]))
            print(dataset.decode([predicts[1:]]))

        return history
