# Copyright 2019 Katsuya Shimabukuro. All rights reserved.
# Licensed under the MIT License.
from typing import Dict, List, Any
import pathlib
from matplotlib import pyplot as plt
import seaborn
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from runner.base import RunnerBase
from dataset.base import DatasetBase
from dataset.imdb import ImdbDataset
from model.base import ModelBase
from model.rnnlm import RNNLM


class LanguageModelTrainer(RunnerBase):
    """Image recognition task trainning runner."""

    def __init__(self):
        """Initilize parameters."""
        self.datasets = {
            'imdb': ImdbDataset,
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
        y_pred = model.model.predict(x_test[:100])
        y_pred = np.argmax(y_pred, axis=1)
        print(dataset.decode(y_test[:1]))
        print(dataset.decode(y_pred[:1]))

        return history
