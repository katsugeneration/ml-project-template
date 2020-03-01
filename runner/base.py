# Copyright 2019 Katsuya Shimabukuro. All rights reserved.
# Licensed under the MIT License.
from typing import Dict, List, Any, Generic, TypeVar
import pathlib
from dataset.base import DatasetBase
from model.base import ModelBase

T = TypeVar('T', bound=DatasetBase)
V = TypeVar('V', bound=ModelBase)


class RunnerBase(Generic[T, V]):
    """Task runner base class."""

    def __init__(self):
        """Initilize parameters."""
        self.datasets: dict
        self.models: dict

    def run(
            self,
            model_name: str,
            dataset_name: str,
            model_params: dict,
            dataset_params: dict,
            log_path: pathlib.Path) -> Dict[str, List[Any]]:
        """Run task.

        Args:
            model_name (str): run model name.
            dataset_name (str): run dataset name.
            model_params (dict): model initialized parametaers.
            dataset_params (dict): dataset initialized parametaers.
            log_path (pathlib.Path): log path object.

        Return:
            history (Dict[str, List[Any]]): task running history.

        """
        # prepare dataset
        if dataset_name in self.datasets:
            dataset = self.datasets[dataset_name](**dataset_params)
        else:
            raise ValueError("dataset {} not found".format(dataset_name))

        # prepare model
        model_params['dataset'] = dataset
        if model_name in self.models:
            model = self.models[model_name](**model_params)
        else:
            raise ValueError("model {} not found".format(model_name))

        history = self._run(dataset, model, log_path)
        return history

    def _run(
            self,
            dataset: T,
            model: V,
            log_path: pathlib.Path) -> Dict[str, List[Any]]:
        """Run task.

        Args:
            dataset (DatasetBase): dataset object.
            model (ModelBase): model object.
            log_path (pathlib.Path): log path object.

        Return:
            history (Dict[str, List[Any]]): task running history.

        """
        raise NotImplementedError
