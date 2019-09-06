# Copyright 2019 Katsuya Shimabukuro. All rights reserved.
# Licensed under the MIT License.
from typing import Any, List, Union
from abc import abstractmethod
import pathlib
import luigi
import mlflow
from projects.utils import mlflow_utils


class ProjectBase(luigi.Task):
    """Mlflow project base extend luigi task."""

    """For avoiding luigi cache parameter."""
    name = luigi.Parameter(default='default')

    def __init__(
            self,
            *args: Any,
            **kwargs: Any):
        super(ProjectBase, self).__init__(*args, **kwargs)
        self.experiment_id: int
        self.run_name: str
        self.parameters: dict
        self.update: bool
        self._run_object: mlflow.entities.Run = None

    def output(self) -> Union[luigi.LocalTarget, List]:
        """Project output path."""
        try:
            return luigi.LocalTarget(str(self.artifact_directory.joinpath('.finish')))
        except Exception:
            return []

    @abstractmethod
    def _run(self) -> None:
        """Project running."""
        pass

    def run(self) -> None:
        """Luigi task run method implementation."""
        with mlflow.start_run(experiment_id=self.experiment_id, run_name=self.run_name) as active_run:
            for k, v in self.parameters.items():
                mlflow.log_param(k, mlflow_utils.convert_valid_log_param_value(v))
            self._run_object = active_run
            self._run()
        self.output().open('w').close()

    @property
    def artifact_directory(self) -> pathlib.Path:
        """Path to artifact directory.

        Return:
            path (pathlib.Path):  path to mlflow project artifacts.

        """
        return mlflow_utils.run_to_run_directory(self.run_object)

    @property
    def run_object(self) -> mlflow.entities.Run:
        """Relational mlflow run object.

        Return:
            run_object (mlflow.entities.Run): mlflow run object.

        """
        if self._run_object:
            return self._run_object
        if not self.update:
            return mlflow_utils.search_run_object(self.run_name, self.parameters, self.experiment_id)
        return None
