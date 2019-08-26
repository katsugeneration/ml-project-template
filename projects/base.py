# Copyright 2019 Katsuya Shimabukuro. All rights reserved.
# Licensed under the MIT License.
from typing import Any, List
from abc import abstractmethod
import pathlib
import luigi
import mlflow


class ProjectBase(luigi.Task):
    """Mlflow project base extend luigi task."""

    def __init__(
            self,
            *args: Any,
            **kwargs: Any):
        super(ProjectBase, self).__init__(*args, **kwargs)
        self.experiment_id: int
        self.parameters: dict
        self._run_object: mlflow.entities.Run

    def output(self) -> List[str]:
        """Project output path."""
        try:
            return [str(self.artifact_directory.joinpath('.finish'))]
        except Exception:
            return []

    @abstractmethod
    def _run(self) -> None:
        """Project running."""
        pass

    def run(self) -> None:
        """Luigi task run method implementation."""
        with mlflow.start_run(experiment_id=self.experiment_id) as active_run:
            mlflow.log_params(self.parameters)
            self._run_object = active_run
            self._run()
        for p in self.output():
            pathlib.Path(p).touch()

    @property
    def artifact_directory(self) -> pathlib.Path:
        """Path to artifact directory.

        Return:
            path (pathlib.Path):  path to mlflow project artifacts.

        """
        return pathlib.Path(
                mlflow.tracking.artifact_utils.get_artifact_uri(
                    self.run_object.info.run_id).split('://')[1])

    @property
    def run_object(self) -> mlflow.entities.Run:
        """Relational mlflow run object.

        Return:
            run_object (mlflow.entities.Run): mlflow run object.

        """
        if self._run_object:
            return self._run_object

        client = mlflow.tracking.MlflowClient()
        filter_string = " and ".join([
            "params.{} = '{}'".format(k, v) for k, v in self.parameters.items()])
        for run in client.search_runs(
                [self.experiment_id],
                filter_string=filter_string,
                run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY,
                order_by='start_time DESC'):
            return run

        return None
