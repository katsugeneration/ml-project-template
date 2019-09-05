# Copyright 2019 Katsuya Shimabukuro. All rights reserved.
# Licensed under the MIT License.
from typing import Dict
import pathlib
import mlflow


def search_run_object(
        run_name: str,
        parameters: Dict,
        experiment_id: int = 0) -> mlflow.entities.Run:
    """Return specific task artifact directory path.

    Args:
        run_name (str): mlflow task run name.
        parameters (Dict): mlflow task parameters.
        experiment_id (int): mlflow experiment ID.

    Return:
        artifact_path (pathlib.Path): task artifact directory path.

    """
    client = mlflow.tracking.MlflowClient()
    filter_string = " and ".join(
        ["params.{} = '{}'".format(k, v) for k, v in parameters.items()] +
        ["tags.mlflow.runName = '{}'".format(run_name)])
    for run in client.search_runs(
            [str(experiment_id)],
            filter_string=filter_string,
            run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY):
        return run

    return None


def run_to_run_directory(
        run: mlflow.entities.Run) -> pathlib.Path:
    """Return run object artifact directory.

    Args:
        run (mlflow.entities.Run) mlflow run object.

    Return:
        path (pathlib.Path): mlflow artifact directory.

    """
    return pathlib.Path(
                mlflow.tracking.artifact_utils.get_artifact_uri(
                    run.info.run_id).split('://')[1])
