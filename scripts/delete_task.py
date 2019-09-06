# Copyright 2019 Katsuya Shimabukuro. All rights reserved.
# Licensed under the MIT License.
import shutil
import pathlib
import mlflow


client = mlflow.tracking.MlflowClient()
for run in client.search_runs('0', run_view_type=mlflow.entities.ViewType.DELETED_ONLY):
    shutil.rmtree(pathlib.Path(
            run.info.artifact_uri.split('file://')[1]).parent)
