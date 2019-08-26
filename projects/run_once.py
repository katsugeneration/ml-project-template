# Copyright 2019 Katsuya Shimabukuro. All rights reserved.
# Licensed under the MIT License.
from typing import Any
import pathlib
import shutil
import importlib
import luigi
import mlflow
import yaml
from projects.base import ProjectBase


class RunOnceProject(ProjectBase):
    runner = luigi.Parameter()
    model = luigi.Parameter('fcnn')
    dataset = luigi.Parameter('mnist')
    model_param_path = luigi.Parameter(default='model.conf.yaml')
    dataset_param_path = luigi.Parameter(default='dataset.conf.yaml')
    logs = luigi.Parameter(default='logs')

    def __init__(
            self,
            *args: Any,
            **kwargs: Any):
        super(RunOnceProject, self).__init__(*args, **kwargs)
        self.experiment_id = 0

        # parameter preprocessing
        with open(self.model_param_path, 'r') as f:
            self.model_params = yaml.load(f)

        with open(self.dataset_param_path, 'r') as f:
            self.dataset_params = yaml.load(f)

        self.parameters = {
            'runner': self.runner,
            'model': self.model,
            'dataset': self.dataset,
            'model_param_path': self.model_param_path,
            'dataset_param_path': self.dataset_param_path,
            'logs': self.logs,
            **self.model_params,
            **self.dataset_params
        }

    def _run(self) -> None:
        logs = pathlib.Path(self.logs)
        if logs.exists():
            shutil.rmtree(str(logs))
        logs.mkdir(parents=True)

        # do runner
        module = importlib.import_module('runner.' + self.runner)
        class_name = "".join(s[:1].upper() + s[1:] for s in self.runner.split('_'))
        c = getattr(module, class_name)
        history = c().run(self.model, self.dataset, self.model_params, self.dataset_params, logs)

        # save to mlflow
        for k in history:
            for i in range(len(history[k])):
                mlflow.log_metric(k, history[k][i], step=i)
        mlflow.log_artifacts(str(logs))
