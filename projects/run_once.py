# Copyright 2019 Katsuya Shimabukuro. All rights reserved.
# Licensed under the MIT License.
from typing import Any, List, Optional
import re
import pathlib
import shutil
import importlib
import luigi
import mlflow
import yaml
from projects.base import ProjectBase
from projects.data import create_data_prepare


class RunOnceProject(ProjectBase):
    """Run the project once."""

    runner = luigi.Parameter()
    model = luigi.Parameter('fcnn')
    dataset = luigi.Parameter('mnist')
    model_param_path = luigi.Parameter(default='model.conf.yaml')
    dataset_param_path = luigi.Parameter(default='dataset.conf.yaml')
    preprocess_param_path = luigi.Parameter(default='preprocess.conf.yaml')
    logs = luigi.Parameter(default='logs')

    def __init__(
            self,
            *args: Any,
            **kwargs: Any):
        super(RunOnceProject, self).__init__(*args, **kwargs)
        self.experiment_id = 0
        self.update = True

        # parameter preprocessing
        with open(self.model_param_path, 'r') as f:
            self.model_params = yaml.full_load(f)

        with open(self.dataset_param_path, 'r') as f:
            self.dataset_params = yaml.full_load(f)

        with open(self.preprocess_param_path, 'r') as f:
            self.preprocess_params = yaml.full_load(f)

        self.parameters = {
            'runner': self.runner,
            'model': self.model,
            'dataset': self.dataset,
            'model_param_path': self.model_param_path,
            'dataset_param_path': self.dataset_param_path,
            'preprocess_param_path': self.preprocess_param_path,
            'logs': self.logs,
            **self.model_params,
            **self.dataset_params
        }

        self.run_name = '_'.join([self.runner, self.model, self.dataset])

        self.before_project = create_data_prepare(
                                {k: getattr(importlib.import_module(".".join(v.split('.')[:-1])), v.split('.')[-1])
                                    for k, v in self.preprocess_params['projects'].items()},
                                self.preprocess_params['parameters'])

    def requires(self) -> List[Optional[ProjectBase]]:
        """Dependency projects."""
        return [self.before_project]

    def _run(self) -> None:
        logs = pathlib.Path(self.logs)
        if logs.exists():
            shutil.rmtree(str(logs))
        logs.mkdir(parents=True)

        # update parameter from local data.
        variables = {
            'before_artifact_directory': self.before_project.artifact_directory,
        }
        pattern = re.compile(r'{{(.*?)}}', flags=re.I | re.M)
        self.dataset_params = {k: eval(pattern.match(v).groups()[0].strip(), variables)
                               if isinstance(v, str) and pattern.match(v) else v
                               for k, v in self.dataset_params.items()}

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
