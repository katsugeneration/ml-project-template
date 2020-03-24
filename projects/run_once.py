# Copyright 2019 Katsuya Shimabukuro. All rights reserved.
# Licensed under the MIT License.
from typing import Any, List, Optional
import re
import importlib
import queue
from multiprocessing import Process, Queue
import luigi
import mlflow
import yaml
from projects.base import ProjectBase
from projects.utils.python_utils import get_attribute
from projects.utils.mlflow_utils import search_run_directory
from projects.data import create_data_prepare, search_preprocess_directory


class RunOnceProject(ProjectBase):
    """Run the project once."""

    runner = luigi.Parameter()
    model = luigi.Parameter('fcnn')
    dataset = luigi.Parameter('mnist')
    param_path = luigi.Parameter(default='params.yaml')

    def __init__(
            self,
            *args: Any,
            **kwargs: Any):
        super(RunOnceProject, self).__init__(*args, **kwargs)
        self.experiment_id = 0
        self.update = True

        # parameter preprocessing
        with open(self.param_path, 'r') as f:
            params = yaml.full_load(f)

        self.model_params = params['model']
        self.dataset_params = params['dataset']
        self.preprocess_params = params['preprocess']

        self.parameters = {
            'runner': self.runner,
            'model': self.model,
            'dataset': self.dataset,
            'param_path': self.param_path,
            **self.model_params,
            **self.dataset_params
        }

        self.run_name = '_'.join([self.runner, self.model, self.dataset])

        if 'projects' not in self.preprocess_params:
            self.preprocess_params['projects'] = {}
        if 'parameters' not in self.preprocess_params:
            self.preprocess_params['parameters'] = {}
        if 'update_task' not in self.preprocess_params:
            self.preprocess_params['update_task'] = ''

        self.before_project = create_data_prepare(
                                {k: get_attribute(v) for k, v in self.preprocess_params['projects'].items()},
                                self.preprocess_params['parameters'],
                                self.preprocess_params['update_task'])

    def requires(self) -> List[Optional[ProjectBase]]:
        """Dependency projects."""
        return [self.before_project]

    def __run_once(self, c, results):
        results.put(c.run(self.model, self.dataset, self.model_params, self.dataset_params, self.artifact_directory))

    def _run(self) -> None:
        # update parameter from local data.
        before_artifact_directory = self.before_project.artifact_directory if self.before_project is not None else None
        variables = {
            'before_artifact_directory': before_artifact_directory,
            'preprocess_params': self.preprocess_params['parameters'],
            'search_preprocess_directory': search_preprocess_directory,
            'search_run_directory': search_run_directory
        }
        pattern = re.compile(r'{{(.*?)}}', flags=re.I | re.M)
        self.model_params = {k: eval(pattern.sub(r'\1', v).strip(), variables)
                             if isinstance(v, str) and pattern.match(v) else v
                             for k, v in self.model_params.items()}
        self.dataset_params = {k: eval(pattern.sub(r'\1', v).strip(), variables)
                               if isinstance(v, str) and pattern.match(v) else v
                               for k, v in self.dataset_params.items()}

        # do runner
        module = importlib.import_module('runner.' + self.runner)
        class_name = "".join(s[:1].upper() + s[1:] for s in self.runner.split('_'))
        c = getattr(module, class_name)

        history = None
        while history is None:
            results: Queue = Queue()
            p = Process(target=self.__run_once, args=(c(), results))
            p.start()
            p.join()

            try:
                history = results.get(False)
            except queue.Empty:
                history = None
                pass

        # save to mlflow
        for k in history:
            for i in range(len(history[k])):
                mlflow.log_metric(k, history[k][i], step=i)
