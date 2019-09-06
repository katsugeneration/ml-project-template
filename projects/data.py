# Copyright 2019 Katsuya Shimabukuro. All rights reserved.
# Licensed under the MIT License.
from typing import Any, Callable, Dict, List, Optional
import re
import importlib
import pathlib
import functools
from projects.base import ProjectBase
from projects.utils import mlflow_utils


class DataPrepareProject(ProjectBase):
    """Data preprocessing project base class."""

    def __init__(
            self,
            *args: Any,
            **kwargs: Any):
        super(DataPrepareProject, self).__init__(*args, **kwargs)
        self.experiment_id = 0
        self.run_func: List[Callable]
        self.parameters: Dict
        self.before_project: ProjectBase

    def _run(self) -> None:
        """Project running."""
        before_artifact_directory = None
        if self.before_project is not None:
            before_artifact_directory = self.before_project.artifact_directory

        variables = {
            'before_artifact_directory': before_artifact_directory,
            'preprocess_params': self.parameters,
            'search_preprocess_directory': search_preprocess_directory,
        }
        pattern = re.compile(r'{{(.*?)}}', flags=re.I | re.M)
        func_params = {k: eval(pattern.sub(r'\1', v).strip(), variables)
                       if isinstance(v, str) and pattern.match(v) else v
                       for k, v in self.parameters.items()}

        self.run_func[0](**dict({
            'artifact_directory': self.artifact_directory,
            'before_artifact_directory': before_artifact_directory}, **func_params))

    def requires(self) -> List[ProjectBase]:
        """Dependency projects."""
        return [self.before_project]


def _get_runname(
        func: Callable) -> str:
    """Return run name from function.

    Args:
        func (Callable): target function.

    Return:
        runname (str): run name string.

    """
    return '.'.join([func.__module__, func.__name__])


def _get_valid_parameters(
        func: Callable,
        parameters: Dict) -> Dict:
    """Extract valid parameter for target function.

    Args:
        func (Callable): target function.
        parameters (Dict): candidate parameters.

    Return:
        valids (Dict): valid parameters.

    """
    return {k: parameters[k] for k in func.__code__.co_varnames if k in parameters}


def create_data_prepare(
        projects: Dict[str, Callable],
        parameters: Dict,
        update_task: str = '') -> Optional[DataPrepareProject]:
    """Create data prepare projects.

    Args:
        projects (Dict[str, Callable]): task name and task function dictionary.
        parameters (Dict): named parameters which have some project uses.
        update_task (str): first update task name.

    Return
        project (Optional[DataPrepareProject]): last project.

    """
    project_names = list(projects.keys())
    before_project: Optional[DataPrepareProject] = None
    update = False
    for task in project_names:
        if update_task == task:
            update = True
        params = _get_valid_parameters(projects[task], parameters)
        run_func = projects[task]

        new_project_class = type(task, (DataPrepareProject, ), {
                                    'run_func': [run_func],
                                    'parameters': params,
                                    'update': update,
                                    'before_project': before_project,
                                    'run_name': _get_runname(run_func)})
        new_project = new_project_class()
        before_project = new_project

    return before_project


def search_preprocess_directory(
        func_name: str,
        parameters: Dict) -> pathlib.Path:
    """Return preprocess artifact directory.

    Args:
        func_name (str): function full name started module name.
        parameters (Dict): target parameters.

    Return:
        path (pathlib.Path): artifact directory path.

    """
    run_func = getattr(importlib.import_module(".".join(func_name.split('.')[:-1])), func_name.split('.')[-1])
    return mlflow_utils.run_to_run_directory(
                mlflow_utils.search_run_object(
                    _get_runname(run_func),
                    _get_valid_parameters(run_func, parameters)))
