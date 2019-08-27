# Copyright 2019 Katsuya Shimabukuro. All rights reserved.
# Licensed under the MIT License.
from typing import Any, Callable, Dict, List, Optional
import functools
from projects.base import ProjectBase


class DataPrepareProject(ProjectBase):
    """Data preprocessing project base class."""

    def __init__(
            self,
            *args: Any,
            **kwargs: Any):
        super(DataPrepareProject, self).__init__(*args, **kwargs)
        self.experiment_id = 0
        self.run_func: Callable
        self.parameters: Dict
        self.before_project: ProjectBase

    def _run(self) -> None:
        """Project running."""
        self.run_func()

    def requires(self) -> List[ProjectBase]:
        """Dependency projects."""
        return [self.before_project]


def create_data_prepare(
        projects: Dict[str, Callable],
        parameters: Dict) -> Optional[DataPrepareProject]:
    """Create data prepare projects.

    Args:
        projects (Dict[str, Callable]): task name and task function dictionary.
        parameters (Dict): named parameters which have some project uses.

    Return
        project (Optional[DataPrepareProject]): last project.

    """
    project_names = reversed(list(projects.keys()))
    before_project: Optional[DataPrepareProject] = None
    for task in project_names:
        params = {k: parameters[k] for k in projects[task].__code__.co_varnames if k in parameters}
        run_func = functools.partial(projects[task], **params)
        new_project_class = type(task, (DataPrepareProject, ), {
                                    'run_func': run_func,
                                    'parameters': params,
                                    'before_project': before_project})
        new_project = new_project_class()
        before_project = new_project

    return before_project
