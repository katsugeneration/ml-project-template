# Copyright 2019 Katsuya Shimabukuro. All rights reserved.
# Licensed under the MIT License.
from nose.tools import ok_, eq_
import luigi
from projects.base import ProjectBase
from tests.utils.dummy_schedular import DummyFactory


class DummyProject(ProjectBase):

    param1 = luigi.IntParameter()

    def __init__(
            self,
            *args,
            **kwargs):
        super(DummyProject, self).__init__(*args, **kwargs)
        self.experiment_id = '0'
        self.parameters = {
            "param1": self.param1,
            "param2": 2
        }
        self._ran = False
        self.run_name = 'dummy_project'

    def _run(self):
        self._ran = True


class TestProjectBase(object):
    def test_init(self):
        project = DummyProject(param1=10)
        eq_(project.parameters, {"param1": 10, "param2": 2})

    def test_run(self):
        project = DummyProject(param1=10, name='1')
        run_result = luigi.build([project], worker_scheduler_factory=DummyFactory())
        ok_(run_result)
        ok_(project._ran)
        ok_(project.output().exists())
        project.output().remove()

    def test_run_twice(self):
        project = DummyProject(param1=10, name='2')
        run_result = luigi.build([project], worker_scheduler_factory=DummyFactory())
        ok_(run_result)
        ok_(project._ran)
        ok_(project.output().exists())

        project = DummyProject(param1=10, name='3')
        run_result = luigi.build([project], worker_scheduler_factory=DummyFactory())
        project.output().remove()
        ok_(run_result)
        ok_(not project._ran)

    def test_run_twice_case_update(self):
        project = DummyProject(param1=10, name='2')
        run_result = luigi.build([project], worker_scheduler_factory=DummyFactory())
        ok_(run_result)
        ok_(project._ran)
        ok_(project.output().exists())

        project = DummyProject(param1=10, name='3')
        project.update = True
        run_result = luigi.build([project], worker_scheduler_factory=DummyFactory())
        project.output().remove()
        ok_(run_result)
        ok_(project._ran)
