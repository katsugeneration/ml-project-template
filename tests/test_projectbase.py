# Copyright 2019 Katsuya Shimabukuro. All rights reserved.
# Licensed under the MIT License.
import os
from nose.tools import ok_, eq_
import luigi
from projects.base import ProjectBase


class DummyWorker(luigi.worker.Worker):
    pass


class DummyFactory(object):
  def create_local_scheduler(self):
      return luigi.scheduler.Scheduler(prune_on_get_work=True, record_task_history=False)

  def create_remote_scheduler(self, url):
      return None

  def create_worker(self, scheduler, worker_processes, assistant=False):
      # return your worker instance
      return DummyWorker(
          scheduler=scheduler, worker_processes=worker_processes, assistant=assistant)


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
