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


class DumDummyProject(ProjectBase):

    param1 = luigi.IntParameter()

    def __init__(
            self,
            *args,
            **kwargs):
        super(DumDummyProject, self).__init__(*args, **kwargs)
        self.experiment_id = 0
        self.parameters = {
            "param1": self.param1,
            "param2": 2
        }
        self._ran = False

    def _run(self):
        self._ran = True


class TestProjectBase(object):
    def test_init(self):
        project = DumDummyProject(param1=10)
        eq_(project.parameters, {"param1": 10, "param2": 2})

    def test_run(self):
        project = DumDummyProject(param1=10)
        run_result = luigi.build([project], worker_scheduler_factory=DummyFactory())
        ok_(run_result)
        ok_(project._ran)
        ok_(os.path.exists(project.output()[0]))
