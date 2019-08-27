# Copyright 2019 Katsuya Shimabukuro. All rights reserved.
# Licensed under the MIT License.
import luigi


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