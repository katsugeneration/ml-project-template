# Copyright 2019 Katsuya Shimabukuro. All rights reserved.
# Licensed under the MIT License.
import pathlib
import shutil
from collections import OrderedDict
from nose.tools import ok_, eq_
import yaml
import luigi
import mlflow
from projects.run_once import RunOnceProject
from tests.utils.dummy_schedular import DummyFactory


CONF_PATH = 'tests/params.yaml'


def _save_conf(model, dataset, preprocess):
    yaml.add_representer(
        OrderedDict,
        lambda dumper, instance: dumper.represent_mapping('tag:yaml.org,2002:map', instance.items()))
    with open(CONF_PATH, 'w') as f:
        yaml.dump({
            'model': model,
            'dataset': dataset,
            'preprocess': preprocess}, f)

def setup(self):
    mlflow.set_tracking_uri('file://' + str(pathlib.Path('./testrun').absolute()))

def teardown(self):
    if pathlib.Path('./testrun').exists():
        shutil.rmtree('./testrun')
    pathlib.Path(CONF_PATH).unlink()

class TestRunOnceProject(object):
    def test_run_once(self):
        _save_conf({'epochs': 1}, {}, {})
        project = RunOnceProject(**{
            'runner': 'image_recognition_trainer',
            'param_path': CONF_PATH,
        })
        run_result = luigi.build([project], worker_scheduler_factory=DummyFactory())
        ok_(run_result)
        ok_(project.output().exists())
        ok_(project.artifact_directory.joinpath('model.h5').exists())

    def test_run_once_add_preprocess(self):
        _save_conf(
            {
                'epochs': 1
            },
            {
                'data_path': '{{ before_artifact_directory }}'
            },
            {
                'projects': OrderedDict(**{
                    'Download': 'dataset.mnist_from_raw.download_data',
                    'Decompose': 'dataset.mnist_from_raw.decompose_data',
                })
            })
        project = RunOnceProject(**{
            'runner': 'image_recognition_trainer',
            'dataset': 'mnistraw',
            'param_path': CONF_PATH,
        })
        run_result = luigi.build([project], worker_scheduler_factory=DummyFactory())
        ok_(run_result)
        ok_(project.output().exists())
        ok_(project.artifact_directory.joinpath('model.h5').exists())

    def test_run_once_with_before_preprocess(self):
        _save_conf(
            {
                'epochs': 1
            },
            {
                'data_path': '{{ search_preprocess_directory("dataset.mnist_from_raw.decompose_data", {}) }}'
            },
            {
                'projects': OrderedDict(**{
                    'Download': 'dataset.mnist_from_raw.download_data',
                    'Decompose': 'dataset.mnist_from_raw.decompose_data',
                })
            })
        project = RunOnceProject(**{
            'runner': 'image_recognition_trainer',
            'dataset': 'mnistraw',
            'param_path': CONF_PATH,
        })
        run_result = luigi.build([project], worker_scheduler_factory=DummyFactory())
        ok_(run_result)
        ok_(project.output().exists())
        ok_(project.artifact_directory.joinpath('model.h5').exists())
