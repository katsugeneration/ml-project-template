# Copyright 2019 Katsuya Shimabukuro. All rights reserved.
# Licensed under the MIT License.
import pathlib
from nose.tools import ok_, eq_
import luigi
from projects.data import create_data_prepare
from tests.utils.dummy_schedular import DummyFactory


class TestProjectsData(object):
    def test_create_data_prepare(self):
        do_test1 = False
        a_value = None

        def test1(a, artifact_directory=None, before_artifact_directory=None):
            nonlocal do_test1
            nonlocal a_value
            do_test1 = True
            a_value = a

        project = create_data_prepare(
                        {'Test1': test1},
                        {'a': 1, 'b': 2})
        run_result = luigi.build([project], worker_scheduler_factory=DummyFactory())
        ok_(run_result)
        ok_(project.output().exists())
        project.output().remove()
        ok_(do_test1)
        eq_(a_value, 1)

    def test_create_data_prepare_has_two(self):
        do_test1 = False
        a_value = None
        do_test2 = False
        b_value = None

        def test1(a, artifact_directory=None, before_artifact_directory=None):
            nonlocal do_test1
            nonlocal a_value
            do_test1 = True
            a_value = a

        def test2(b, artifact_directory=None, before_artifact_directory=None):
            nonlocal do_test2
            nonlocal b_value
            do_test2 = True
            b_value = b

        project = create_data_prepare(
                        {'Test2': test2, 'Test1': test1},
                        {'a': 1, 'b': 2})
        run_result = luigi.build([project], worker_scheduler_factory=DummyFactory())
        ok_(run_result)
        ok_(project.output().exists())
        project.output().remove()
        ok_(project.requires()[0].output().exists())
        project.requires()[0].output().remove()
        ok_(do_test1)
        eq_(a_value, 1)
        ok_(do_test2)
        eq_(b_value, 2)

    def test_create_data_prepare_save(self):
        do_test1 = False
        a_value = None

        def test1(a, artifact_directory=None, before_artifact_directory=None):
            nonlocal do_test1
            nonlocal a_value
            do_test1 = True
            a_value = a
            artifact_directory.joinpath('aaa').touch()

        project = create_data_prepare(
                        {'Test1': test1},
                        {'a': 1, 'b': 2})
        run_result = luigi.build([project], worker_scheduler_factory=DummyFactory())
        ok_(run_result)
        ok_(project.output().exists())
        ok_(project.artifact_directory.joinpath('aaa').exists())
        project.output().remove()
        ok_(do_test1)
        eq_(a_value, 1)

    def test_create_data_prepare_save_and_load(self):
        do_test1 = False
        a_value = None
        before_value = None
        do_test2 = False
        b_value = None

        def test1(a, artifact_directory=None, before_artifact_directory=None):
            nonlocal do_test1
            nonlocal a_value
            nonlocal before_value
            do_test1 = True
            a_value = a
            with before_artifact_directory.joinpath('aaa').open('r') as f:
                before_value = f.read()

        def test2(b, artifact_directory=None, before_artifact_directory=None):
            nonlocal do_test2
            nonlocal b_value
            do_test2 = True
            b_value = b
            with artifact_directory.joinpath('aaa').open('w') as f:
                f.write('aaa')

        project = create_data_prepare(
                        {'Test2': test2, 'Test1': test1},
                        {'a': 1, 'b': 2})
        run_result = luigi.build([project], worker_scheduler_factory=DummyFactory())
        ok_(run_result)
        ok_(project.output().exists())
        project.output().remove()
        ok_(project.requires()[0].output().exists())
        project.requires()[0].output().remove()
        ok_(do_test1)
        eq_(a_value, 1)
        ok_(do_test2)
        eq_(b_value, 2)
        eq_(before_value, 'aaa')
