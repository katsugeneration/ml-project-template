# Copyright 2019 Katsuya Shimabukuro. All rights reserved.
# Licensed under the MIT License.
import os
from nose.tools import ok_, eq_
import luigi
from projects.data import create_data_prepare
from tests.utils.dummy_schedular import DummyFactory


class TestProjectsData(object):
    def test_create_data_prepare(self):
        do_test1 = False
        a_value = None

        def test1(a):
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

        def test1(a):
            nonlocal do_test1
            nonlocal a_value
            do_test1 = True
            a_value = a

        def test2(b):
            nonlocal do_test2
            nonlocal b_value
            do_test2 = True
            b_value = b

        project = create_data_prepare(
                        {'Test1': test1, 'Test2': test2},
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
